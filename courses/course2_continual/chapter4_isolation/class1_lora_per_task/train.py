"""Course 2 / ch4 / class 1 — parameter isolation: LoRA-per-task.

Two adapters (one per task) over a frozen shared backbone. PEFT's
`modules_to_save=["classifier"]` makes the classification head per-adapter
too. At eval time we `set_adapter(name)` to swap in the right (LoRA + head)
combination for each task.

By construction the adapters are disjoint, so BWT should be ~0 — there's
no mechanism for Task B's training to overwrite Task A's parameters.
"""

from __future__ import annotations


# --- ensure repo root is importable when invoked via `python <path>/train.py` ---
import sys as _sys, pathlib as _pathlib
_root = _pathlib.Path(__file__).resolve()
for _p in [_root.parent, *_root.parents]:
    if (_p / "pyproject.toml").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
del _sys, _pathlib, _root, _p
# --- end shim ---

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from shared.config import apply_overrides, load_yaml
from shared.continual import History, summarize
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import hf_cache
from shared.repro import set_seed
from shared.training import (
    classification_metrics,
    lora_target_modules,
    make_output_dir,
    trainable_param_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


@dataclass
class TaskSplits:
    name: str
    train: Dataset
    eval: Dataset


def _load_one_task(spec: dict, cap_train: int | None, cap_eval: int | None,
                   seed: int, log) -> TaskSplits:
    hf_id = spec["hf_id"]
    cfg_name = spec.get("config")
    split = spec["split"]
    text_field = spec["text_field"]
    label_field = spec["label_field"]
    remap: dict[int, int | None] = {int(k): v for k, v in spec["label_remap"].items()}

    def _load_split(split_name: str, cap: int | None) -> Dataset:
        ds = load_dataset(hf_id, cfg_name, split=split_name) if cfg_name \
            else load_dataset(hf_id, split=split_name)

        def _remap_row(row):
            new_label = remap.get(int(row[label_field]))
            row["_keep"] = new_label is not None
            row["_label"] = new_label if new_label is not None else 0
            return row

        ds = ds.map(_remap_row)
        ds = ds.filter(lambda r: r["_keep"])
        ds = ds.remove_columns([c for c in ds.column_names
                                if c not in {text_field, "_label"}])
        ds = ds.rename_columns({text_field: "text", "_label": "label"})
        ds = ds.shuffle(seed=seed)
        if cap is not None:
            ds = ds.select(range(min(cap, len(ds))))
        return ds

    train = _load_split(split["train"], cap_train)
    ev = _load_split(split["eval"], cap_eval)
    log.info("[task %s] train=%d eval=%d (after remap)", spec["name"], len(train), len(ev))
    return TaskSplits(name=spec["name"], train=train, eval=ev)


def _tokenize(ds: Dataset, tokenizer, max_len: int) -> Dataset:
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_len, padding=False),
        batched=True,
        remove_columns=["text"],
    )


def _evaluate_task(model, tokenizer, task: TaskSplits, cfg: dict, log) -> float:
    """Evaluate the *currently-active* adapter+head combination on one task."""
    eval_args = TrainingArguments(
        output_dir=str(make_output_dir(
            course=cfg["course"], klass=cfg["class_id"],
            backbone=cfg["backbone"], method=cfg["method"], run_tag=f"eval_{task.name}",
        )),
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        fp16=cfg["train"]["fp16"] and torch.cuda.is_available(),
        report_to=[],
    )
    tokenized = _tokenize(task.eval, tokenizer, cfg["train"]["max_len"])

    def _compute(p):
        return classification_metrics(np.argmax(p.predictions, axis=-1), p.label_ids)

    trainer = Trainer(
        model=model, args=eval_args,
        eval_dataset=tokenized,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute,
    )
    out = trainer.evaluate()
    acc = float(out["eval_accuracy"])
    log.info("[eval %s] accuracy=%.4f", task.name, acc)
    return acc


def _train_on_task(model, tokenizer, task: TaskSplits, cfg: dict, log) -> None:
    """Train (in place) on one task — only the currently-active adapter updates."""
    output_dir = make_output_dir(
        course=cfg["course"], klass=cfg["class_id"],
        backbone=cfg["backbone"], method=cfg["method"], run_tag=f"train_{task.name}",
    )
    tokenized = _tokenize(task.train, tokenizer, cfg["train"]["max_len"])
    epochs = cfg["limits"][cfg["mode"]]["epochs_per_task"]

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        fp16=cfg["train"]["fp16"] and torch.cuda.is_available(),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=cfg["train"]["log_steps"],
        seed=cfg["seed"],
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    log.info("[train %s] starting (%d rows, %d epochs)", task.name, len(tokenized), epochs)
    trainer.train()


def main() -> None:
    args = parse_args()
    log = get_logger("course2.ch4.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    limits = cfg["limits"][cfg["mode"]]
    tasks = [
        _load_one_task(t, limits["train_per_task"], limits["eval_per_task"], cfg["seed"], log)
        for t in cfg["tasks"]
    ]
    if len(tasks) != 2:
        raise ValueError("This class is hard-wired for exactly 2 tasks.")
    task_a, task_b = tasks

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["backbone"], num_labels=cfg["num_labels"]
    )

    target = cfg["lora"]["target_modules"] or lora_target_modules(base)
    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=target,
        modules_to_save=cfg["lora"]["modules_to_save"],
        bias=cfg["lora"]["bias"],
        task_type="SEQ_CLS",
    )
    log.info("LoRA target_modules=%s modules_to_save=%s",
             target, cfg["lora"]["modules_to_save"])

    history = History()

    log.info("=== STAGE 0 (before any training, no adapter active) ===")
    history.add_stage(0, {i: _evaluate_task(base, tokenizer, t, cfg, log)
                          for i, t in enumerate(tasks)})

    log.info("=== STAGE 1: wrap with adapter '%s' and train task A ===", task_a.name)
    model = get_peft_model(base, lora_config, adapter_name=task_a.name)
    model.set_adapter(task_a.name)
    trainable, total = trainable_param_count(model)
    ratio_pct = 100.0 * trainable / total
    log.info("trainable=%d total=%d ratio=%.4f%% (one adapter active)",
             trainable, total, ratio_pct)
    _train_on_task(model, tokenizer, task_a, cfg, log)

    # Eval each task with the only adapter that exists right now.
    history.add_stage(1, {i: _evaluate_task(model, tokenizer, t, cfg, log)
                          for i, t in enumerate(tasks)})

    log.info("=== STAGE 2: add adapter '%s' and train task B ===", task_b.name)
    model.add_adapter(task_b.name, lora_config)
    model.set_adapter(task_b.name)
    log.info("now has adapters: %s; active=%s",
             list(model.peft_config.keys()), model.active_adapter)
    _train_on_task(model, tokenizer, task_b, cfg, log)

    # Eval each task with its OWN adapter — the central point of this lesson.
    log.info("=== STAGE 2 eval: swapping adapter per task ===")
    stage2 = {}
    for i, t in enumerate(tasks):
        model.set_adapter(t.name)
        log.info("active adapter -> %s (for task %s)", model.active_adapter, t.name)
        stage2[i] = _evaluate_task(model, tokenizer, t, cfg, log)
    history.add_stage(2, stage2)

    summary = summarize(history)
    log.info("history matrix:\n%s", _format_matrix(history))
    log.info("summary: %s", summary)

    metrics = {
        "avg_accuracy": float(summary["avg_accuracy"]),
        "acc_A_after_A": float(history.acc(1, 0)),
        "acc_A_after_B": float(history.acc(2, 0)),
        "acc_B_after_A": float(history.acc(1, 1)),
        "acc_B_after_B": float(history.acc(2, 1)),
        "BWT": float(summary["BWT"]),
        "n_adapters": len(model.peft_config),
        "trainable_ratio_pct": ratio_pct,
    }

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "history_matrix": history.to_matrix(),
            "FWT": float(summary["FWT"]),
            "task_names": [t.name for t in tasks],
            "adapter_names": list(model.peft_config.keys()),
            "lora_target_modules": target,
            "trainable_params": trainable,
            "total_params": total,
            "mode": cfg["mode"],
        },
    )


def _format_matrix(history: History) -> str:
    rows = history.to_matrix()
    header = " " * 8 + "".join(f"  task{i:>2}" for i in range(history.n_tasks()))
    lines = [header]
    for k, row in enumerate(rows):
        cells = "  ".join(f"{(v if v is not None else float('nan')):.4f}" for v in row)
        lines.append(f"stage{k:>2}  {cells}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
