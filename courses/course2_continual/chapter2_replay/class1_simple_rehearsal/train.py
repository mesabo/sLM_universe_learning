"""Course 2 / ch2 / class 1 — simple rehearsal against catastrophic forgetting.

Pipeline mirrors course2/ch1/class1, with one change: when training Task B,
we mix in `int(ratio * |B|)` randomly-sampled rows of Task A. Set ratio=0
to reproduce ch1's catastrophic baseline; set ratio=1 to approximate
joint training (the no-forgetting upper bound).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
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
    freeze_base,
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
    """Load + remap labels for a single task; cap sizes per `mode`."""
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


def _build_replay_train(task_b: TaskSplits, task_a_train: Dataset,
                        ratio: float, seed: int, log) -> Dataset:
    """Concatenate `int(ratio * |B|)` random Task A rows into Task B's train split."""
    n_replay = int(round(ratio * len(task_b.train)))
    if n_replay <= 0:
        log.info("[replay] ratio=%.3f -> no replay; falling back to plain Task B", ratio)
        return task_b.train
    n_replay = min(n_replay, len(task_a_train))
    replay = task_a_train.shuffle(seed=seed + 1).select(range(n_replay))
    mixed = concatenate_datasets([task_b.train, replay]).shuffle(seed=seed + 2)
    log.info("[replay] ratio=%.3f -> mixed |B|=%d + replay=%d -> total=%d",
             ratio, len(task_b.train), n_replay, len(mixed))
    return mixed


def _train_on_dataset(model, tokenizer, train_split: Dataset,
                      tag: str, cfg: dict, log) -> None:
    """Train (in place) on the given dataset for `epochs_per_task` epochs."""
    output_dir = make_output_dir(
        course=cfg["course"], klass=cfg["class_id"],
        backbone=cfg["backbone"], method=cfg["method"], run_tag=tag,
    )
    tokenized = _tokenize(train_split, tokenizer, cfg["train"]["max_len"])
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
    log.info("[train %s] starting (%d rows, %d epochs)", tag, len(tokenized), epochs)
    trainer.train()


def main() -> None:
    args = parse_args()
    log = get_logger("course2.ch2.class1")
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
    ratio = float(cfg["replay"]["ratio"])
    # Auto-derive method tag from ratio so grid sweeps don't collide on the
    # result JSON path. Anything explicitly passed at the CLI wins.
    if cfg.get("method", "").startswith("replay-r"):
        cfg["method"] = f"replay-r{ratio:.2f}"

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["backbone"], num_labels=cfg["num_labels"]
    )
    if cfg["freeze_backbone"]:
        n_frozen = freeze_base(model.base_model)
        log.info("froze %d backbone params (head still trainable)", n_frozen)
    trainable, total = trainable_param_count(model)
    log.info("trainable=%d total=%d ratio=%.4f replay_ratio=%.3f",
             trainable, total, trainable / total, ratio)

    history = History()

    log.info("=== STAGE 0 (before any training) ===")
    history.add_stage(0, {i: _evaluate_task(model, tokenizer, t, cfg, log)
                          for i, t in enumerate(tasks)})

    log.info("=== STAGE 1 (after training task 0 = %s) ===", task_a.name)
    _train_on_dataset(model, tokenizer, task_a.train, f"train_{task_a.name}", cfg, log)
    history.add_stage(1, {i: _evaluate_task(model, tokenizer, t, cfg, log)
                          for i, t in enumerate(tasks)})

    log.info("=== STAGE 2 (after training task 1 = %s + replay r=%.3f) ===",
             task_b.name, ratio)
    mixed_train = _build_replay_train(task_b, task_a.train, ratio, cfg["seed"], log)
    _train_on_dataset(model, tokenizer, mixed_train,
                      f"train_{task_b.name}_replay{int(ratio * 100)}", cfg, log)
    history.add_stage(2, {i: _evaluate_task(model, tokenizer, t, cfg, log)
                          for i, t in enumerate(tasks)})

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
        "replay_ratio": ratio,
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
            "replay_n_samples": int(round(ratio * len(task_b.train))),
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
