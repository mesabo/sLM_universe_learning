"""Course 2 / ch1 / class 1 — measure catastrophic forgetting on AG News -> TREC.

Pipeline:
  1. Load both tasks with a shared 4-output classification head.
  2. Stage 0: evaluate on each task BEFORE any training.
  3. Stage 1: train on Task A (AG News). Re-evaluate both.
  4. Stage 2: continue training on Task B (TREC). Re-evaluate both.
  5. Build a continual.History; compute BWT / FWT / avg_accuracy.

The metric band asserts that BWT is meaningfully negative — i.e. the
forgetting *did* happen, so the lesson reproduces.
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
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
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
        # Remap labels and drop rows whose label was mapped to null.
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
    """Evaluate the *current* model on one task's eval split. Returns accuracy."""
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
    """Train (in place) on one task for `limits[mode].epochs_per_task` epochs."""
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
        save_strategy="epoch",
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
    log = get_logger("course2.ch1.class1")
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
    log.info("trainable=%d total=%d ratio=%.4f", trainable, total, trainable / total)

    history = History()

    # --- Stage 0: pre-training evaluation -----------------------------------
    log.info("=== STAGE 0 (before any training) ===")
    history.add_stage(0, {i: _evaluate_task(model, tokenizer, t, cfg, log)
                          for i, t in enumerate(tasks)})

    # --- Stage 1+: sequential train then eval-all ---------------------------
    for j, task in enumerate(tasks):
        log.info("=== STAGE %d (after training task %d = %s) ===", j + 1, j, task.name)
        _train_on_task(model, tokenizer, task, cfg, log)
        history.add_stage(j + 1, {i: _evaluate_task(model, tokenizer, t, cfg, log)
                                  for i, t in enumerate(tasks)})

    # --- Compute metrics ----------------------------------------------------
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
            "trainable_params": trainable,
            "total_params": total,
            "mode": cfg["mode"],
        },
    )


def _format_matrix(history: History) -> str:
    """Pretty-print the accuracy matrix for the log."""
    rows = history.to_matrix()
    header = " " * 8 + "".join(f"  task{i:>2}" for i in range(history.n_tasks()))
    lines = [header]
    for k, row in enumerate(rows):
        cells = "  ".join(f"{(v if v is not None else float('nan')):.4f}" for v in row)
        lines.append(f"stage{k:>2}  {cells}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
