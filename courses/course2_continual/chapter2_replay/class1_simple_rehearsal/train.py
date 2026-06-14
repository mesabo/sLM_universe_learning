"""Course 2 / ch2 / class 1 — Experience Replay with simple rehearsal.

Pipeline:
  1. Train Task 0 (AG News) normally; populate the replay buffer with random samples.
  2. For Task 1+ (Emotion), mix replay examples (mixing_ratio fraction) with the
     current task's examples before training. Evaluate ALL tasks after each task.
  3. Compute BWT = mean(acc_after_all[i] - acc_after_task_i[i]) for i in 0..T-2.

Lesson: a bounded replay buffer dramatically reduces catastrophic forgetting
at the cost of a small amount of memory (capacity * avg_seq_len * 4 bytes).
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
import random
from dataclasses import dataclass, field
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
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import hf_cache
from shared.repro import set_seed
from shared.training import (
    classification_metrics,
    make_output_dir,
    trainable_param_count,
)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-capacity buffer with reservoir sampling.

    Reservoir sampling ensures every seen example has equal probability
    min(1, capacity / n_seen) of being retained.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buf: list[dict] = []
        self._n_seen: int = 0

    def add(self, examples: list[dict]) -> None:
        """Insert examples via reservoir sampling (Vitter 1985)."""
        for ex in examples:
            self._n_seen += 1
            if len(self._buf) < self._capacity:
                self._buf.append(ex)
            else:
                idx = random.randint(0, self._n_seen - 1)
                if idx < self._capacity:
                    self._buf[idx] = ex

    def sample(self, n: int) -> list[dict]:
        """Return up to n random examples from the buffer (no replacement)."""
        if not self._buf:
            return []
        return random.sample(self._buf, min(n, len(self._buf)))

    def fill_rate(self) -> float:
        """Fraction of capacity occupied (0.0 .. 1.0)."""
        if self._capacity == 0:
            return 1.0
        return min(1.0, len(self._buf) / self._capacity)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


@dataclass
class TaskData:
    name: str
    train_examples: list[dict]   # raw dicts with text_col + label_col
    eval_examples: list[dict]
    n_labels: int
    text_col: str
    label_col: str


def _load_task(spec: dict, seed: int, log) -> TaskData:
    """Load a task from HF datasets; return raw example dicts."""
    ds_name = spec["dataset"]
    split = spec["split"]
    n_train = spec["n_samples"]
    n_eval = spec["eval_n"]
    text_col = spec["text_col"]
    label_col = spec["label_col"]
    n_labels = spec["n_labels"]

    train_ds = load_dataset(ds_name, split=split).shuffle(seed=seed)
    train_ds = train_ds.select(range(min(n_train, len(train_ds))))

    # Eval split: use "validation" if available, else carve from "test" or "train".
    try:
        eval_ds = load_dataset(ds_name, split="validation").shuffle(seed=seed)
    except Exception:
        try:
            eval_ds = load_dataset(ds_name, split="test").shuffle(seed=seed)
        except Exception:
            eval_ds = load_dataset(ds_name, split="train").shuffle(seed=seed + 1)
    eval_ds = eval_ds.select(range(min(n_eval, len(eval_ds))))

    # Keep only the relevant columns as plain dicts.
    def _to_dict(ds: Dataset) -> list[dict]:
        return [{text_col: row[text_col], label_col: int(row[label_col])} for row in ds]

    train_examples = _to_dict(train_ds)
    eval_examples = _to_dict(eval_ds)

    log.info("[task %s] train=%d eval=%d n_labels=%d",
             spec["name"], len(train_examples), len(eval_examples), n_labels)
    return TaskData(
        name=spec["name"],
        train_examples=train_examples,
        eval_examples=eval_examples,
        n_labels=n_labels,
        text_col=text_col,
        label_col=label_col,
    )


def _examples_to_dataset(examples: list[dict], text_col: str, label_col: str) -> Dataset:
    """Convert a list of raw dicts into a HF Dataset with 'text' and 'label'."""
    texts = [ex[text_col] for ex in examples]
    labels = [ex[label_col] for ex in examples]
    return Dataset.from_dict({"text": texts, "label": labels})


def _tokenize(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_length, padding=False),
        batched=True,
        remove_columns=["text"],
    )


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------


def _evaluate_task(model, tokenizer, task: TaskData, cfg: dict, log) -> float:
    """Evaluate current model on one task's eval split. Returns accuracy."""
    eval_args = TrainingArguments(
        output_dir=str(make_output_dir(
            course=cfg["course"], klass=cfg["class_id"],
            backbone=cfg["backbone"], method=cfg["method"], run_tag=f"eval_{task.name}",
        )),
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        report_to=[],
    )
    eval_ds = _examples_to_dataset(task.eval_examples, task.text_col, task.label_col)
    tokenized = _tokenize(eval_ds, tokenizer, cfg["training"]["max_length"])

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


def _train_step(model, tokenizer, examples: list[dict],
                text_col: str, label_col: str, cfg: dict, tag: str, log) -> None:
    """Run one Trainer.train() call on the given examples list."""
    output_dir = make_output_dir(
        course=cfg["course"], klass=cfg["class_id"],
        backbone=cfg["backbone"], method=cfg["method"], run_tag=f"train_{tag}",
    )
    ds = _examples_to_dataset(examples, text_col, label_col)
    tokenized = _tokenize(ds, tokenizer, cfg["training"]["max_length"])

    train_cfg = cfg["training"]
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["max_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["lr"],
        warmup_ratio=train_cfg["warmup_ratio"],
        eval_strategy="no",
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=cfg["seed"],
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    log.info("[train %s] %d rows, %d epoch(s)", tag, len(tokenized), train_cfg["max_epochs"])
    trainer.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    log = get_logger("course2.ch2.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    tasks = [_load_task(spec, cfg["seed"], log) for spec in cfg["tasks"]]
    n_tasks = len(tasks)

    # Use the maximum number of labels across all tasks for the shared head.
    num_labels = max(t.n_labels for t in tasks)
    log.info("num_labels=%d (shared head across %d tasks)", num_labels, n_tasks)

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["backbone"], num_labels=num_labels
    )
    trainable, total = trainable_param_count(model)
    log.info("trainable=%d total=%d ratio=%.4f", trainable, total, trainable / total)

    replay_cfg = cfg["replay"]
    buffer = ReplayBuffer(capacity=replay_cfg["buffer_capacity"])
    mixing_ratio = float(replay_cfg["mixing_ratio"])

    # acc_matrix[stage][task_idx] — stage 0 = before any training
    # stage k+1 = after training task k
    acc_matrix: list[list[float | None]] = [[None] * n_tasks for _ in range(n_tasks + 1)]

    # Stage 0: evaluate everything before any training.
    log.info("=== STAGE 0 (before any training) ===")
    for i, t in enumerate(tasks):
        acc_matrix[0][i] = _evaluate_task(model, tokenizer, t, cfg, log)

    for task_idx, task in enumerate(tasks):
        log.info("=== STAGE %d (training task %d = %s) ===", task_idx + 1, task_idx, task.name)

        if task_idx == 0:
            # First task: train normally on task data only.
            train_examples = task.train_examples
        else:
            # Subsequent tasks: mix current examples with replay samples.
            n_current = len(task.train_examples)
            n_replay = int(n_current * mixing_ratio / (1.0 - mixing_ratio + 1e-9))
            replay_samples = buffer.sample(n_replay)
            # Replay examples use the same text/label keys as current task.
            # We standardize both to "text" / "label" in _examples_to_dataset,
            # so mix must be in the same format.
            current_std = [{"text": ex[task.text_col], "label": ex[task.label_col]}
                           for ex in task.train_examples]
            replay_std = [{"text": ex.get("text", ex.get(tasks[0].text_col, "")),
                           "label": ex.get("label", ex.get(tasks[0].label_col, 0))}
                          for ex in replay_samples]
            train_examples_raw = current_std + replay_std
            log.info("[replay] current=%d replay=%d total=%d fill_rate=%.3f",
                     len(current_std), len(replay_std), len(train_examples_raw),
                     buffer.fill_rate())
            # Shuffle the mixed set.
            random.shuffle(train_examples_raw)
            # Use generic "text"/"label" keys for the mixed dataset.
            _train_step(model, tokenizer, train_examples_raw, "text", "label", cfg,
                        f"task{task_idx}_mixed", log)
            # Add current task examples to buffer (reservoir sampling).
            buffer_entries = [{"text": ex[task.text_col], "label": ex[task.label_col]}
                              for ex in task.train_examples]
            buffer.add(buffer_entries)
            # Evaluate all tasks after training.
            for i, t in enumerate(tasks):
                acc_matrix[task_idx + 1][i] = _evaluate_task(model, tokenizer, t, cfg, log)
            continue

        # Task 0 path: train then seed the buffer.
        _train_step(model, tokenizer,
                    [{"text": ex[task.text_col], "label": ex[task.label_col]}
                     for ex in train_examples],
                    "text", "label", cfg, f"task{task_idx}", log)
        buffer_entries = [{"text": ex[task.text_col], "label": ex[task.label_col]}
                          for ex in task.train_examples]
        buffer.add(buffer_entries)
        log.info("[buffer] seeded with %d examples, fill_rate=%.3f",
                 len(buffer_entries), buffer.fill_rate())

        # Evaluate all tasks after training task 0.
        for i, t in enumerate(tasks):
            acc_matrix[task_idx + 1][i] = _evaluate_task(model, tokenizer, t, cfg, log)

    # ---------------------------------------------------------------------------
    # Compute BWT = mean(acc_after_all_tasks[i] - acc_after_task_i[i])
    # for i in 0 .. T-2
    # ---------------------------------------------------------------------------
    final_stage = n_tasks  # index of the last row in acc_matrix
    bwt_values = []
    for i in range(n_tasks - 1):
        acc_after_i = acc_matrix[i + 1][i]   # accuracy on task i right after training task i
        acc_final_i = acc_matrix[final_stage][i]  # accuracy on task i at the very end
        if acc_after_i is not None and acc_final_i is not None:
            bwt_values.append(acc_final_i - acc_after_i)

    bwt = float(np.mean(bwt_values)) if bwt_values else float("nan")
    final_acc = acc_matrix[final_stage][n_tasks - 1]
    if final_acc is None:
        final_acc = float("nan")
    fill_rate = buffer.fill_rate()

    log.info("BWT=%.4f final_acc=%.4f buffer_fill_rate=%.4f", bwt, final_acc, fill_rate)
    log.info("accuracy matrix:")
    for stage_idx, row in enumerate(acc_matrix):
        row_str = "  ".join(f"{(v if v is not None else float('nan')):.4f}" for v in row)
        log.info("  stage%d  %s", stage_idx, row_str)

    metrics = {
        "bwt": bwt,
        "final_acc": float(final_acc),
        "buffer_fill_rate": fill_rate,
    }

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"],
        extras={
            "accuracy_matrix": acc_matrix,
            "task_names": [t.name for t in tasks],
            "buffer_capacity": replay_cfg["buffer_capacity"],
            "mixing_ratio": mixing_ratio,
            "trainable_params": trainable,
            "total_params": total,
        },
    )


if __name__ == "__main__":
    main()
