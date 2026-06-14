"""Standalone evaluator for course2/chapter2/class1 — experience replay.

Usage:
  python eval.py --config configs/smoke.yaml
  python eval.py --config configs/smoke.yaml --checkpoint /path/to/checkpoint

If --checkpoint points to a directory that exists, the model is loaded from it
and evaluated on all task eval splits. If the checkpoint is absent, eval falls
back to re-running train.py (which trains and then evaluates), with a warning.
"""

from __future__ import annotations


# --- ensure repo root is importable when invoked via `python <path>/eval.py` ---
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
import runpy
import sys
from pathlib import Path

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
from shared.logging_utils import get_logger
from shared.paths import hf_cache
from shared.training import classification_metrics, make_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a saved checkpoint directory")
    args, extra = parser.parse_known_args()
    return args, extra


def _tokenize(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_length, padding=False),
        batched=True,
        remove_columns=["text"],
    )


def _eval_one(model, tokenizer, task_spec: dict, cfg: dict, log) -> float:
    ds_name = task_spec["dataset"]
    text_col = task_spec["text_col"]
    label_col = task_spec["label_col"]
    n_eval = task_spec["eval_n"]

    try:
        eval_ds = load_dataset(ds_name, split="validation")
    except Exception:
        try:
            eval_ds = load_dataset(ds_name, split="test")
        except Exception:
            eval_ds = load_dataset(ds_name, split="train")

    eval_ds = eval_ds.shuffle(seed=cfg["seed"]).select(range(min(n_eval, len(eval_ds))))
    eval_ds = Dataset.from_dict({
        "text": [row[text_col] for row in eval_ds],
        "label": [int(row[label_col]) for row in eval_ds],
    })
    tokenized = _tokenize(eval_ds, tokenizer, cfg["training"]["max_length"])

    eval_args = TrainingArguments(
        output_dir=str(make_output_dir(
            course=cfg["course"], klass=cfg["class_id"],
            backbone=cfg["backbone"], method=cfg["method"],
            run_tag=f"eval_{task_spec['name']}",
        )),
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        report_to=[],
    )

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
    log.info("[eval %s] accuracy=%.4f", task_spec["name"], acc)
    return acc


def main() -> None:
    args, extra = _parse_args()
    log = get_logger("course2.ch2.eval")

    cfg = apply_overrides(load_yaml(args.config), extra)
    import os
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    ckpt = Path(args.checkpoint) if args.checkpoint else None

    if ckpt and ckpt.exists():
        log.info("Evaluating from checkpoint", checkpoint=str(ckpt))
        num_labels = max(spec["n_labels"] for spec in cfg["tasks"])
        tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt), num_labels=num_labels
        )
        model.eval()

        results = {}
        for spec in cfg["tasks"]:
            results[spec["name"]] = _eval_one(model, tokenizer, spec, cfg, log)

        log.info("Checkpoint eval results: %s", results)
    else:
        if args.checkpoint:
            log.warning("Checkpoint not found, falling back to train.py", checkpoint=str(ckpt))
        else:
            log.warning("No checkpoint provided, running full train+eval via train.py")

        here = Path(__file__).parent
        sys.argv = ["train.py", "--config", args.config] + extra
        runpy.run_path(str(here / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
