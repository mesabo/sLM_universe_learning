"""Evaluation-only entrypoint: loads the latest checkpoint and re-runs eval.

For this class the train script already evaluates and writes a result JSON.
This file exists for symmetry — call it after `train.py` to recompute
metrics from a different checkpoint or split.
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

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from shared.config import apply_overrides, load_yaml
from shared.datasets import load_spec, to_classification
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.repro import set_seed
from shared.training import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to a saved checkpoint dir")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = get_logger("course1.ch1.class1.eval")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])

    spec = load_spec({k: v for k, v in cfg["dataset"].items() if k != "num_labels"})
    splits = to_classification(spec)
    limits = cfg["limits"][cfg["mode"]]
    if limits["eval"]:
        splits["eval"] = splits["eval"].shuffle(seed=cfg["seed"]).select(range(limits["eval"]))

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)

    splits["eval"] = splits["eval"].map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=cfg["train"]["max_len"]),
        batched=True, remove_columns=["text"],
    )

    args_tr = TrainingArguments(
        output_dir=str(args.ckpt) + "_eval",
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        report_to=[],
    )

    def _compute(p):
        return classification_metrics(np.argmax(p.predictions, axis=-1), p.label_ids)

    trainer = Trainer(model=model, args=args_tr,
                      eval_dataset=splits["eval"],
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=_compute)
    out = trainer.evaluate()
    metrics = {"accuracy": float(out["eval_accuracy"]), "f1_macro": float(out["eval_f1_macro"])}
    log.info("metrics=%s", metrics)

    run_eval(
        method=cfg["method"] + "-rerun",
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
    )


if __name__ == "__main__":
    main()
