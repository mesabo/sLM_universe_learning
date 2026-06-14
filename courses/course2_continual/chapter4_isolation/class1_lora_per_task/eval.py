"""Evaluator for Course 2 ch4 class 1 — LoRA-per-task isolation.

Usage:
  python eval.py --config configs/default.yaml
  python eval.py --config configs/default.yaml --checkpoint /path/to/checkpoint

If --checkpoint points to an existing directory, loads the PEFT model from it
and evaluates on all task eval splits. If absent, falls back to re-running
train.py with a warning.
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

from shared.logging_utils import get_logger


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a saved PEFT checkpoint directory")
    return parser.parse_known_args()


def main() -> None:
    args, extra = _parse_args()
    log = get_logger("course2.ch4.eval")

    ckpt = Path(args.checkpoint) if args.checkpoint else None

    if ckpt and ckpt.exists():
        import os
        import numpy as np
        from datasets import load_dataset
        from peft import PeftModel
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
        from shared.config import apply_overrides, load_yaml
        from shared.paths import hf_cache
        from shared.training import classification_metrics, make_output_dir

        log.info("Evaluating from checkpoint", checkpoint=str(ckpt))
        cfg = apply_overrides(load_yaml(args.config), extra)
        os.environ.setdefault("HF_HOME", str(hf_cache()))

        tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForSequenceClassification.from_pretrained(
            cfg["backbone"], num_labels=cfg["num_labels"]
        )
        model = PeftModel.from_pretrained(base, str(ckpt))
        model.eval()

        for task_spec in cfg["tasks"]:
            hf_id = task_spec["hf_id"]
            split = task_spec["split"].get("eval", "test")
            text_field = task_spec["text_field"]
            label_field = task_spec["label_field"]
            remap = {int(k): v for k, v in task_spec["label_remap"].items()}
            limits = cfg["limits"][cfg.get("mode", "smoke")]
            cap = limits.get("eval_per_task")

            ds = load_dataset(hf_id, task_spec.get("config"), split=split)

            def _remap(row):
                new_label = remap.get(int(row[label_field]))
                row["_keep"] = new_label is not None
                row["_label"] = new_label if new_label is not None else 0
                return row

            ds = ds.map(_remap)
            ds = ds.filter(lambda r: r["_keep"])
            ds = ds.remove_columns([c for c in ds.column_names
                                    if c not in {text_field, "_label"}])
            ds = ds.rename_columns({text_field: "text", "_label": "label"})
            ds = ds.shuffle(seed=cfg["seed"])
            if cap:
                ds = ds.select(range(min(cap, len(ds))))

            tokenized = ds.map(
                lambda b: tokenizer(b["text"], truncation=True,
                                    max_length=cfg["train"]["max_len"], padding=False),
                batched=True, remove_columns=["text"],
            )
            eval_args = TrainingArguments(
                output_dir=str(make_output_dir(
                    course=cfg["course"], klass=cfg["class_id"],
                    backbone=cfg["backbone"], method=cfg["method"],
                    run_tag=f"eval_{task_spec['name']}",
                )),
                per_device_eval_batch_size=cfg["train"]["per_device_batch"],
                report_to=[],
            )

            def _compute(p):
                return classification_metrics(
                    np.argmax(p.predictions, axis=-1), p.label_ids
                )

            trainer = Trainer(
                model=model, args=eval_args,
                eval_dataset=tokenized,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=_compute,
            )
            out = trainer.evaluate()
            log.info("[eval %s] accuracy=%.4f", task_spec["name"],
                     float(out["eval_accuracy"]))
    else:
        if args.checkpoint:
            log.warning("Checkpoint not found, falling back to train.py",
                        checkpoint=str(ckpt))
        else:
            log.warning("No checkpoint provided, running full train+eval via train.py")
        here = Path(__file__).parent
        sys.argv = ["train.py", "--config", args.config] + extra
        runpy.run_path(str(here / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
