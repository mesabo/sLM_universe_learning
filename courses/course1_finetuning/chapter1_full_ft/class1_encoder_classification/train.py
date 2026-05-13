"""Course 1 / ch1 / class 1 — full FT encoder classification on AG News.

Uses HF `Trainer`. Smoke vs full split is config-driven.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.config import apply_overrides, load_yaml
from shared.datasets import load_spec, to_classification
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import hf_cache
from shared.repro import set_seed
from shared.training import (
    classification_metrics,
    make_output_dir,
    trainable_param_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = get_logger("course1.ch1.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])

    os.environ.setdefault("HF_HOME", str(hf_cache()))

    spec = load_spec({k: v for k, v in cfg["dataset"].items() if k != "num_labels"})
    splits = to_classification(spec)
    limits = cfg["limits"][cfg["mode"]]
    if limits["train"]:
        splits["train"] = splits["train"].shuffle(seed=cfg["seed"]).select(range(limits["train"]))
    if limits["eval"]:
        splits["eval"] = splits["eval"].shuffle(seed=cfg["seed"]).select(range(limits["eval"]))
    log.info("dataset sizes: train=%d eval=%d", len(splits["train"]), len(splits["eval"]))

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["backbone"], num_labels=cfg["dataset"]["num_labels"]
    )
    trainable, total = trainable_param_count(model)
    log.info("trainable=%d total=%d ratio=%.4f", trainable, total, trainable / total)

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg["train"]["max_len"],
            padding=False,
        )

    splits = {role: ds.map(_tokenize, batched=True, remove_columns=["text"])
              for role, ds in splits.items()}

    output_dir = make_output_dir(
        course=cfg["course"], klass=cfg["class_id"],
        backbone=cfg["backbone"], method=cfg["method"], run_tag=cfg["mode"],
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        eval_strategy=cfg["train"]["eval_strategy"],
        save_strategy=cfg["train"]["save_strategy"],
        logging_steps=cfg["train"]["log_steps"],
        fp16=cfg["train"]["fp16"] and torch.cuda.is_available(),
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        seed=cfg["seed"],
        report_to=[],
        disable_tqdm=False,
    )

    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return classification_metrics(preds, labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["eval"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute,
    )

    log.info("training...")
    trainer.train()
    log.info("evaluating...")
    eval_metrics = trainer.evaluate()
    metrics = {
        "accuracy": float(eval_metrics["eval_accuracy"]),
        "f1_macro": float(eval_metrics["eval_f1_macro"]),
    }
    log.info("metrics=%s", metrics)

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "trainable_params": trainable,
            "total_params": total,
            "mode": cfg["mode"],
        },
    )


if __name__ == "__main__":
    main()
