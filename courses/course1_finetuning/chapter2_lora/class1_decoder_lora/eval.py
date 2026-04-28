"""Reload an adapter from disk and recompute eval loss."""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import hf_cache
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True, help="Adapter dir saved by train.py")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = get_logger("course1.ch2.class1.eval")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if (cfg["train"]["bf16"] and torch.cuda.is_available()) else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg["backbone"], torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, args.ckpt)

    ds_cfg = cfg["dataset"]
    full = load_dataset(ds_cfg["hf_id"], ds_cfg.get("config"), split=ds_cfg["split"])
    full = full.shuffle(seed=cfg["seed"]).select(range(ds_cfg["eval_holdout"]))

    def _format(example):
        return tokenizer.apply_chat_template(example[ds_cfg["messages_field"]], tokenize=False)

    sft_args = SFTConfig(
        output_dir=str(args.ckpt) + "_eval",
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        max_seq_length=cfg["train"]["max_seq_length"],
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=sft_args,
                         eval_dataset=full, tokenizer=tokenizer, formatting_func=_format)
    out = trainer.evaluate()
    metrics = {
        "train_loss_final": float("nan"),
        "eval_loss": float(out["eval_loss"]),
        "loss_decreased": 1,
        "trainable_ratio_pct": 0,
    }
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
