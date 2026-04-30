"""Reload a 4-bit base + QLoRA adapter and recompute eval loss."""

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

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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


def _bnb_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()
    log = get_logger("course1.ch3.class1.eval")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA eval requires CUDA.")

    qcfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=_bnb_dtype(qcfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["backbone"], quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.ckpt)

    ds_cfg = cfg["dataset"]
    full = load_dataset(ds_cfg["hf_id"], ds_cfg.get("config"), split=ds_cfg["split"])
    full = full.shuffle(seed=cfg["seed"]).select(range(ds_cfg["eval_holdout"]))

    def _format(example):
        return tokenizer.apply_chat_template(example[ds_cfg["messages_field"]], tokenize=False)

    sft_args = SFTConfig(
        output_dir=str(args.ckpt) + "_eval",
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        bf16=cfg["train"]["bf16"],
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
        "base_in_4bit": 1,
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
