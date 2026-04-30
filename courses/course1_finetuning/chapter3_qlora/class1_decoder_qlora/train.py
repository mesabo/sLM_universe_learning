"""Course 1 / ch3 / class 1 — QLoRA: 4-bit base + LoRA adapters via TRL.

Requires the GPU env (bitsandbytes, NVIDIA GPU). Will fail fast on CPU
with a clear error from bitsandbytes.
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

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import hf_cache
from shared.repro import set_seed
from shared.training import lora_target_modules, make_output_dir, trainable_param_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _bnb_dtype(name: str) -> torch.dtype:
    table = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in table:
        raise ValueError(f"bnb_4bit_compute_dtype must be one of {list(table)}, got {name!r}")
    return table[name]


def _is_4bit_loaded(model) -> bool:
    """True if any base parameter ended up as a `Linear4bit` layer."""
    try:
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        return False
    return any(isinstance(m, Linear4bit) for m in model.modules())


def main() -> None:
    args = parse_args()
    log = get_logger("course1.ch3.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    if not torch.cuda.is_available():
        raise SystemExit(
            "QLoRA requires a CUDA GPU; bitsandbytes has no CPU 4-bit path. "
            "Use the slm-gpu conda env."
        )

    qcfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=_bnb_dtype(qcfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg["backbone"], quantization_config=bnb_config, device_map="auto"
    )
    base_in_4bit = _is_4bit_loaded(base)
    log.info("base_in_4bit=%s", base_in_4bit)

    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"]
    )

    target = cfg["lora"]["target_modules"] or lora_target_modules(base)
    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=target,
        bias=cfg["lora"]["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)
    trainable, total = trainable_param_count(model)
    ratio_pct = 100.0 * trainable / total
    log.info("QLoRA trainable=%d total=%d ratio=%.4f%% targets=%s",
             trainable, total, ratio_pct, target)

    ds_cfg = cfg["dataset"]
    full = load_dataset(ds_cfg["hf_id"], ds_cfg.get("config"), split=ds_cfg["split"])
    full = full.shuffle(seed=cfg["seed"]).select(
        range(min(cfg["limits"][cfg["mode"]]["train"], len(full)))
    )
    holdout = ds_cfg["eval_holdout"]
    eval_split = full.select(range(holdout))
    train_split = full.select(range(holdout, len(full)))

    def _format(example):
        return tokenizer.apply_chat_template(example[ds_cfg["messages_field"]], tokenize=False)

    output_dir = make_output_dir(
        course=cfg["course"], klass=cfg["class_id"],
        backbone=cfg["backbone"], method=cfg["method"], run_tag=cfg["mode"],
    )

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=cfg["limits"][cfg["mode"]]["max_steps"],
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        max_seq_length=cfg["train"]["max_seq_length"],
        bf16=cfg["train"]["bf16"],
        packing=cfg["train"]["packing"],
        optim=cfg["train"]["optim"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        logging_steps=cfg["train"]["log_steps"],
        eval_strategy="steps",
        eval_steps=cfg["train"]["log_steps"],
        save_strategy="no",
        seed=cfg["seed"],
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model, args=sft_args,
        train_dataset=train_split, eval_dataset=eval_split,
        tokenizer=tokenizer, formatting_func=_format,
    )

    log.info("training (QLoRA, paged optimizer)...")
    trainer.train()
    history = trainer.state.log_history
    train_losses = [h["loss"] for h in history if "loss" in h]
    eval_losses = [h["eval_loss"] for h in history if "eval_loss" in h]
    train_initial = float(train_losses[0]) if train_losses else float("nan")
    train_final = float(train_losses[-1]) if train_losses else float("nan")
    eval_loss = float(eval_losses[-1]) if eval_losses else float("nan")
    log.info("train_initial=%.4f train_final=%.4f eval=%.4f",
             train_initial, train_final, eval_loss)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {
        "train_loss_final": train_final,
        "eval_loss": eval_loss,
        "loss_decreased": int(train_final < train_initial),
        "trainable_ratio_pct": ratio_pct,
        "base_in_4bit": int(base_in_4bit),
    }

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "trainable_params": trainable, "total_params": total,
            "lora_target_modules": target, "mode": cfg["mode"],
            "adapter_dir": str(output_dir),
            "quantization": qcfg,
        },
    )


if __name__ == "__main__":
    main()
