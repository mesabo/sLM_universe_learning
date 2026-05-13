"""Course 1 / ch2 / class 1 — LoRA fine-tuning of a decoder sLM with TRL + PEFT."""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
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


def main() -> None:
    args = parse_args()
    log = get_logger("course1.ch2.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (cfg["train"]["bf16"] and torch.cuda.is_available()) else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg["backbone"], torch_dtype=dtype)

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
    log.info("LoRA trainable=%d total=%d ratio=%.4f%% targets=%s",
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
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        packing=cfg["train"]["packing"],
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

    log.info("training...")
    trainer.train()
    history = trainer.state.log_history
    train_losses = [h["loss"] for h in history if "loss" in h]
    eval_losses = [h["eval_loss"] for h in history if "eval_loss" in h]
    train_initial = float(train_losses[0]) if train_losses else float("nan")
    train_final = float(train_losses[-1]) if train_losses else float("nan")
    eval_loss = float(eval_losses[-1]) if eval_losses else float("nan")
    log.info("train_initial=%.4f train_final=%.4f eval=%.4f",
             train_initial, train_final, eval_loss)

    # Save adapter only.
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {
        "train_loss_final": train_final,
        "eval_loss": eval_loss,
        "loss_decreased": int(train_final < train_initial),
        "trainable_ratio_pct": ratio_pct,
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
        },
    )


if __name__ == "__main__":
    main()
