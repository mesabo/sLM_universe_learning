"""Course 1 / ch4 / class 1 — DPO on SmolLM2-135M-Instruct via TRL.

Loads the policy + a frozen reference (independent copy of the same SFT
model). Trains on Intel/orca_dpo_pairs with the standard sigmoid DPO loss.
Reports eval loss + reward margins/accuracies.
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import hf_cache
from shared.repro import set_seed
from shared.training import make_output_dir, trainable_param_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _to_dpo_format(ds: Dataset, ds_cfg: dict, tokenizer) -> Dataset:
    """Rename columns to TRL's (prompt, chosen, rejected) format.

    The Intel/orca_dpo_pairs schema is {system, question, chosen, rejected}.
    We render `system` + `question` through `tokenizer.apply_chat_template` to
    build the `prompt` so chat-template-aware models see the full instruction.
    """
    q_field = ds_cfg["question_field"]
    s_field = ds_cfg.get("system_field")
    c_field = ds_cfg["chosen_field"]
    r_field = ds_cfg["rejected_field"]

    def _render(row):
        messages = []
        if s_field and row.get(s_field):
            messages.append({"role": "system", "content": row[s_field]})
        messages.append({"role": "user", "content": row[q_field]})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": row[c_field], "rejected": row[r_field]}

    keep = {"prompt", "chosen", "rejected"}
    ds = ds.map(_render)
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    return ds


def main() -> None:
    args = parse_args()
    log = get_logger("course1.ch4.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    # Auto-derive method tag from beta for grid sweeps.
    beta = float(cfg["dpo"]["beta"])
    if cfg.get("method", "").startswith("dpo-b"):
        cfg["method"] = f"dpo-b{beta}"

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (cfg["train"]["bf16"] and torch.cuda.is_available()) else torch.float32
    log.info("loading policy and reference (both = %s)", cfg["backbone"])
    policy = AutoModelForCausalLM.from_pretrained(cfg["backbone"], torch_dtype=dtype)
    ref = AutoModelForCausalLM.from_pretrained(cfg["backbone"], torch_dtype=dtype)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    trainable, total = trainable_param_count(policy)
    log.info("policy trainable=%d total=%d (ref frozen, same shape)", trainable, total)

    ds_cfg = cfg["dataset"]
    full = load_dataset(ds_cfg["hf_id"], ds_cfg.get("config"), split=ds_cfg["split"])
    full = full.shuffle(seed=cfg["seed"]).select(
        range(min(cfg["limits"][cfg["mode"]]["train"], len(full)))
    )
    holdout = ds_cfg["eval_holdout"]
    eval_split = full.select(range(holdout))
    train_split = full.select(range(holdout, len(full)))
    train_split = _to_dpo_format(train_split, ds_cfg, tokenizer)
    eval_split = _to_dpo_format(eval_split, ds_cfg, tokenizer)
    log.info("split sizes: train=%d eval=%d", len(train_split), len(eval_split))

    output_dir = make_output_dir(
        course=cfg["course"], klass=cfg["class_id"],
        backbone=cfg["backbone"], method=cfg["method"], run_tag=cfg["mode"],
    )

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        max_steps=cfg["limits"][cfg["mode"]]["max_steps"],
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        beta=beta,
        max_length=cfg["dpo"]["max_length"],
        max_prompt_length=cfg["dpo"]["max_prompt_length"],
        loss_type=cfg["dpo"]["loss_type"],
        logging_steps=cfg["train"]["log_steps"],
        eval_strategy="steps",
        eval_steps=cfg["train"]["log_steps"],
        save_strategy="no",
        seed=cfg["seed"],
        report_to=[],
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=dpo_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        tokenizer=tokenizer,
    )

    log.info("training (DPO beta=%.3f)...", beta)
    trainer.train()
    history = trainer.state.log_history
    log.info("history keys (last entry): %s", list(history[-1].keys()) if history else "[]")

    train_losses = [h["loss"] for h in history if "loss" in h]
    eval_losses = [h["eval_loss"] for h in history if "eval_loss" in h]
    eval_margins = [h["eval_rewards/margins"] for h in history if "eval_rewards/margins" in h]
    eval_accs = [h["eval_rewards/accuracies"] for h in history if "eval_rewards/accuracies" in h]
    train_initial = float(train_losses[0]) if train_losses else float("nan")
    train_final = float(train_losses[-1]) if train_losses else float("nan")
    eval_loss = float(eval_losses[-1]) if eval_losses else float("nan")
    margin = float(eval_margins[-1]) if eval_margins else float("nan")
    accuracy = float(eval_accs[-1]) if eval_accs else float("nan")
    log.info("train_initial=%.4f train_final=%.4f eval=%.4f margin=%.4f acc=%.4f",
             train_initial, train_final, eval_loss, margin, accuracy)

    metrics = {
        "train_loss_final": train_final,
        "eval_loss": eval_loss,
        "rewards_margin": margin,
        "rewards_accuracy": accuracy,
        "loss_decreased": int(train_final < train_initial),
    }

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "trainable_params": trainable, "total_params": total,
            "beta": beta, "loss_type": cfg["dpo"]["loss_type"],
            "mode": cfg["mode"],
        },
    )


if __name__ == "__main__":
    main()
