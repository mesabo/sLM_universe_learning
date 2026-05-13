"""Course 1 / ch5 / class 1 — contrastive fine-tuning of a sentence-encoder.

Uses sentence-transformers' MultipleNegativesRankingLoss + the new
SentenceTransformerTrainer (HF-Trainer-compatible API). After training,
runs a held-out retrieval eval (MRR / recall@k) and persists a result JSON.
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
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
from shared.training import make_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _retrieval_metrics(model: SentenceTransformer, anchors: list[str],
                       positives: list[str], log) -> dict[str, float]:
    """Encode held-out anchors + positives and compute MRR / Recall@1 / Recall@5.

    For each anchor, the candidate set is `positives` and the true positive
    is `positives[i]` (diagonal). Cosine similarity ranks candidates.
    """
    a_emb = model.encode(anchors, convert_to_tensor=True, normalize_embeddings=True,
                         show_progress_bar=False)
    p_emb = model.encode(positives, convert_to_tensor=True, normalize_embeddings=True,
                         show_progress_bar=False)
    sim = a_emb @ p_emb.T  # [N, N]
    n = sim.shape[0]
    diag = sim.diag().unsqueeze(1)
    # rank = number of candidates with sim >= true-positive sim (1 = perfect).
    ranks = (sim >= diag).sum(dim=1).float()
    mrr = (1.0 / ranks).mean().item()
    r1 = (ranks <= 1).float().mean().item()
    r5 = (ranks <= 5).float().mean().item()
    log.info("[eval] N=%d MRR=%.4f R@1=%.4f R@5=%.4f", n, mrr, r1, r5)
    return {"mrr": float(mrr), "recall_at_1": float(r1), "recall_at_5": float(r5)}


def main() -> None:
    args = parse_args()
    log = get_logger("course1.ch5.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    # Auto-derive method tag from batch size.
    batch = int(cfg["train"]["per_device_batch"])
    if cfg.get("method", "").startswith("mnrl-b"):
        cfg["method"] = f"mnrl-b{batch}"

    log.info("loading backbone: %s", cfg["backbone"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(cfg["backbone"], device=device)

    ds_cfg = cfg["dataset"]
    full = load_dataset(ds_cfg["hf_id"], ds_cfg.get("config"), split=ds_cfg["split"])
    full = full.shuffle(seed=cfg["seed"]).select(
        range(min(cfg["limits"][cfg["mode"]]["train"], len(full)))
    )
    holdout = ds_cfg["eval_holdout"]
    eval_split = full.select(range(holdout))
    train_split = full.select(range(holdout, len(full)))

    # SentenceTransformerTrainer expects columns named 'anchor' and 'positive'.
    a_field = ds_cfg["anchor_field"]
    p_field = ds_cfg["positive_field"]
    if a_field != "anchor" or p_field != "positive":
        train_split = train_split.rename_columns({a_field: "anchor", p_field: "positive"})
        eval_split = eval_split.rename_columns({a_field: "anchor", p_field: "positive"})
    log.info("split sizes: train=%d eval=%d", len(train_split), len(eval_split))

    output_dir = make_output_dir(
        course=cfg["course"], klass=cfg["class_id"],
        backbone=cfg["backbone"], method=cfg["method"], run_tag=cfg["mode"],
    )

    loss = losses.MultipleNegativesRankingLoss(model)

    sft_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        max_steps=cfg["limits"][cfg["mode"]]["max_steps"],
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        eval_strategy="no",   # custom retrieval eval below
        save_strategy="no",
        logging_steps=cfg["train"]["log_steps"],
        seed=cfg["seed"],
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_split,
        loss=loss,
    )

    # Pre-FT baseline: encode held-out and measure once.
    log.info("=== pre-FT baseline retrieval eval ===")
    pre_metrics = _retrieval_metrics(
        model, eval_split["anchor"], eval_split["positive"], log
    )

    log.info("training (MNRL, batch=%d)...", batch)
    trainer.train()
    history = trainer.state.log_history
    train_losses = [h["loss"] for h in history if "loss" in h]
    train_initial = float(train_losses[0]) if train_losses else float("nan")
    train_final = float(train_losses[-1]) if train_losses else float("nan")

    log.info("=== post-FT retrieval eval ===")
    post_metrics = _retrieval_metrics(
        model, eval_split["anchor"], eval_split["positive"], log
    )

    metrics = {
        **post_metrics,
        "train_loss_final": train_final,
        "loss_decreased": int(train_final < train_initial),
    }

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "pre_ft": pre_metrics,
            "post_ft": post_metrics,
            "delta_mrr": post_metrics["mrr"] - pre_metrics["mrr"],
            "delta_r1": post_metrics["recall_at_1"] - pre_metrics["recall_at_1"],
            "batch_size": batch,
            "mode": cfg["mode"],
        },
    )


if __name__ == "__main__":
    main()
