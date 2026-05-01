"""Course 3 / ch4 / class 1 — monitoring: PSI + canary accuracy + latency.

Build an embedding-distribution baseline from AG News, bootstrap a small
classifier for canary scoring, then for each "tick" pull a live batch
that ramps OOD content (Emotion-mixed) and report per-tick:
  - PSI vs baseline (early warning, no labels)
  - canary accuracy on a fixed labeled set (ground truth)
  - mean batch encode latency (operational health)
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np
import torch
from datasets import Dataset, Value, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from shared import drift
from shared.backbones import load_backbone
from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import hf_cache
from shared.repro import set_seed
from shared.training import classification_metrics, make_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _load_ag_news(spec: dict, n: int, split: str, seed: int) -> Dataset:
    ds = load_dataset(spec["hf_id"], split=spec["split"][split])
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    ds = ds.rename_columns({spec["text_field"]: "text", spec["label_field"]: "label"})
    ds = ds.cast_column("label", Value("int64"))
    return ds


def _load_emotion(spec: dict, n: int, seed: int) -> Dataset:
    ds = load_dataset(spec["hf_id"], split=spec["split"])
    remap: dict = {int(k): v for k, v in spec["label_remap"].items()}

    def _row(row):
        new_label = remap.get(int(row[spec["label_field"]]))
        row["_keep"] = new_label is not None
        row["_label"] = new_label if new_label is not None else 0
        return row

    ds = ds.map(_row).filter(lambda r: r["_keep"])
    ds = ds.remove_columns(
        [c for c in ds.column_names if c not in {spec["text_field"], "_label"}]
    )
    ds = ds.rename_columns({spec["text_field"]: "text", "_label": "label"})
    ds = ds.cast_column("label", Value("int64"))
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    return ds


def _tokenize(ds: Dataset, tokenizer, max_len: int) -> Dataset:
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_len, padding=False),
        batched=True,
        remove_columns=["text"],
    )


def _train_canary_classifier(backbone: str, num_labels: int, train_ds: Dataset,
                             tokenizer, cfg: dict, log) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=num_labels)
    output_dir = make_output_dir(
        course=cfg["course"], klass=cfg["class_id"],
        backbone=cfg["backbone"], method=cfg["method"], run_tag="bootstrap_classifier",
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["train"]["epochs_for_classifier"],
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=cfg["train"]["log_steps"],
        seed=cfg["seed"],
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    log.info("[bootstrap_classifier] training on %d labels", len(train_ds))
    trainer.train()
    return model


def _evaluate_canary(model, tokenizer, eval_ds: Dataset, cfg: dict) -> float:
    eval_args = TrainingArguments(
        output_dir=str(make_output_dir(
            course=cfg["course"], klass=cfg["class_id"],
            backbone=cfg["backbone"], method=cfg["method"], run_tag="eval_canary",
        )),
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        report_to=[],
    )

    def _compute(p):
        return classification_metrics(np.argmax(p.predictions, axis=-1), p.label_ids)

    trainer = Trainer(
        model=model, args=eval_args,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute,
    )
    out = trainer.evaluate()
    return float(out["eval_accuracy"])


def _encode_texts(bb, texts: list[str]) -> np.ndarray:
    """Use the SentenceTransformer wrapper to produce a [N, D] numpy array."""
    if bb.kind != "sentence-encoder":
        raise ValueError(f"monitoring requires a sentence-encoder, got {bb.kind}")
    emb = bb.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    return np.asarray(emb, dtype=np.float32)


def main() -> None:
    args = parse_args()
    log = get_logger("course3.ch4.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    n_ticks = len(cfg["live"]["shift_schedule"])
    if cfg.get("method", "").startswith("psi-canary-t"):
        cfg["method"] = f"psi-canary-t{n_ticks}"

    backbone_name = cfg["backbone"]
    log.info("loading encoder: %s", backbone_name)
    bb = load_backbone(backbone_name)

    # ---- Build baseline distribution ---------------------------------------
    log.info("=== building baseline (size=%d, %d PCs, %d bins) ===",
             cfg["baseline"]["size"], cfg["baseline"]["n_components"],
             cfg["baseline"]["n_bins"])
    baseline_ag = _load_ag_news(cfg["dataset"]["ag_news"], n=cfg["baseline"]["size"],
                                split="train", seed=cfg["seed"])
    baseline_emb = _encode_texts(bb, list(baseline_ag["text"]))
    log.info("baseline embeddings: %s", baseline_emb.shape)
    projection = drift.fit_projection(baseline_emb, n_components=cfg["baseline"]["n_components"])
    base_hist, edges = drift.histogram_along_projection(
        baseline_emb, projection, n_bins=cfg["baseline"]["n_bins"]
    )
    psi_baseline = drift.psi(base_hist, base_hist)  # baseline against itself = 0
    log.info("psi_baseline (sanity, baseline-vs-itself) = %.6f", psi_baseline)

    # ---- Bootstrap canary classifier ---------------------------------------
    log.info("=== bootstrapping canary classifier ===")
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ag = _load_ag_news(cfg["dataset"]["ag_news"],
                             n=cfg["bootstrap_classifier"]["train_size"],
                             split="train", seed=cfg["seed"] + 1)
    canary_ag = _load_ag_news(cfg["dataset"]["ag_news"],
                              n=cfg["bootstrap_classifier"]["canary_size"],
                              split="eval", seed=cfg["seed"])
    train_tok = _tokenize(train_ag, tokenizer, cfg["train"]["max_len"])
    canary_tok = _tokenize(canary_ag, tokenizer, cfg["train"]["max_len"])
    classifier = _train_canary_classifier(
        backbone_name, cfg["dataset"]["num_labels"], train_tok, tokenizer, cfg, log
    )
    canary_baseline_acc = _evaluate_canary(classifier, tokenizer, canary_tok, cfg)
    log.info("canary_baseline_acc = %.4f", canary_baseline_acc)

    # ---- Pre-load OOD pool once --------------------------------------------
    ood_pool = _load_emotion(cfg["dataset"]["emotion"],
                             n=cfg["live"]["batch_size"] * n_ticks,
                             seed=cfg["seed"])
    ood_texts = list(ood_pool["text"])

    # ---- Pre-load fresh AG News pool for live batches ----------------------
    live_ag_pool = _load_ag_news(cfg["dataset"]["ag_news"],
                                 n=cfg["live"]["batch_size"] * n_ticks,
                                 split="eval", seed=cfg["seed"] + 2)
    live_ag_texts = list(live_ag_pool["text"])

    # ---- Monitoring ticks --------------------------------------------------
    tick_log: list[dict] = []
    psi_values: list[float] = []
    canary_accs: list[float] = []
    latencies: list[float] = []

    schedule = cfg["live"]["shift_schedule"]
    bs = int(cfg["live"]["batch_size"])

    for tick, ood_frac in enumerate(schedule):
        ood_n = int(round(float(ood_frac) * bs))
        ag_n = bs - ood_n
        offset_ag = tick * bs
        offset_ood = tick * bs
        live_texts = (
            live_ag_texts[offset_ag: offset_ag + ag_n]
            + ood_texts[offset_ood: offset_ood + ood_n]
        )
        if not live_texts:
            log.warning("[tick %d] empty live batch, skipping", tick)
            continue

        # Encode + time
        t0 = time.perf_counter()
        live_emb = _encode_texts(bb, live_texts)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Histogram + PSI
        live_hist, _ = drift.histogram_along_projection(
            live_emb, projection, n_bins=cfg["baseline"]["n_bins"], edges=edges
        )
        score = drift.psi(base_hist, live_hist)

        # Live accuracy: classifier accuracy on the LABELED live batch for
        # this tick. As OOD fraction rises, this naturally falls because
        # the classifier wasn't trained on Emotion text. (A *fixed* canary
        # set wouldn't move tick-to-tick — the lesson needs accuracy that
        # tracks input drift, which is what the live batch provides.)
        live_for_eval_parts = []
        if ag_n > 0:
            live_for_eval_parts.append(
                live_ag_pool.select(range(offset_ag, offset_ag + ag_n))
            )
        if ood_n > 0:
            live_for_eval_parts.append(
                ood_pool.select(range(offset_ood, offset_ood + ood_n))
            )
        live_for_eval = (
            concatenate_datasets(live_for_eval_parts)
            if len(live_for_eval_parts) > 1 else live_for_eval_parts[0]
        )
        live_eval_tok = _tokenize(live_for_eval, tokenizer, cfg["train"]["max_len"])
        live_acc = _evaluate_canary(classifier, tokenizer, live_eval_tok, cfg)

        psi_values.append(float(score))
        canary_accs.append(float(live_acc))
        latencies.append(float(latency_ms))
        tick_log.append({
            "tick": tick, "ood_frac": float(ood_frac),
            "psi": float(score), "live_accuracy": float(live_acc),
            "latency_ms": float(latency_ms),
        })
        log.info("[tick %d] ood=%.2f psi=%.4f live_acc=%.4f latency=%.1fms %s",
                 tick, float(ood_frac), score, live_acc, latency_ms,
                 "(ALARM)" if score >= cfg["psi"]["alarm_threshold"] else "")

    if not tick_log:
        raise RuntimeError("no monitoring ticks ran; check live.shift_schedule + batch_size")

    psi_max = max(psi_values)
    n_above = sum(1 for x in psi_values if x >= cfg["psi"]["alarm_threshold"])
    mean_latency = statistics.fmean(latencies)
    # Negative correlation between PSI and accuracy = PSI is doing its job.
    if len(psi_values) >= 2 and not all(p == psi_values[0] for p in psi_values) \
            and not all(a == canary_accs[0] for a in canary_accs):
        corr_matrix = np.corrcoef(np.asarray(psi_values), np.asarray(canary_accs))
        corr = float(corr_matrix[0, 1])
    else:
        corr = 0.0  # degenerate case — cannot compute correlation
    log.info("FINAL: psi_max=%.4f n_above_alarm=%d corr(psi,live_acc)=%.4f",
             psi_max, n_above, corr)

    metrics = {
        "psi_baseline": float(psi_baseline),
        "psi_max": float(psi_max),
        "live_accuracy_baseline": float(canary_accs[0] if canary_accs else 0.0),
        "live_accuracy_min": float(min(canary_accs) if canary_accs else 0.0),
        "accuracy_psi_correlation": float(corr),
        "n_ticks_above_alarm": int(n_above),
        "mean_latency_ms": float(mean_latency),
    }
    run_eval(
        method=cfg["method"],
        backbone=backbone_name,
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "tick_log": tick_log,
            "shift_schedule": list(schedule),
            "alarm_threshold": cfg["psi"]["alarm_threshold"],
            "n_components": cfg["baseline"]["n_components"],
            "n_bins": cfg["baseline"]["n_bins"],
            "mode": cfg["mode"],
        },
    )


if __name__ == "__main__":
    main()
