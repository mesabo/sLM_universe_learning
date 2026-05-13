"""Course 3 / ch5 / class 1 — end-to-end mini-deployment (capstone).

Combines ch1 (serving) + ch2 (active learning) + ch3 (auto-update) + ch4
(monitoring) into one tick-based simulation. Each tick: serve a batch
(latency + entropy), update the AL queue, periodically compute PSI +
live accuracy (monitoring), and trigger a candidate retrain when any of
{PSI alarm, live_acc < threshold, AL queue commit} fires.
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
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, Value, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from shared import drift, registry
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
    ds = ds.remove_columns([c for c in ds.column_names if c not in {spec["text_field"], "_label"}])
    ds = ds.rename_columns({spec["text_field"]: "text", "_label": "label"})
    ds = ds.cast_column("label", Value("int64"))
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    return ds


def _tokenize(ds: Dataset, tokenizer, max_len: int) -> Dataset:
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_len, padding=False),
        batched=True, remove_columns=["text"],
    )


def _train_classifier(backbone: str, num_labels: int, train_ds: Dataset,
                      tokenizer, cfg: dict, run_tag: str, log) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=num_labels)
    output_dir = make_output_dir(
        course=cfg["course"], klass=cfg["class_id"],
        backbone=cfg["backbone"], method=cfg["method"], run_tag=run_tag,
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["train"]["epochs_per_train"],
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        fp16=cfg["train"]["fp16"] and torch.cuda.is_available(),
        eval_strategy="no", save_strategy="no",
        logging_steps=cfg["train"]["log_steps"],
        seed=cfg["seed"], report_to=[],
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    log.info("[%s] training on %d labels", run_tag, len(train_ds))
    trainer.train()
    return model


def _evaluate(model, tokenizer, eval_ds: Dataset, cfg: dict, tag: str = "eval") -> float:
    eval_args = TrainingArguments(
        output_dir=str(make_output_dir(
            course=cfg["course"], klass=cfg["class_id"],
            backbone=cfg["backbone"], method=cfg["method"], run_tag=f"eval_{tag}",
        )),
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        report_to=[],
    )

    def _compute(p):
        return classification_metrics(np.argmax(p.predictions, axis=-1), p.label_ids)

    trainer = Trainer(
        model=model, args=eval_args, eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute,
    )
    return float(trainer.evaluate()["eval_accuracy"])


def _predict_with_entropy(model, tokenizer, ds: Dataset, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Predict; return (predictions, per-row softmax entropy)."""
    eval_args = TrainingArguments(
        output_dir=str(make_output_dir(
            course=cfg["course"], klass=cfg["class_id"],
            backbone=cfg["backbone"], method=cfg["method"], run_tag="predict",
        )),
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=eval_args, eval_dataset=ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    out = trainer.predict(ds)
    logits = torch.from_numpy(out.predictions.astype(np.float32))
    probs = F.softmax(logits, dim=-1)
    entropy = (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).numpy()
    preds = np.argmax(out.predictions, axis=-1)
    return preds, entropy


def _save_model_under_version(model, tokenizer, handle: registry.CheckpointHandle) -> None:
    model.save_pretrained(str(handle.path / "model"))
    tokenizer.save_pretrained(str(handle.path / "model"))


def main() -> None:
    args = parse_args()
    log = get_logger("course3.ch5.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    n_ticks = int(cfg["pipeline"]["n_ticks"])
    if cfg.get("method", "").startswith("pipeline-t"):
        cfg["method"] = f"pipeline-t{n_ticks}"
    schedule = cfg["pipeline"]["shift_schedule"]
    if len(schedule) != n_ticks:
        raise ValueError(
            f"shift_schedule length ({len(schedule)}) must equal n_ticks ({n_ticks})"
        )

    backbone = cfg["backbone"]
    num_labels = cfg["dataset"]["num_labels"]
    course, klass, run_id = cfg["course"], cfg["class_id"], cfg["run_id"]

    # ---- Load encoder for monitoring ---------------------------------------
    log.info("=== loading encoder for monitoring ===")
    enc = load_backbone(backbone)
    if enc.kind != "sentence-encoder":
        raise ValueError(f"need a sentence-encoder, got {enc.kind}")

    log.info("=== building monitoring baseline ===")
    baseline_ag = _load_ag_news(cfg["dataset"]["ag_news"], n=cfg["baseline"]["size"],
                                split="train", seed=cfg["seed"])
    baseline_emb = enc.model.encode(list(baseline_ag["text"]), convert_to_numpy=True,
                                    show_progress_bar=False, batch_size=64)
    projection = drift.fit_projection(baseline_emb,
                                      n_components=cfg["baseline"]["n_components"])
    base_hist, edges = drift.histogram_along_projection(
        baseline_emb, projection, n_bins=cfg["baseline"]["n_bins"]
    )

    # ---- Bootstrap classifier + register v1 -------------------------------
    log.info("=== bootstrapping production classifier ===")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bootstrap_ds = _load_ag_news(cfg["dataset"]["ag_news"],
                                 n=cfg["bootstrap"]["train_size"],
                                 split="train", seed=cfg["seed"] + 1)
    canary_ds = _load_ag_news(cfg["dataset"]["ag_news"],
                              n=cfg["bootstrap"]["canary_size"],
                              split="eval", seed=cfg["seed"])
    bootstrap_tok = _tokenize(bootstrap_ds, tokenizer, cfg["train"]["max_len"])
    canary_tok = _tokenize(canary_ds, tokenizer, cfg["train"]["max_len"])
    classifier = _train_classifier(backbone, num_labels, bootstrap_tok,
                                   tokenizer, cfg, run_tag="bootstrap", log=log)
    canary_v1 = _evaluate(classifier, tokenizer, canary_tok, cfg, tag="bootstrap")
    log.info("canary v1 = %.4f", canary_v1)
    h1 = registry.register_version(course, klass, run_id, manifest={
        "canary_accuracy": canary_v1, "train_size": len(bootstrap_tok),
    })
    _save_model_under_version(classifier, tokenizer, h1)
    registry.promote(course, klass, run_id, h1.version)

    cumulative_train = bootstrap_tok

    # ---- Pre-load live + OOD pools ----------------------------------------
    bs = int(cfg["pipeline"]["live_batch_size"])
    total_n = bs * n_ticks
    live_ag_pool = _load_ag_news(cfg["dataset"]["ag_news"], n=total_n,
                                 split="eval", seed=cfg["seed"] + 2)
    ood_pool = _load_emotion(cfg["dataset"]["emotion"], n=total_n, seed=cfg["seed"])

    # ---- Ticks -------------------------------------------------------------
    n_predictions_served = 0
    n_drift_alarms = 0
    n_active_learning_commits = 0
    n_promotions = 0
    al_queue: list[int] = []   # indices into the cumulative live row pool
    al_buffer: list[Dataset] = []  # accumulating committed labeled batches
    last_psi = 0.0
    last_live_acc = canary_v1
    tick_log: list[dict] = []
    latencies: list[float] = []

    for tick in range(n_ticks):
        ood_frac = float(schedule[tick])
        ood_n = int(round(ood_frac * bs))
        ag_n = bs - ood_n
        offset = tick * bs

        # Build live batch (with labels for AL/monitoring).
        parts = []
        if ag_n > 0:
            parts.append(live_ag_pool.select(range(offset, offset + ag_n)))
        if ood_n > 0:
            parts.append(ood_pool.select(range(offset, offset + ood_n)))
        live_batch = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        live_tok = _tokenize(live_batch, tokenizer, cfg["train"]["max_len"])

        # SERVING: predict + entropy + latency
        t0 = time.perf_counter()
        preds, entropy = _predict_with_entropy(classifier, tokenizer, live_tok, cfg)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)
        n_predictions_served += int(len(preds))

        # ACTIVE LEARNING: queue high-entropy rows
        thr = float(cfg["active"]["uncertainty_threshold"])
        high_ent = [i for i, e in enumerate(entropy) if e >= thr]
        al_queue.extend([(tick, i) for i in high_ent])
        al_committed = False
        if len(al_queue) >= int(cfg["active"]["label_budget"]):
            committed_rows = []
            for (t_idx, row_idx) in al_queue:
                # Reconstruct the labeled row from the live pool
                src_offset = t_idx * bs
                if row_idx < (bs - int(round(float(schedule[t_idx]) * bs))):
                    committed_rows.append(
                        live_ag_pool.select([src_offset + row_idx])
                    )
                else:
                    ood_row_idx = row_idx - (bs - int(round(float(schedule[t_idx]) * bs)))
                    committed_rows.append(
                        ood_pool.select([src_offset + ood_row_idx])
                    )
            committed_ds = concatenate_datasets(committed_rows)
            committed_tok = _tokenize(committed_ds, tokenizer, cfg["train"]["max_len"])
            cumulative_train = concatenate_datasets([cumulative_train, committed_tok])
            n_active_learning_commits += 1
            al_committed = True
            log.info("[tick %d] AL COMMIT: %d new labels (cumulative=%d)",
                     tick, len(committed_tok), len(cumulative_train))
            al_queue = []

        # MONITORING (every tick_period ticks)
        psi_score = None
        live_acc = None
        if tick % int(cfg["monitoring"]["tick_period"]) == 0:
            live_emb = enc.model.encode(list(live_batch["text"]), convert_to_numpy=True,
                                        show_progress_bar=False, batch_size=64)
            live_hist, _ = drift.histogram_along_projection(
                live_emb, projection, n_bins=cfg["baseline"]["n_bins"], edges=edges
            )
            psi_score = float(drift.psi(base_hist, live_hist))
            live_acc = float((preds == live_batch["label"]).mean()) if "label" in live_batch.column_names else None
            last_psi = psi_score
            last_live_acc = live_acc if live_acc is not None else last_live_acc
            if psi_score >= float(cfg["monitoring"]["alarm_threshold"]):
                n_drift_alarms += 1

        # AUTO-UPDATE TRIGGER
        decision = "no_action"
        candidate_canary = None
        psi_alarm = (psi_score is not None and
                     psi_score >= float(cfg["monitoring"]["alarm_threshold"]))
        acc_alarm = (live_acc is not None and
                     live_acc < float(cfg["gate"]["degradation_threshold"]))
        if psi_alarm or acc_alarm or al_committed:
            log.info("[tick %d] auto-update trigger (psi_alarm=%s acc_alarm=%s al=%s)",
                     tick, psi_alarm, acc_alarm, al_committed)
            production = registry.get_production(course, klass, run_id)
            prod_canary = _evaluate(classifier, tokenizer, canary_tok, cfg,
                                    tag=f"prod_t{tick}")
            candidate = _train_classifier(
                backbone, num_labels, cumulative_train, tokenizer, cfg,
                run_tag=f"candidate_t{tick}", log=log,
            )
            candidate_canary = _evaluate(candidate, tokenizer, canary_tok, cfg,
                                         tag=f"candidate_t{tick}")
            h_cand = registry.register_version(course, klass, run_id, manifest={
                "canary_accuracy": candidate_canary,
                "train_size": len(cumulative_train),
            }, parent_version=production.version)
            _save_model_under_version(candidate, tokenizer, h_cand)
            margin = float(cfg["gate"]["acceptance_margin"])
            if candidate_canary >= prod_canary - margin:
                registry.promote(course, klass, run_id, h_cand.version)
                classifier = candidate
                n_promotions += 1
                decision = "promote"
                log.info("[tick %d] PROMOTE v%d (cand=%.4f >= prod=%.4f - margin)",
                         tick, h_cand.version, candidate_canary, prod_canary)
            else:
                decision = "reject"
                log.info("[tick %d] REJECT v%d (cand=%.4f < prod=%.4f - margin)",
                         tick, h_cand.version, candidate_canary, prod_canary)

        tick_log.append({
            "tick": tick, "ood_frac": ood_frac,
            "n_served": int(len(preds)), "latency_ms": float(latency_ms),
            "queue_size": len(al_queue), "al_committed": al_committed,
            "psi": psi_score, "live_acc": live_acc, "decision": decision,
            "candidate_canary": candidate_canary,
        })

    final_production = registry.get_production(course, klass, run_id)
    at_least_one = int((n_drift_alarms + n_active_learning_commits + n_promotions) > 0)
    log.info("FINAL: served=%d alarms=%d AL_commits=%d promotions=%d prod=v%d",
             n_predictions_served, n_drift_alarms, n_active_learning_commits,
             n_promotions, final_production.version)

    metrics = {
        "n_predictions_served": int(n_predictions_served),
        "n_drift_alarms": int(n_drift_alarms),
        "n_active_learning_commits": int(n_active_learning_commits),
        "n_promotions": int(n_promotions),
        "final_production_version": int(final_production.version),
        "final_live_accuracy": float(last_live_acc),
        "mean_latency_ms": float(statistics.fmean(latencies)) if latencies else 0.0,
        "at_least_one_loop_fired": int(at_least_one),
    }
    run_eval(
        method=cfg["method"], backbone=backbone,
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "tick_log": tick_log,
            "shift_schedule": list(schedule),
            "registry_run_dir": str(registry.run_dir(course, klass, run_id)),
            "mode": cfg["mode"],
        },
    )


if __name__ == "__main__":
    main()
