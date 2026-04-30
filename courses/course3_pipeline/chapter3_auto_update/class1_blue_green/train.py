"""Course 3 / ch3 / class 1 — auto-update with drift trigger + blue/green swap.

Bootstraps an AG News classifier (v1, promoted), then runs N cycles where
synthetic drift (Emotion mixed into the train pool) periodically forces a
retrain. Each candidate is promoted only if it clears the acceptance-margin
gate; otherwise it is archived under `checkpoints/.../v{n+1}/` for inspection
and `production.json` stays put.
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

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from shared import registry
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


def _tokenize(ds: Dataset, tokenizer, max_len: int) -> Dataset:
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_len, padding=False),
        batched=True,
        remove_columns=["text"],
    )


def _load_ag_news(spec: dict, n: int, split: str, seed: int) -> Dataset:
    ds = load_dataset(spec["hf_id"], split=spec["split"][split])
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    ds = ds.rename_columns({spec["text_field"]: "text", spec["label_field"]: "label"})
    # Cast label from `ClassLabel` (AG News' default) to plain int64 so it
    # matches the Emotion-loader's schema for concatenate_datasets.
    from datasets import Value
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
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    return ds


def _train_classifier(backbone: str, num_labels: int, train_ds: Dataset,
                      tokenizer, cfg: dict, run_tag: str, log) -> AutoModelForSequenceClassification:
    """Train a fresh classifier from `from_pretrained` on `train_ds`."""
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=num_labels
    )
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
    log.info("[%s] training on %d labels", run_tag, len(train_ds))
    trainer.train()
    return model


def _evaluate(model, tokenizer, eval_ds: Dataset, cfg: dict, log, tag: str) -> float:
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
        model=model, args=eval_args,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute,
    )
    out = trainer.evaluate()
    return float(out["eval_accuracy"])


def _save_model_under_version(model, tokenizer, handle: registry.CheckpointHandle) -> None:
    model.save_pretrained(str(handle.path / "model"))
    tokenizer.save_pretrained(str(handle.path / "model"))


def main() -> None:
    args = parse_args()
    log = get_logger("course3.ch3.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    n_cycles = int(cfg["cycles"]["n"])
    if cfg.get("method", "").startswith("blue-green-c"):
        cfg["method"] = f"blue-green-c{n_cycles}"

    backbone = cfg["backbone"]
    num_labels = cfg["dataset"]["num_labels"]
    course, klass, run_id = cfg["course"], cfg["class_id"], cfg["run_id"]

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Bootstrap v1 ------------------------------------------------------
    log.info("=== bootstrap v1 ===")
    bootstrap_ds = _load_ag_news(cfg["dataset"]["ag_news"],
                                 n=cfg["bootstrap"]["train_size"],
                                 split="train", seed=cfg["seed"])
    canary_ds = _load_ag_news(cfg["dataset"]["ag_news"],
                              n=cfg["bootstrap"]["canary_size"],
                              split="eval", seed=cfg["seed"])
    bootstrap_tok = _tokenize(bootstrap_ds, tokenizer, cfg["train"]["max_len"])
    canary_tok = _tokenize(canary_ds, tokenizer, cfg["train"]["max_len"])

    model = _train_classifier(backbone, num_labels, bootstrap_tok,
                              tokenizer, cfg, run_tag="bootstrap", log=log)
    canary_at_birth = _evaluate(model, tokenizer, canary_tok, cfg, log, tag="bootstrap")
    log.info("[bootstrap] canary accuracy = %.4f", canary_at_birth)

    h1 = registry.register_version(course, klass, run_id, manifest={
        "config_hash": "bootstrap", "canary_accuracy": canary_at_birth,
        "train_size": len(bootstrap_tok),
    })
    _save_model_under_version(model, tokenizer, h1)
    registry.promote(course, klass, run_id, h1.version)
    registry.write_decision_log(course, klass, run_id, {
        "cycle": 0, "decision": "bootstrap", "production_version": h1.version,
        "production_canary_acc": canary_at_birth,
    })

    # ---- Cycles ------------------------------------------------------------
    decisions: list[dict] = []
    n_drift_triggers = 0
    n_promotions = 0
    n_rejections = 0
    cumulative_train = bootstrap_tok
    last_canary_acc = canary_at_birth

    threshold = float(cfg["gate"]["degradation_threshold"])
    margin = float(cfg["gate"]["acceptance_margin"])

    for cycle in range(1, n_cycles + 1):
        log.info("=== cycle %d/%d (production v%d) ===",
                 cycle, n_cycles, registry.get_production(course, klass, run_id).version)

        production = registry.get_production(course, klass, run_id)
        prod_model = AutoModelForSequenceClassification.from_pretrained(
            str(production.path / "model")
        )
        prod_canary_acc = _evaluate(prod_model, tokenizer, canary_tok, cfg, log,
                                    tag=f"prod_c{cycle}")
        log.info("[cycle %d] production v%d canary_acc=%.4f", cycle,
                 production.version, prod_canary_acc)

        decision_row = {
            "cycle": cycle,
            "production_version": production.version,
            "production_canary_acc": prod_canary_acc,
        }

        if prod_canary_acc >= threshold:
            log.info("[cycle %d] no trigger (acc %.4f >= %.4f)", cycle,
                     prod_canary_acc, threshold)
            decision_row["decision"] = "no_trigger"
            decisions.append(decision_row)
            registry.write_decision_log(course, klass, run_id, decision_row)
            last_canary_acc = prod_canary_acc
            continue

        n_drift_triggers += 1
        log.info("[cycle %d] DRIFT TRIGGER (acc %.4f < %.4f) — training candidate",
                 cycle, prod_canary_acc, threshold)
        fresh_ag = _load_ag_news(cfg["dataset"]["ag_news"],
                                 n=cfg["cycles"]["fresh_size"],
                                 split="train", seed=cfg["seed"] + cycle)
        n_drift = int(round(cfg["cycles"]["drift_emotion_ratio"]
                            * cfg["cycles"]["fresh_size"]))
        fresh_em = _load_emotion(cfg["dataset"]["emotion"], n=n_drift,
                                 seed=cfg["seed"] + cycle)
        fresh_combined = concatenate_datasets([fresh_ag, fresh_em]).shuffle(
            seed=cfg["seed"] + cycle
        )
        fresh_tok = _tokenize(fresh_combined, tokenizer, cfg["train"]["max_len"])
        cumulative_train = concatenate_datasets([cumulative_train, fresh_tok])

        candidate = _train_classifier(backbone, num_labels, cumulative_train,
                                      tokenizer, cfg,
                                      run_tag=f"candidate_c{cycle}", log=log)
        candidate_canary_acc = _evaluate(candidate, tokenizer, canary_tok, cfg,
                                         log, tag=f"candidate_c{cycle}")
        log.info("[cycle %d] candidate canary_acc=%.4f", cycle, candidate_canary_acc)

        h_cand = registry.register_version(course, klass, run_id, manifest={
            "config_hash": f"cycle{cycle}",
            "canary_accuracy": candidate_canary_acc,
            "train_size": len(cumulative_train),
        }, parent_version=production.version)
        _save_model_under_version(candidate, tokenizer, h_cand)

        decision_row["candidate_version"] = h_cand.version
        decision_row["candidate_canary_acc"] = candidate_canary_acc

        if candidate_canary_acc >= prod_canary_acc - margin:
            registry.promote(course, klass, run_id, h_cand.version)
            n_promotions += 1
            decision_row["decision"] = "promote"
            decision_row["reason"] = (
                f"candidate {candidate_canary_acc:.4f} >= "
                f"production {prod_canary_acc:.4f} - margin {margin}"
            )
            log.info("[cycle %d] PROMOTE v%d -> production", cycle, h_cand.version)
            last_canary_acc = candidate_canary_acc
        else:
            n_rejections += 1
            decision_row["decision"] = "reject"
            decision_row["reason"] = (
                f"candidate {candidate_canary_acc:.4f} < "
                f"production {prod_canary_acc:.4f} - margin {margin}"
            )
            log.info("[cycle %d] REJECT v%d (kept on disk; production stays v%d)",
                     cycle, h_cand.version, production.version)
            last_canary_acc = prod_canary_acc

        decisions.append(decision_row)
        registry.write_decision_log(course, klass, run_id, decision_row)

    final_production = registry.get_production(course, klass, run_id)
    log.info("FINAL: production=v%d, canary_acc=%.4f, promotions=%d, rejections=%d",
             final_production.version, last_canary_acc, n_promotions, n_rejections)

    metrics = {
        "n_cycles_completed": n_cycles,
        "n_drift_triggers": n_drift_triggers,
        "n_promotions": n_promotions,
        "n_rejections": n_rejections,
        "final_production_version": int(final_production.version),
        "final_canary_accuracy": float(last_canary_acc),
    }
    run_eval(
        method=cfg["method"],
        backbone=backbone,
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "promotion_decisions": decisions,
            "canary_at_birth": canary_at_birth,
            "registry_run_dir": str(registry.run_dir(course, klass, run_id)),
            "mode": cfg["mode"],
        },
    )


if __name__ == "__main__":
    main()
