"""Course 3 / ch2 / class 1 — active learning: uncertainty sampling vs random.

Two strategies share the same seed labeled set, then add `query_size`
examples per round for `n_rounds` rounds. Compares final accuracy.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

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


def _train_classifier(backbone: str, num_labels: int, train_split: Dataset,
                      tokenizer, cfg: dict, run_tag: str, log) -> AutoModelForSequenceClassification:
    """Train a fresh classifier from `from_pretrained` on the given labeled split."""
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=num_labels
    )
    output_dir = make_output_dir(
        course=cfg["course"], klass=cfg["class_id"],
        backbone=cfg["backbone"], method=cfg["method"], run_tag=run_tag,
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["train"]["epochs_per_round"],
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
        train_dataset=train_split,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    log.info("[%s] training on %d labels", run_tag, len(train_split))
    trainer.train()
    return model


def _evaluate(model, tokenizer, eval_split: Dataset, cfg: dict, log, tag: str) -> float:
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
        eval_dataset=eval_split,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute,
    )
    out = trainer.evaluate()
    acc = float(out["eval_accuracy"])
    log.info("[%s] eval accuracy=%.4f", tag, acc)
    return acc


def _predict_logits(model, tokenizer, ds: Dataset, cfg: dict) -> np.ndarray:
    """Return logits [N, K] for every row in ds."""
    eval_args = TrainingArguments(
        output_dir=str(make_output_dir(
            course=cfg["course"], klass=cfg["class_id"],
            backbone=cfg["backbone"], method=cfg["method"], run_tag="predict_pool",
        )),
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=eval_args,
        eval_dataset=ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    out = trainer.predict(ds)
    return out.predictions


def _entropy(logits: np.ndarray) -> np.ndarray:
    """Per-row entropy of softmax(logits). Higher = more uncertain."""
    t = torch.from_numpy(logits.astype(np.float32))
    p = F.softmax(t, dim=-1)
    return (-(p * (p.clamp_min(1e-12).log())).sum(dim=-1)).numpy()


def _query_random(unlabeled_indices: list[int], k: int, rng: np.random.Generator) -> list[int]:
    if k >= len(unlabeled_indices):
        return list(unlabeled_indices)
    chosen = rng.choice(len(unlabeled_indices), size=k, replace=False)
    return [unlabeled_indices[int(i)] for i in chosen]


def _query_uncertainty(model, tokenizer, pool_tok: Dataset,
                       unlabeled_indices: list[int], k: int, cfg: dict, log) -> list[int]:
    sub = pool_tok.select(unlabeled_indices)
    logits = _predict_logits(model, tokenizer, sub, cfg)
    ents = _entropy(logits)
    top_local = np.argsort(-ents)[:k]
    log.info("[uncertainty] entropies: top=%.4f median=%.4f bottom=%.4f",
             float(ents[top_local[0]]) if len(top_local) else 0.0,
             float(np.median(ents)) if len(ents) else 0.0,
             float(ents.min()) if len(ents) else 0.0)
    return [unlabeled_indices[int(i)] for i in top_local]


def _run_strategy(strategy: str, pool_tok: Dataset, eval_tok: Dataset,
                  cfg: dict, tokenizer, log) -> list[float]:
    """Run a strategy for n_rounds. Returns the accuracy curve (one entry per round)."""
    rng = np.random.default_rng(cfg["seed"])
    seed_size = int(cfg["active"]["seed_size"])
    query_size = int(cfg["active"]["query_size"])
    n_rounds = int(cfg["active"]["n_rounds"])
    n = len(pool_tok)
    indices = rng.permutation(n).tolist()
    labeled = indices[:seed_size]
    unlabeled = indices[seed_size:]

    accs: list[float] = []
    for r in range(n_rounds):
        log.info("=== [%s] round %d/%d, labeled=%d ===", strategy, r + 1, n_rounds, len(labeled))
        train_subset = pool_tok.select(labeled)
        model = _train_classifier(
            cfg["backbone"], cfg["dataset"]["num_labels"], train_subset, tokenizer,
            cfg, run_tag=f"{strategy}_r{r}", log=log,
        )
        accs.append(_evaluate(model, tokenizer, eval_tok, cfg, log, tag=f"{strategy}_r{r}"))

        if r < n_rounds - 1:  # one fewer query than rounds (last round just evaluates)
            if strategy == "random":
                queried = _query_random(unlabeled, query_size, rng)
            elif strategy == "uncertainty":
                queried = _query_uncertainty(model, tokenizer, pool_tok, unlabeled, query_size,
                                             cfg, log)
            else:
                raise ValueError(f"unknown strategy {strategy!r}")
            queried_set = set(queried)
            labeled = labeled + queried
            unlabeled = [i for i in unlabeled if i not in queried_set]

    return accs


def main() -> None:
    args = parse_args()
    log = get_logger("course3.ch2.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    n_rounds = int(cfg["active"]["n_rounds"])
    if cfg.get("method", "").startswith("al-r"):
        cfg["method"] = f"al-r{n_rounds}"

    ds_cfg = cfg["dataset"]
    pool = load_dataset(ds_cfg["hf_id"], split=ds_cfg["split"]["train"])
    pool = pool.shuffle(seed=cfg["seed"])
    cap_pool = cfg["limits"][cfg["mode"]]["pool_size"]
    if cap_pool is not None:
        pool = pool.select(range(min(cap_pool, len(pool))))
    pool = pool.rename_columns({ds_cfg["text_field"]: "text", ds_cfg["label_field"]: "label"})

    eval_split = load_dataset(ds_cfg["hf_id"], split=ds_cfg["split"]["eval"])
    eval_split = eval_split.shuffle(seed=cfg["seed"])
    cap_eval = cfg["limits"][cfg["mode"]]["eval_size"]
    if cap_eval is not None:
        eval_split = eval_split.select(range(min(cap_eval, len(eval_split))))
    eval_split = eval_split.rename_columns({ds_cfg["text_field"]: "text", ds_cfg["label_field"]: "label"})

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pool_tok = _tokenize(pool, tokenizer, cfg["train"]["max_len"])
    eval_tok = _tokenize(eval_split, tokenizer, cfg["train"]["max_len"])
    log.info("pool=%d eval=%d", len(pool_tok), len(eval_tok))

    log.info("##### STRATEGY: random #####")
    curve_random = _run_strategy("random", pool_tok, eval_tok, cfg, tokenizer, log)
    log.info("##### STRATEGY: uncertainty #####")
    curve_uncertainty = _run_strategy("uncertainty", pool_tok, eval_tok, cfg, tokenizer, log)

    final_random = curve_random[-1] if curve_random else 0.0
    final_uncertainty = curve_uncertainty[-1] if curve_uncertainty else 0.0
    delta = final_uncertainty - final_random
    final_budget = int(cfg["active"]["seed_size"] + (n_rounds - 1) * cfg["active"]["query_size"])
    log.info("FINAL: random=%.4f uncertainty=%.4f delta=%+.4f budget=%d",
             final_random, final_uncertainty, delta, final_budget)
    for i, (r, u) in enumerate(zip(curve_random, curve_uncertainty)):
        log.info("  round %d: random=%.4f uncertainty=%.4f delta=%+.4f", i + 1, r, u, u - r)

    metrics = {
        "final_accuracy_random": float(final_random),
        "final_accuracy_uncertainty": float(final_uncertainty),
        "delta_accuracy": float(delta),
        "n_rounds_completed": int(n_rounds),
        "final_label_budget": final_budget,
    }
    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "accuracy_curve_random": [round(x, 4) for x in curve_random],
            "accuracy_curve_uncertainty": [round(x, 4) for x in curve_uncertainty],
            "seed_size": cfg["active"]["seed_size"],
            "query_size": cfg["active"]["query_size"],
            "mode": cfg["mode"],
        },
    )


if __name__ == "__main__":
    main()
