"""Course 2 / ch3 / class 1 — Elastic Weight Consolidation (EWC) on AG News -> Emotion.

Pipeline mirrors ch1's measurement, with two new steps between training Task A
and training Task B:

  - Compute the diagonal Fisher information matrix on Task A's eval split.
  - Snapshot theta_A (the parameter values at the end of Task A).

Then training on Task B uses an EWCTrainer whose compute_loss adds
  lambda/2 * sum_i F_i * (theta_i - theta_A_i)^2
to the standard cross-entropy. The metric band asserts BWT improves over
ch1's catastrophic baseline.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from shared.config import apply_overrides, load_yaml
from shared.continual import History, summarize
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import hf_cache
from shared.repro import set_seed
from shared.training import (
    classification_metrics,
    freeze_base,
    make_output_dir,
    trainable_param_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


@dataclass
class TaskSplits:
    name: str
    train: Dataset
    eval: Dataset


def _load_one_task(spec: dict, cap_train: int | None, cap_eval: int | None,
                   seed: int, log) -> TaskSplits:
    hf_id = spec["hf_id"]
    cfg_name = spec.get("config")
    split = spec["split"]
    text_field = spec["text_field"]
    label_field = spec["label_field"]
    remap: dict[int, int | None] = {int(k): v for k, v in spec["label_remap"].items()}

    def _load_split(split_name: str, cap: int | None) -> Dataset:
        ds = load_dataset(hf_id, cfg_name, split=split_name) if cfg_name \
            else load_dataset(hf_id, split=split_name)

        def _remap_row(row):
            new_label = remap.get(int(row[label_field]))
            row["_keep"] = new_label is not None
            row["_label"] = new_label if new_label is not None else 0
            return row

        ds = ds.map(_remap_row)
        ds = ds.filter(lambda r: r["_keep"])
        ds = ds.remove_columns([c for c in ds.column_names
                                if c not in {text_field, "_label"}])
        ds = ds.rename_columns({text_field: "text", "_label": "label"})
        ds = ds.shuffle(seed=seed)
        if cap is not None:
            ds = ds.select(range(min(cap, len(ds))))
        return ds

    train = _load_split(split["train"], cap_train)
    ev = _load_split(split["eval"], cap_eval)
    log.info("[task %s] train=%d eval=%d (after remap)", spec["name"], len(train), len(ev))
    return TaskSplits(name=spec["name"], train=train, eval=ev)


def _tokenize(ds: Dataset, tokenizer, max_len: int) -> Dataset:
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_len, padding=False),
        batched=True,
        remove_columns=["text"],
    )


def _evaluate_task(model, tokenizer, task: TaskSplits, cfg: dict, log) -> float:
    eval_args = TrainingArguments(
        output_dir=str(make_output_dir(
            course=cfg["course"], klass=cfg["class_id"],
            backbone=cfg["backbone"], method=cfg["method"], run_tag=f"eval_{task.name}",
        )),
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        bf16=cfg["train"]["bf16"] and torch.cuda.is_available(),
        fp16=cfg["train"]["fp16"] and torch.cuda.is_available(),
        report_to=[],
    )
    tokenized = _tokenize(task.eval, tokenizer, cfg["train"]["max_len"])

    def _compute(p):
        return classification_metrics(np.argmax(p.predictions, axis=-1), p.label_ids)

    trainer = Trainer(
        model=model, args=eval_args,
        eval_dataset=tokenized,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute,
    )
    out = trainer.evaluate()
    acc = float(out["eval_accuracy"])
    log.info("[eval %s] accuracy=%.4f", task.name, acc)
    return acc


# ----------------------------------------------------------------------------
# EWC core
# ----------------------------------------------------------------------------


def _compute_fisher_diagonal(model, tokenizer, task: TaskSplits, cfg: dict,
                             log) -> dict[str, torch.Tensor]:
    """Compute the diagonal Fisher information matrix on Task A.

    For each parameter θ_i, F_i = E_{(x,y) ~ D_A}[(d log p(y|x) / dθ_i)^2].
    We use true labels (Empirical Fisher); cap rows at `ewc.fisher_n_samples`.
    Returns a dict: param_name -> Fisher tensor (same shape as the parameter,
    fp32, on the same device as the model).
    """
    n_samples = int(cfg["ewc"]["fisher_n_samples"])
    batch_size = int(cfg["ewc"]["fisher_batch"])

    subset = task.eval.select(range(min(n_samples, len(task.eval))))
    tokenized = _tokenize(subset, tokenizer, cfg["train"]["max_len"])
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model.eval()
    device = next(model.parameters()).device

    fisher: dict[str, torch.Tensor] = {
        n: torch.zeros_like(p, dtype=torch.float32, device=p.device)
        for n, p in model.named_parameters() if p.requires_grad
    }
    n_seen = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
        # Negative log-prob of the true label, summed over the batch.
        nll = -log_probs.gather(1, labels.view(-1, 1)).sum()
        nll.backward()
        for n, p in model.named_parameters():
            if p.grad is None or n not in fisher:
                continue
            fisher[n] += p.grad.detach().float().pow(2)
        n_seen += labels.numel()

    if n_seen > 0:
        for n in fisher:
            fisher[n] /= float(n_seen)

    model.zero_grad(set_to_none=True)
    fisher_means = [t.mean().item() for t in fisher.values()]
    log.info("[fisher] computed over %d samples, mean=%.6e min=%.6e max=%.6e",
             n_seen, float(np.mean(fisher_means)) if fisher_means else 0.0,
             float(np.min(fisher_means)) if fisher_means else 0.0,
             float(np.max(fisher_means)) if fisher_means else 0.0)
    return fisher


def _snapshot_theta(model) -> dict[str, torch.Tensor]:
    """Clone every trainable parameter into a fp32 reference snapshot."""
    return {
        n: p.detach().clone().float()
        for n, p in model.named_parameters() if p.requires_grad
    }


class EWCTrainer(Trainer):
    """HF Trainer that adds the EWC penalty to compute_loss."""

    def __init__(self, *args, fisher: dict[str, torch.Tensor],
                 theta_star: dict[str, torch.Tensor], ewc_lambda: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._fisher = fisher
        self._theta_star = theta_star
        self._lambda = float(ewc_lambda)

    def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
        outputs = model(**inputs)
        ce_loss = outputs.loss
        if self._lambda <= 0.0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        ewc_loss = torch.zeros((), device=ce_loss.device, dtype=torch.float32)
        for n, p in model.named_parameters():
            # DataParallel / DDP / accelerate wrap the model and prefix names
            # with "module." (DP/DDP) or sometimes other prefixes. Strip a
            # leading "module." so Fisher keys (gathered pre-wrap) match.
            key = n[len("module."):] if n.startswith("module.") else n
            if key not in self._fisher or key not in self._theta_star:
                continue
            f = self._fisher[key].to(p.device)
            star = self._theta_star[key].to(p.device)
            ewc_loss = ewc_loss + (f * (p.float() - star).pow(2)).sum()
        penalty = 0.5 * self._lambda * ewc_loss
        total = ce_loss + penalty
        # Periodically log the penalty so the student sees EWC actually biting.
        step = int(getattr(self.state, "global_step", 0))
        if step % max(1, int(self.args.logging_steps)) == 0:
            ce_val = float(ce_loss.mean().item()) if ce_loss.numel() > 1 else float(ce_loss.item())
            print(f"[EWC] step={step} ce={ce_val:.4f} penalty={float(penalty.item()):.4f} "
                  f"lambda={self._lambda}")
        return (total, outputs) if return_outputs else total


# ----------------------------------------------------------------------------
# Stage drivers
# ----------------------------------------------------------------------------


def _train_on_task(model, tokenizer, task: TaskSplits, cfg: dict, log,
                   trainer_cls=Trainer, trainer_kwargs: dict | None = None) -> None:
    """Train (in place) on one task for `epochs_per_task` epochs."""
    output_dir = make_output_dir(
        course=cfg["course"], klass=cfg["class_id"],
        backbone=cfg["backbone"], method=cfg["method"], run_tag=f"train_{task.name}",
    )
    tokenized = _tokenize(task.train, tokenizer, cfg["train"]["max_len"])
    epochs = cfg["limits"][cfg["mode"]]["epochs_per_task"]

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
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
    extra = trainer_kwargs or {}
    trainer = trainer_cls(
        model=model, args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorWithPadding(tokenizer),
        **extra,
    )
    log.info("[train %s] starting (%d rows, %d epochs)%s", task.name, len(tokenized), epochs,
             f" ({trainer_cls.__name__})" if trainer_cls is not Trainer else "")
    trainer.train()


def main() -> None:
    args = parse_args()
    log = get_logger("course2.ch3.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    # Auto-derive method tag from lambda for grid sweeps.
    ewc_lambda = float(cfg["ewc"]["lambda"])
    if cfg.get("method", "").startswith("ewc-l"):
        cfg["method"] = f"ewc-l{int(ewc_lambda)}"

    limits = cfg["limits"][cfg["mode"]]
    tasks = [
        _load_one_task(t, limits["train_per_task"], limits["eval_per_task"], cfg["seed"], log)
        for t in cfg["tasks"]
    ]
    if len(tasks) != 2:
        raise ValueError("This class is hard-wired for exactly 2 tasks.")
    task_a, task_b = tasks

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["backbone"], num_labels=cfg["num_labels"]
    )
    if cfg["freeze_backbone"]:
        n_frozen = freeze_base(model.base_model)
        log.info("froze %d backbone params (head still trainable)", n_frozen)
    trainable, total = trainable_param_count(model)
    log.info("trainable=%d total=%d ratio=%.4f ewc_lambda=%.2f",
             trainable, total, trainable / total, ewc_lambda)

    history = History()

    log.info("=== STAGE 0 (before any training) ===")
    history.add_stage(0, {i: _evaluate_task(model, tokenizer, t, cfg, log)
                          for i, t in enumerate(tasks)})

    log.info("=== STAGE 1 (after training task 0 = %s) ===", task_a.name)
    _train_on_task(model, tokenizer, task_a, cfg, log)
    history.add_stage(1, {i: _evaluate_task(model, tokenizer, t, cfg, log)
                          for i, t in enumerate(tasks)})

    log.info("=== EWC: snapshot theta_A and compute Fisher on %s ===", task_a.name)
    theta_star = _snapshot_theta(model)
    fisher = _compute_fisher_diagonal(model, tokenizer, task_a, cfg, log)
    fisher_mean = float(np.mean([t.mean().item() for t in fisher.values()])) if fisher else 0.0

    log.info("=== STAGE 2 (after training task 1 = %s with EWC l=%.2f) ===",
             task_b.name, ewc_lambda)
    _train_on_task(
        model, tokenizer, task_b, cfg, log,
        trainer_cls=EWCTrainer,
        trainer_kwargs={"fisher": fisher, "theta_star": theta_star, "ewc_lambda": ewc_lambda},
    )
    history.add_stage(2, {i: _evaluate_task(model, tokenizer, t, cfg, log)
                          for i, t in enumerate(tasks)})

    summary = summarize(history)
    log.info("history matrix:\n%s", _format_matrix(history))
    log.info("summary: %s", summary)

    metrics = {
        "avg_accuracy": float(summary["avg_accuracy"]),
        "acc_A_after_A": float(history.acc(1, 0)),
        "acc_A_after_B": float(history.acc(2, 0)),
        "acc_B_after_A": float(history.acc(1, 1)),
        "acc_B_after_B": float(history.acc(2, 1)),
        "BWT": float(summary["BWT"]),
        "ewc_lambda": ewc_lambda,
        "fisher_mean": fisher_mean,
    }

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "history_matrix": history.to_matrix(),
            "FWT": float(summary["FWT"]),
            "task_names": [t.name for t in tasks],
            "trainable_params": trainable,
            "total_params": total,
            "fisher_n_samples_used": int(cfg["ewc"]["fisher_n_samples"]),
            "mode": cfg["mode"],
        },
    )


def _format_matrix(history: History) -> str:
    rows = history.to_matrix()
    header = " " * 8 + "".join(f"  task{i:>2}" for i in range(history.n_tasks()))
    lines = [header]
    for k, row in enumerate(rows):
        cells = "  ".join(f"{(v if v is not None else float('nan')):.4f}" for v in row)
        lines.append(f"stage{k:>2}  {cells}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
