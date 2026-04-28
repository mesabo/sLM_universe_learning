"""Tiny helpers used across Course 1+ training classes.

Keep this module thin. If something is class-specific (e.g. a custom LoRA
target-module list for a niche backbone) it lives in that class, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import checkpoints_root


@dataclass
class TrainingArtifacts:
    """What every train.py returns to its caller."""

    output_dir: Path
    final_metrics: dict[str, float]
    extras: dict[str, Any]


def make_output_dir(course: str, klass: str, backbone: str, method: str, run_tag: str) -> Path:
    """Standard checkpoint layout: checkpoints/<course>/<class>/<backbone>/<method>/<tag>/."""
    backbone_safe = backbone.replace("/", "__")
    out = checkpoints_root() / course / klass / backbone_safe / method / run_tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def lora_target_modules(model: Any) -> list[str]:
    """Best-effort guess of LoRA target modules per architecture.

    Returns a conservative set that covers most decoder & encoder backbones
    used in the course. Override in class config if you need fine control.
    """
    arch = getattr(model.config, "architectures", [None])
    name = (arch[0] if arch else type(model).__name__).lower()
    if "llama" in name or "smollm" in name or "mistral" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "bert" in name:
        return ["query", "key", "value", "dense"]
    return ["q_proj", "k_proj", "v_proj"]


def freeze_base(model: Any) -> int:
    """Set requires_grad=False on every parameter; return count frozen."""
    n = 0
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad = False
            n += 1
    return n


def trainable_param_count(model: Any) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def classification_metrics(predictions: Any, labels: Any) -> dict[str, float]:
    """Accuracy + macro-F1. Pure numpy to avoid pulling sklearn."""
    import numpy as np

    preds = np.asarray(predictions)
    labs = np.asarray(labels)
    accuracy = float((preds == labs).mean())

    # Macro-F1 = mean over classes of F1 = 2*P*R / (P+R)
    classes = np.unique(np.concatenate([preds, labs]))
    f1s = []
    for c in classes:
        tp = int(((preds == c) & (labs == c)).sum())
        fp = int(((preds == c) & (labs != c)).sum())
        fn = int(((preds != c) & (labs == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return {"accuracy": accuracy, "f1_macro": float(np.mean(f1s)) if f1s else 0.0}
