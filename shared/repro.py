"""Reproducibility helpers: seeding, environment manifest, config hashing."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA when available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch = import_module("torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def config_hash(config: dict[str, Any]) -> str:
    """Stable SHA-256 prefix for a config dict (12 hex chars)."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _safe_version(module_name: str) -> str | None:
    try:
        mod = import_module(module_name)
    except ImportError:
        return None
    return getattr(mod, "__version__", None)


def _git_sha() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return sha.decode().strip() or None


def env_manifest() -> dict[str, Any]:
    """Snapshot of the runtime — saved next to every result for reproducibility."""
    cuda = None
    try:
        torch = import_module("torch")
        cuda = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "version": getattr(torch.version, "cuda", None),
        }
    except ImportError:
        pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_sha": _git_sha(),
        "versions": {
            "torch": _safe_version("torch"),
            "transformers": _safe_version("transformers"),
            "datasets": _safe_version("datasets"),
            "peft": _safe_version("peft"),
            "trl": _safe_version("trl"),
            "accelerate": _safe_version("accelerate"),
            "sentence_transformers": _safe_version("sentence_transformers"),
        },
        "cuda": cuda,
    }
