"""Single source for all on-disk paths.

The course is launched from the repo root. We resolve everything relative to
the project root (the directory containing `pyproject.toml`) so scripts can
be invoked from anywhere.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Walk upward from this file until we find pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"pyproject.toml not found above {here}")


def configs_dir() -> Path:
    return project_root() / "configs"


def shared_dir() -> Path:
    return project_root() / "shared"


def courses_dir() -> Path:
    return project_root() / "courses"


def results_root() -> Path:
    root = Path(os.environ.get("RESULTS_ROOT", project_root() / "results"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def runs_root() -> Path:
    root = project_root() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def checkpoints_root() -> Path:
    root = project_root() / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def hf_cache() -> Path:
    """HF_HOME — project-local. Set HF_HOME env to override."""
    cache = Path(os.environ.get("HF_HOME", project_root() / ".cache" / "huggingface"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def result_path(course: str, klass: str, backbone: str, task: str, method: str) -> Path:
    """Standard layout: results/full/<backbone>/<course>/<class>/<task>/<method>.json."""
    backbone_safe = backbone.replace("/", "__")
    out = results_root() / "full" / backbone_safe / course / klass / task
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{method}.json"
