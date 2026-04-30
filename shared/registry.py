"""Versioned model checkpoint registry with a `production.json` pointer.

Layout on disk:

    checkpoints/<course>/<klass>/<run_id>/
        production.json       {"version": N}             — atomically swapped
        v1/
            manifest.json     {config_hash, env, metrics, parent_version, timestamp}
            model/            — saved by `save_pretrained` (or your own writer)
        v2/
            manifest.json
            model/
        ...

Designed for Course 3 ch3 (blue/green swap) and ch5 (end-to-end pipeline),
but useful anywhere you want to keep multiple model versions on disk and
flip between them safely.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths as _paths


@dataclass
class CheckpointHandle:
    """A single registered version of a model under one run."""

    course: str
    klass: str
    run_id: str
    version: int
    path: Path                                # versioned directory
    manifest: dict = field(default_factory=dict)


def run_dir(course: str, klass: str, run_id: str) -> Path:
    """Resolve `checkpoints/<course>/<klass>/<run_id>/`. Does NOT create it.

    Uses `shared.paths.checkpoints_root()` via module lookup so test
    monkeypatching of that function is observed at call time.
    """
    return _paths.checkpoints_root() / course / klass / run_id


def init_run(course: str, klass: str, run_id: str) -> Path:
    """Create the run directory if missing. Returns the path."""
    p = run_dir(course, klass, run_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _next_version(handle_dir: Path) -> int:
    existing = [int(p.name[1:]) for p in handle_dir.iterdir()
                if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()]
    return (max(existing) + 1) if existing else 1


def register_version(course: str, klass: str, run_id: str,
                     manifest: dict, parent_version: int | None = None) -> CheckpointHandle:
    """Allocate the next vN under the run, write a manifest, and return the handle.

    The caller is responsible for writing the actual model artifacts under
    `handle.path / "model"` (use `model.save_pretrained(handle.path / "model")`
    for HF models).
    """
    rdir = init_run(course, klass, run_id)
    v = _next_version(rdir)
    vdir = rdir / f"v{v}"
    (vdir / "model").mkdir(parents=True, exist_ok=True)
    full_manifest = {
        **manifest,
        "version": v,
        "parent_version": parent_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (vdir / "manifest.json").write_text(
        json.dumps(full_manifest, indent=2, default=str), encoding="utf-8"
    )
    return CheckpointHandle(
        course=course, klass=klass, run_id=run_id, version=v,
        path=vdir, manifest=full_manifest,
    )


def list_versions(course: str, klass: str, run_id: str) -> list[CheckpointHandle]:
    """Return all registered versions under the run, sorted ascending."""
    rdir = run_dir(course, klass, run_id)
    if not rdir.exists():
        return []
    out: list[CheckpointHandle] = []
    for p in sorted(rdir.iterdir(), key=lambda x: x.name):
        if not (p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()):
            continue
        manifest_path = p / "manifest.json"
        manifest: dict = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
        out.append(CheckpointHandle(
            course=course, klass=klass, run_id=run_id,
            version=int(p.name[1:]), path=p, manifest=manifest,
        ))
    return sorted(out, key=lambda h: h.version)


def get_production(course: str, klass: str, run_id: str) -> CheckpointHandle | None:
    """Resolve the version that `production.json` currently points at."""
    rdir = run_dir(course, klass, run_id)
    pointer = rdir / "production.json"
    if not pointer.is_file():
        return None
    try:
        v = int(json.loads(pointer.read_text(encoding="utf-8"))["version"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None
    versions = {h.version: h for h in list_versions(course, klass, run_id)}
    return versions.get(v)


def promote(course: str, klass: str, run_id: str, version: int) -> None:
    """Atomically swap `production.json` to point at `vN`.

    Raises `ValueError` if `vN` doesn't exist. The write is atomic via a
    `tempfile.NamedTemporaryFile` in the same directory + `os.replace`,
    so a reader never sees a half-written pointer.
    """
    rdir = run_dir(course, klass, run_id)
    vdir = rdir / f"v{version}"
    if not vdir.is_dir():
        raise ValueError(f"version v{version} does not exist under {rdir}")
    pointer = rdir / "production.json"
    payload = json.dumps({"version": int(version)}, indent=2) + "\n"
    fd, tmp_path = tempfile.mkstemp(prefix=".production-", suffix=".json", dir=str(rdir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, pointer)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def rollback(course: str, klass: str, run_id: str, to_version: int) -> None:
    """Same as `promote(to_version)` — exists as an explicit operational verb."""
    promote(course, klass, run_id, to_version)


def write_decision_log(course: str, klass: str, run_id: str,
                       entry: dict[str, Any]) -> Path:
    """Append one JSON line to `decisions.jsonl` under the run directory."""
    rdir = init_run(course, klass, run_id)
    log_path = rdir / "decisions.jsonl"
    line = json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        **entry,
    }, default=str)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return log_path
