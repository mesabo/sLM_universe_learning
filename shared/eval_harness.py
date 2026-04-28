"""Evaluation harness — every class writes a normalized `<method>.json`.

```python
from shared.eval_harness import run_eval

def predict(batch): ...   # returns list[dict] aligned with the dataset

run_eval(
    method="lora-r16",
    backbone="HuggingFaceTB/SmolLM2-135M-Instruct",
    course="course1_finetuning",
    klass="ch2_lora",
    task="ag_news",
    config={"lr": 2e-4, "rank": 16, "seed": 42},
    metrics={"accuracy": 0.92, "f1_macro": 0.91},
    expected_band={"accuracy": [0.85, 0.97]},
)
```

If a metric falls outside its declared band, `run_eval` raises
`MetricBandFailure` and exits the process non-zero (when called from a
`__main__` script). That's the per-class verification contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .paths import result_path
from .repro import config_hash, env_manifest


class MetricBandFailure(RuntimeError):
    """Raised when a measured metric falls outside its expected band."""


def _check_bands(metrics: dict[str, float], bands: dict[str, list[float]]) -> list[str]:
    failures: list[str] = []
    for key, (lo, hi) in bands.items():
        if key not in metrics:
            failures.append(f"missing metric {key!r}")
            continue
        value = float(metrics[key])
        if not (lo <= value <= hi):
            failures.append(f"{key}={value:.4f} outside [{lo}, {hi}]")
    return failures


def run_eval(
    *,
    method: str,
    backbone: str,
    course: str,
    klass: str,
    task: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    expected_band: dict[str, list[float]] | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Persist a result JSON; verify expected bands; return the path."""
    record = {
        "method": method,
        "backbone": backbone,
        "course": course,
        "class": klass,
        "task": task,
        "config_hash": config_hash(config),
        "config": config,
        "metrics": metrics,
        "expected_band": expected_band or {},
        "env": env_manifest(),
    }
    if extras:
        record["extras"] = extras

    path = result_path(course=course, klass=klass, backbone=backbone, task=task, method=method)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    failures = _check_bands(metrics, expected_band or {})
    if failures:
        msg = f"Metric band failure for {method}/{backbone}: " + "; ".join(failures)
        if hasattr(sys.modules["__main__"], "__file__"):
            print(msg, file=sys.stderr)
            sys.exit(1)
        raise MetricBandFailure(msg)
    return path


def aggregate(glob: str = "results/full/**/*.json"):
    """Collect all per-(method, backbone) JSONs into a DataFrame.

    Lazy import of pandas so the rest of the harness works without it.
    """
    import pandas as pd

    from .paths import project_root

    rows: list[dict[str, Any]] = []
    for p in project_root().glob(glob):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        flat = {
            "backbone": data.get("backbone"),
            "course": data.get("course"),
            "class": data.get("class"),
            "task": data.get("task"),
            "method": data.get("method"),
            "config_hash": data.get("config_hash"),
            **{f"metric.{k}": v for k, v in (data.get("metrics") or {}).items()},
        }
        rows.append(flat)
    return pd.DataFrame(rows)
