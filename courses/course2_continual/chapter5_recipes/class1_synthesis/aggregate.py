"""Course 2 / ch5 / class 1 — synthesis aggregator.

Reads ch1-4 result JSONs and emits a Markdown comparison table. No model
loading, no training. The verification band asserts at least one prior
chapter's result was found.

  python aggregate.py --config configs/default.yaml [--output PATH]
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
import json
from pathlib import Path
from typing import Any

from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import project_root
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None,
                        help="Optional path for the rendered Markdown table; overrides config.output_md")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _matches(class_id: str, includes: list[str], excludes: list[str]) -> bool:
    if any(s in class_id for s in excludes):
        return False
    if not includes:
        return True
    return any(s in class_id for s in includes)


def _collect_rows(cfg: dict, log) -> list[dict[str, Any]]:
    pattern = cfg["glob"]["pattern"]
    includes = list(cfg.get("include_class_substrings") or [])
    excludes = list(cfg.get("exclude_class_substrings") or [])
    paths = sorted(project_root().glob(pattern))
    log.info("scanning %d result JSON(s) under glob %r", len(paths), pattern)
    rows = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("skipping %s: %s", p, e)
            continue
        klass = data.get("class") or ""
        if not _matches(klass, includes, excludes):
            continue
        row = {
            "backbone": data.get("backbone"),
            "class": klass,
            "method": data.get("method"),
            "task": data.get("task"),
            "metrics": data.get("metrics") or {},
            "path": str(p.relative_to(project_root())),
        }
        rows.append(row)
    log.info("kept %d rows after include/exclude filtering", len(rows))
    return rows


def _render_md(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_(no rows)_"
    header = ["backbone", "class", "method"] + columns
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    rows_sorted = sorted(rows, key=lambda r: (r["backbone"] or "", r["class"] or "", r["method"] or ""))
    for r in rows_sorted:
        cells = [r["backbone"] or "", r["class"] or "", r["method"] or ""]
        for col in columns:
            v = r["metrics"].get(col)
            if isinstance(v, float):
                cells.append(f"{v:+.4f}" if col == "BWT" else f"{v:.4f}")
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    log = get_logger("course2.ch5.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])

    rows = _collect_rows(cfg, log)
    columns = list(cfg["columns"])
    table_md = _render_md(rows, columns)
    print("\n" + table_md + "\n")

    out_path = Path(args.output or cfg.get("output_md") or "")
    if str(out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table_md + "\n", encoding="utf-8")
        log.info("wrote markdown table -> %s", out_path)

    unique_methods = len({r["method"] for r in rows if r["method"]})
    unique_backbones = len({r["backbone"] for r in rows if r["backbone"]})
    metrics = {
        "n_methods_aggregated": len(rows),
        "unique_methods": unique_methods,
        "unique_backbones": unique_backbones,
    }
    log.info("metrics=%s", metrics)

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "rows": rows,
            "columns": columns,
            "markdown_path": str(out_path) if str(out_path) else None,
            "table_preview": table_md,
        },
    )


if __name__ == "__main__":
    main()
