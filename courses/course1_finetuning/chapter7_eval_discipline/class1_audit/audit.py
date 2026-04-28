"""Course 1 / ch7 / class 1 — eval discipline audit.

Read-only walk over result JSONs, flagging the most common discipline holes:
missing config_hash, missing seed, undeclared bands, out-of-band metrics,
single-seed cells in published comparisons, and path collisions.

  python audit.py --config configs/default.yaml [--output PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import project_root
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None,
                        help="Optional path for the rendered Markdown report; overrides config.output_md")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _flag(violations: list, rule: str, path: str, message: str) -> None:
    violations.append({"rule": rule, "path": path, "message": message})


def _audit_single(data: dict, path: str) -> list[dict]:
    """Per-file checks. Returns a list of violations from this file."""
    out: list[dict] = []
    if not data.get("config_hash"):
        _flag(out, "missing_config_hash", path, "no `config_hash` in result JSON")
    cfg = data.get("config") or {}
    if "seed" not in cfg:
        _flag(out, "missing_seed", path, "no `seed` in config — runs not reproducible")

    metrics = data.get("metrics") or {}
    bands = data.get("expected_band") or {}
    for key in metrics:
        if key not in bands:
            _flag(out, "missing_band", path, f"metric `{key}` has no expected_band entry")
    for key, val in metrics.items():
        if key not in bands:
            continue
        try:
            lo, hi = bands[key]
            v = float(val)
            if not (float(lo) <= v <= float(hi)):
                _flag(out, "out_of_band", path,
                      f"metric `{key}`={v:.4f} outside [{lo}, {hi}]")
        except (TypeError, ValueError):
            _flag(out, "malformed_band", path,
                  f"band for `{key}` is malformed: {bands[key]!r}")
    return out


def _cell_key(data: dict) -> tuple:
    return (
        data.get("backbone") or "",
        data.get("course") or "",
        data.get("class") or "",
        data.get("task") or "",
        data.get("method") or "",
    )


def _is_excluded(class_id: str, excludes: list[str]) -> bool:
    return any(s in (class_id or "") for s in excludes)


def _render_md(violations: list[dict]) -> str:
    if not violations:
        return "_(no violations — your eval discipline is clean)_"
    header = ["rule", "path", "message"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    by_rule = defaultdict(list)
    for v in violations:
        by_rule[v["rule"]].append(v)
    for rule in sorted(by_rule):
        for v in by_rule[rule]:
            cells = [rule, v["path"], v["message"]]
            cells = [c.replace("|", "\\|") for c in cells]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    log = get_logger("course1.ch7.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])

    pattern = cfg["glob"]["pattern"]
    excludes = list(cfg.get("single_seed_excludes") or [])
    paths = sorted(project_root().glob(pattern))
    log.info("scanning %d JSON files under %r", len(paths), pattern)

    violations: list[dict] = []
    cell_seeds: defaultdict = defaultdict(set)
    cell_paths: defaultdict = defaultdict(list)
    n_audited = 0

    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            _flag(violations, "unreadable", str(p.relative_to(project_root())),
                  f"could not parse JSON: {e}")
            continue
        if _is_excluded(data.get("class") or "", excludes):
            continue
        n_audited += 1
        rel = str(p.relative_to(project_root()))
        violations.extend(_audit_single(data, rel))
        key = _cell_key(data)
        cell_seeds[key].add((data.get("config") or {}).get("seed"))
        cell_paths[key].append(rel)

    # Cross-cell checks.
    for key, seeds in cell_seeds.items():
        clean_seeds = {s for s in seeds if s is not None}
        if len(clean_seeds) <= 1:
            backbone, course, klass, task, method = key
            if not _is_excluded(klass, excludes):
                _flag(violations, "single_seed",
                      "; ".join(cell_paths[key]),
                      f"cell ({backbone}, {course}, {klass}, {task}, {method}) "
                      f"has only {len(clean_seeds)} seed(s); need >= 3 to claim significance")
    for key, files in cell_paths.items():
        if len(files) > len(set(files)):
            _flag(violations, "path_collision",
                  "; ".join(files),
                  "more than one JSON wrote to the same path (only the last wins)")

    md = _render_md(violations)
    print("\n" + md + "\n")

    out_path = Path(args.output or cfg.get("output_md") or "")
    if str(out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md + "\n", encoding="utf-8")
        log.info("wrote markdown report -> %s", out_path)

    metrics = {
        "n_files_audited": n_audited,
        "n_unique_cells": len(cell_seeds),
        "n_violations": len(violations),
    }
    log.info("metrics=%s", metrics)

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "violations": violations,
            "by_rule_counts": {r: sum(1 for v in violations if v["rule"] == r)
                               for r in {v["rule"] for v in violations}},
        },
    )


if __name__ == "__main__":
    main()
