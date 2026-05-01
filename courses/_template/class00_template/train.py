"""Class training entrypoint — fill in the body for your lesson.

Conventions (enforced by the instructor checklist in this folder's README):
  - All numeric literals other than 0 / 1 live in configs/default.yaml.
  - Every result is persisted via shared.eval_harness.run_eval, which writes
    a JSON at results/full/<backbone>/<course>/<class>/<task>/<method>.json
    and exits non-zero if the metric falls outside its declared band.
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

from shared.backbones import load_backbone
from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*",
                        help="Dotted overrides like backbone=BAAI/bge-small-en-v1.5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = get_logger("courseX.chY.classZ")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])

    bb = load_backbone(cfg["backbone"])
    log.info("backbone=%s kind=%s hidden=%d", bb.name, bb.kind, bb.hidden_size)

    # ---- TODO: implement the lesson's main computation here ------------------
    # Use bb.model / bb.tokenizer / bb.kind. Keep all hyperparameters in cfg.
    metrics: dict[str, float] = {
        "metric_a": 0.0,
    }
    # --------------------------------------------------------------------------

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
    )
    log.info("done")


if __name__ == "__main__":
    main()
