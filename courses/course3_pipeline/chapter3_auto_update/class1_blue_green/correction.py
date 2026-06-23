"""Correction for Course 3 / ch3 / class 1 exercises."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import apply_overrides, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.overrides)

    print("\n=== Exercise 1: Sweep the acceptance margin ===")
    print("NEW vs train.py: the whole blue/green pipeline stays the same; only the promotion gate margin changes.")
    print("Expected output tip: looser margins usually produce more promotions, but also make it easier for weaker candidates to replace production.")

    print("\n=== Exercise 2: Add an explicit rollback rule ===")
    print("NEW vs train.py: before drift handling starts in each cycle, compare current production accuracy against the previous cycle and call `shared.registry.rollback(...)` on a large drop.")
    print("Expected output tip: rollback should fire only under strong degradation settings, not in every normal cycle.")

    print("\n=== Exercise 3: Paired-bootstrap promotion gate ===")
    print("NEW vs train.py: replace the fixed margin rule with a bootstrap-based significance test inside the promotion decision.")
    print("Expected output tip: this usually makes promotions rarer but better justified statistically.")
    print("\nCurrent acceptance margin:", cfg.get("gate", {}).get("acceptance_margin"))


if __name__ == "__main__":
    main()
