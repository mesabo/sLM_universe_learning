"""Correction for Course 2 / ch3 / class 1 exercises."""

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

    print("\n=== Exercise 1: Sweep lambda ===")
    print("NEW vs train.py: vary `ewc.lambda` while keeping the rest of the continual-learning pipeline fixed.")
    print("Expected output tip: lambda that is too small behaves like the baseline, while lambda that is too large protects Task A but can hurt Task B learning.")

    print("\n=== Exercise 2: Compare to chapter 1 and chapter 2 ===")
    print("NEW vs train.py: align the random seed across baseline, replay, and EWC runs so the BWT / accuracy table is actually comparable.")
    print("Expected output tip: replay and EWC should both beat the catastrophic baseline on BWT, but they trade data storage against regularization strength.")

    print("\n=== Exercise 3: Replay + EWC together ===")
    print("NEW vs train.py: port the replay-mixing logic from chapter 2 into the EWC trainer so both defenses act at once.")
    print("Expected output tip: a combined method can help, but returns often diminish once either replay or EWC is already strong enough.")
    print("\nCurrent EWC lambda:", cfg.get("ewc", {}).get("lambda"))


if __name__ == "__main__":
    main()
