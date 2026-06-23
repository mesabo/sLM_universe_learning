"""Correction for Course 3 / ch2 / class 1 exercises."""

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

    print("\n=== Exercise 1: Sweep the round count ===")
    print("NEW vs train.py: keep the same random vs uncertainty comparison, but vary `active.n_rounds`.")
    print("Expected output tip: `delta_accuracy` may grow early, then plateau once the model has already consumed the most informative labels.")

    print("\n=== Exercise 2: Margin sampling ===")
    print("NEW vs train.py: add a `_query_margin(...)` strategy that chooses rows with the smallest `p_top1 - p_top2` gap.")
    print("Expected output tip: margin sampling and entropy sampling are often close; one may edge out the other depending on how peaky the model probabilities are.")
    print("Implementation hint: compute sorted class probabilities per row and rank by `(top1 - top2)` ascending.")

    print("\n=== Exercise 3: Diversity-aware querying ===")
    print("NEW vs train.py: cluster the top uncertain candidates and query one from each cluster instead of taking raw entropy order.")
    print("Expected output tip: this helps most when the unlabeled pool contains many near-duplicates.")
    print("\nConfigured number of rounds:", cfg.get("active", {}).get("n_rounds"))


if __name__ == "__main__":
    main()
