"""Correction for Course 1 / ch5 / class 1 exercises."""

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

    print("\n=== Exercise 1: Sweep batch size ===")
    print("NEW vs train.py: vary `train.per_device_batch` and compare retrieval MRR.")
    print("Expected output tip: MRR often improves with larger batches because InfoNCE sees more negatives, but the gains are not usually linear forever.")

    print("\n=== Exercise 2: Compare backbones ===")
    print("NEW vs train.py: rerun the same contrastive trainer on multiple encoder backbones and compare `extras.delta_mrr` in the result JSONs.")
    print("Expected output tip: a retrieval-tuned backbone like BGE may still improve, but often by less than a weaker generic encoder.")

    print("\n=== Exercise 3: Hard-negative mining ===")
    print("NEW vs train.py: extend `_retrieval_metrics` so it also records the closest wrong positive for each anchor, then feed that as an explicit negative.")
    print("Expected output tip: if the negatives are genuinely hard, MRR can jump, but the training signal also becomes more brittle.")
    print("\nCurrent batch size:", cfg.get("train", {}).get("per_device_batch"))


if __name__ == "__main__":
    main()
