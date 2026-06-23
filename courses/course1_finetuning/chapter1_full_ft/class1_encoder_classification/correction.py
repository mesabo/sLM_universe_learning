"""Correction for Course 1 / ch1 / class 1 exercises.

Same bootstrap spirit as ``train.py``; the new parts focus only on the three
exercise extensions.
"""

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

    print("\n=== Exercise 1: Sweep the backbone ===")
    print("NEW vs train.py: run the same encoder-classification pipeline across several backbones and compare the final metrics.")
    print("Expected output tip: a small table of backbone -> accuracy/f1. The exact winner may vary, but all runs should complete with valid metrics.")
    print("Suggested command:")
    print('for bb in "sentence-transformers/all-MiniLM-L6-v2" "BAAI/bge-small-en-v1.5" "thenlper/gte-small"; do')
    print('  python train.py --config configs/default.yaml backbone="$bb"')
    print("done")

    print("\n=== Exercise 2: Linear probe baseline ===")
    print("NEW vs train.py: add a `freeze_base` config knob and call `shared.training.freeze_base(model.base_model)` before training.")
    print("Expected output tip: trainable parameters should collapse to the classifier head only, accuracy should usually drop a bit, and memory use should improve.")
    print("Implementation hint:")
    print("if cfg.get('freeze_base'):\n    from shared.training import freeze_base\n    freeze_base(model.base_model)")

    print("\n=== Exercise 3: Deeper classification head ===")
    print("NEW vs train.py: replace the one-layer `model.classifier` with a small MLP head.")
    print("Expected output tip: the model should still train normally, but AG News often shows little or no gain because the backbone already solves most of the task.")
    print("Implementation hint:")
    print("model.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, num_labels))")
    print("\nLoaded config backbone:", cfg.get("backbone"))


if __name__ == "__main__":
    main()
