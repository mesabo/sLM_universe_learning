"""Correction for Course 1 / ch3 / class 1 exercises."""

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

    print("\n=== Exercise 1: Measure the memory win ===")
    print("NEW vs train.py: compare GPU memory usage between full SFT and QLoRA while the model is loaded.")
    print("Expected output tip: QLoRA should substantially reduce base-model memory, though real-world savings are often smaller than the idealized 4x headline.")

    print("\n=== Exercise 2: Turn off double quantization ===")
    print("NEW vs train.py: override `quantization.bnb_4bit_use_double_quant=false` and rerun.")
    print("Expected output tip: memory and loss usually move only a little, which is why double quantization is considered almost free.")

    print("\n=== Exercise 3: QLoRA on the 135M model ===")
    print("NEW vs train.py: shrink the backbone and compare wall-clock overhead against plain LoRA.")
    print("Expected output tip: at 135M scale, QLoRA can look less compelling because quantization overhead may cancel part of the memory benefit.")
    print("\nCurrent quantization config:", cfg.get("quantization"))


if __name__ == "__main__":
    main()
