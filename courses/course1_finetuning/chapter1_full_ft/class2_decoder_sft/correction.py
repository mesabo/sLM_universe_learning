"""Correction for Course 1 / ch1 / class 2 exercises."""

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

    print("\n=== Exercise 1: Generate before and after ===")
    print("NEW vs train.py: compare one prompt before training and from the saved checkpoint after training.")
    print("Expected output tip: both outputs should remain stylistically similar, but the post-SFT answer should better match the fine-tuning domain.")
    print("Suggested workflow: capture one base-model completion, run smoke mode, then call `model.generate(...)` from the saved checkpoint on the same prompt.")

    print("\n=== Exercise 2: Bigger backbone ===")
    print("NEW vs train.py: swap `backbone=HuggingFaceTB/SmolLM2-360M-Instruct` and re-run the same SFT pipeline.")
    print("Expected output tip: eval loss may improve modestly, but GPU memory and wall-clock cost should increase noticeably.")

    print("\n=== Exercise 3: Turn on packing ===")
    print("NEW vs train.py: this exercise is just a config toggle because `SFTConfig(..., packing=cfg['train']['packing'])` already exists in the baseline.")
    print("Expected output tip: packing often improves utilization for many short examples, but can complicate behavior when sequence lengths vary a lot.")
    print("\nCurrent packing setting:", cfg.get("train", {}).get("packing"))


if __name__ == "__main__":
    main()
