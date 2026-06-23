"""Correction for Course 1 / ch2 / class 1 exercises."""

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

    print("\n=== Exercise 1: Read the trainable-params print-out ===")
    print("NEW vs train.py: use the already-logged LoRA ratio to verify the parameter-count math by hand.")
    print("Expected output tip: the reported trainable percentage should stay well below 1%, which is the core point of LoRA.")

    print("\n=== Exercise 2: Sweep r ===")
    print("NEW vs train.py: only `lora.r` changes; the rest of the pipeline stays fixed.")
    print("Expected output tip: larger `r` usually improves capacity at the cost of a bigger adapter file. At some point the adapter stops feeling 'tiny'.")
    print("Suggested command:")
    print("for r in 4 16 64; do bash run.sh --config configs/default.yaml lora.r=$r; done")

    print("\n=== Exercise 3: Multi-adapter loading ===")
    print("NEW vs train.py: load two saved adapters onto one base model and switch between them with `set_adapter(...)`.")
    print("Expected output tip: generations from adapter A and adapter B should differ if the two fine-tuning datasets pushed the model in different directions.")
    print("\nCurrent LoRA rank:", cfg.get("lora", {}).get("r"))


if __name__ == "__main__":
    main()
