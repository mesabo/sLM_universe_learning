"""Correction for Course 3 / ch1 / class 1 exercises."""

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
    print("NEW vs train.py: vary `batches.batch_size` and compare the logged `tokens_per_second` metric.")
    print("Expected output tip: throughput should climb at first, then flatten once the hardware is saturated.")

    print("\n=== Exercise 2: Long-context regime ===")
    print("NEW vs train.py: replace the short prompt with a very long one and cut `max_new_tokens` so prefill dominates the generation budget.")
    print("Expected output tip: `prefill_fraction` should rise sharply, often above 0.9, because most work happens before decoding starts.")

    print("\n=== Exercise 3: vLLM comparison ===")
    print("NEW vs train.py: keep the benchmark logic but send the requests to a vLLM server instead of local HF generation.")
    print("Expected output tip: the throughput gap is usually modest at tiny batches and clearer at larger concurrent batches.")
    print("\nConfigured batch size:", cfg.get("batches", {}).get("batch_size"))


if __name__ == "__main__":
    main()
