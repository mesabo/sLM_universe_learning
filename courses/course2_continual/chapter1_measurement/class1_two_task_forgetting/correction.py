"""Correction for Course 2 / ch1 / class 1 exercises."""

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

    print("\n=== Exercise 1: Read the matrix ===")
    print("NEW vs train.py: inspect `extras.history_matrix` from the result JSON and recompute BWT and average accuracy by hand.")
    print("Expected output tip: your manual BWT should match the saved metric exactly if you index the right stage/task cells.")

    print("\n=== Exercise 2: Closer-domain Task B ===")
    print("NEW vs train.py: replace Task B with SST-2 in the YAML and keep the same sequential-training procedure.")
    print("Expected output tip: forgetting is often smaller when tasks are more related, though the exact magnitude depends on label remapping and sample size.")

    print("\n=== Exercise 3: Freeze the backbone ===")
    print("NEW vs train.py: this is already supported by the baseline through `freeze_backbone`; the correction is to run it deliberately and compare BWT.")
    print("Expected output tip: freezing the backbone often reduces forgetting but also caps forward transfer because only the head can move.")
    print("\nfreeze_backbone currently:", cfg.get("freeze_backbone"))


if __name__ == "__main__":
    main()
