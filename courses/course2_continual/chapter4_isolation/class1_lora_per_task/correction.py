"""Correction for Course 2 / ch4 / class 1 exercises."""

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

    print("\n=== Exercise 1: Verify the head is per-adapter ===")
    print("NEW vs train.py: inspect `modules_to_save=['classifier']` by switching adapters and reading the classifier weights.")
    print("Expected output tip: the classifier rows for `ag_news` and `emotion` should differ, proving each adapter keeps its own head copy.")

    print("\n=== Exercise 2: Drop modules_to_save ===")
    print("NEW vs train.py: remove the saved classifier module from the LoRA config and rerun the same task-isolation pipeline.")
    print("Expected output tip: BWT should worsen because the head becomes shared again, even though the LoRA matrices remain separate.")

    print("\n=== Exercise 3: Three tasks ===")
    print("NEW vs train.py: generalize the hard-wired 2-task loop to N tasks by adding one adapter per task and swapping adapters at eval time.")
    print("Expected output tip: if the isolation logic is correct, forgetting should stay near zero even as the number of tasks grows.")
    print("\nCurrent modules_to_save:", cfg.get("lora", {}).get("modules_to_save"))


if __name__ == "__main__":
    main()
