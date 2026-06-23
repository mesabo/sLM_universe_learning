"""Correction for Course 3 / ch4 / class 1 exercises."""

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

    print("\n=== Exercise 1: Sweep the OOD ramp ===")
    print("NEW vs train.py: modify `live.shift_schedule` to change how quickly OOD traffic ramps across ticks.")
    print("Expected output tip: `psi_max` should generally rise with a steeper OOD ramp, while a very flat schedule may never trigger a strong alarm.")

    print("\n=== Exercise 2: Refresh the canary periodically ===")
    print("NEW vs train.py: replace a fraction of the canary set every K ticks using new labeled live data.")
    print("Expected output tip: if the canary stays more representative, the accuracy-vs-drift relationship often becomes tighter and easier to interpret.")

    print("\n=== Exercise 3: MMD as an alternative to PSI ===")
    print("NEW vs train.py: run `shared.drift.mmd(...)` beside PSI using the same live/baseline embeddings.")
    print("Expected output tip: MMD may detect some shifts earlier, but it is usually computationally heavier than PSI.")
    print("\nCurrent shift schedule:", cfg.get("live", {}).get("shift_schedule"))


if __name__ == "__main__":
    main()
