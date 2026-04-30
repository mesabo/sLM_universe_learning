"""For Course 3 ch3 class 1, train.py runs the full bootstrap + cycles pipeline."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
