"""For Course 1 ch5 class 1, train.py performs both training and retrieval eval."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
