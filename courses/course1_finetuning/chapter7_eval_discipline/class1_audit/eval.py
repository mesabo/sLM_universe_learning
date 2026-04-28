"""For Course 1 ch7 class 1, audit.py is the entire pipeline."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("audit.py")), run_name="__main__")
