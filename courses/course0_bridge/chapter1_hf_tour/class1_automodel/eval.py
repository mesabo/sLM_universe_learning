"""For this sanity class, eval == train (the forward pass IS the verification).

Real training classes in Course 1+ separate them. This file exists so the
per-class template (`train.py` + `eval.py`) is uniform across all classes.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
