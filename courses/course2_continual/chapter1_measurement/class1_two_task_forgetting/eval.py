"""For Course 2 ch1 class 1, train.py performs the full sequential measurement.

This file exists for symmetry with other classes — re-running it executes
the same pipeline (re-trains and re-measures). To recompute on a saved
multi-task checkpoint, write a custom script; the simplest reproduction
is just `bash run.sh` again.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
