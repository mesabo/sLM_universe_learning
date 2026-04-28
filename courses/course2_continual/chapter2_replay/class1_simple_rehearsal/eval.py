"""For Course 2 ch2 class 1, train.py performs the full replay pipeline.

Re-running this entrypoint repeats the sequential training; for a true
re-eval of a saved checkpoint, write a custom script. The simplest
reproduction is just `bash run.sh`.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
