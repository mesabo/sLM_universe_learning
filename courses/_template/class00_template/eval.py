"""Class eval entrypoint — used to recompute metrics on a saved checkpoint.

For lessons where train.py performs the only meaningful evaluation, you can
keep this as a one-line forwarder:

    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")

Otherwise, mirror the train.py argparse, accept --ckpt, and reuse shared.* helpers.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
