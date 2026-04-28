"""Grid expander for the sharded launcher.

Reads a YAML grid file like:

```yaml
script: courses/course1_finetuning/chapter2_lora/class1/train.py
grid:
  backbone:
    - sentence-transformers/all-MiniLM-L6-v2
    - HuggingFaceTB/SmolLM2-135M-Instruct
  seed: [42, 1337]
  lr: [1e-4, 2e-4]
fixed:
  epochs: 1
```

Emits one shell command line per cell of the grid (Cartesian product),
to stdout. The bash sharder pipes those into `xargs -P` round-robin
across `cuda_devices`.
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import sys
from typing import Any

from .config import load_yaml


def _flatten_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def expand(spec_path: str) -> list[str]:
    spec = load_yaml(spec_path)
    script = spec.get("script")
    if not script:
        raise ValueError(f"{spec_path}: missing top-level `script` field")
    fixed = spec.get("fixed") or {}
    grid = spec.get("grid") or {}

    cells = _flatten_grid(grid)
    lines: list[str] = []
    for cell in cells:
        merged = {**fixed, **cell}
        overrides = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in merged.items())
        lines.append(f"python {shlex.quote(script)} {overrides}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expand a grid spec to shell job lines.")
    parser.add_argument("spec", help="Path to a grid YAML")
    args = parser.parse_args(argv)
    for line in expand(args.spec):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
