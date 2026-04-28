"""Lightweight logging — rich console + JSONL file."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from rich.logging import RichHandler

    _RICH = True
except ImportError:
    _RICH = False


def _level() -> int:
    return getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)


def get_logger(name: str = "slm") -> logging.Logger:
    """Console logger. Use the same name across a class for grouping."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(_level())
    handler = (
        RichHandler(rich_tracebacks=True, markup=False, show_path=False)
        if _RICH
        else logging.StreamHandler(sys.stderr)
    )
    handler.setLevel(_level())
    if not _RICH:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class JsonlWriter:
    """Append-only JSONL writer for per-step metrics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, **fields: Any) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **fields}
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
