"""YAML config loader with `!include` support and dotted-key overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .paths import configs_dir, project_root


class _IncludeLoader(yaml.SafeLoader):
    """SafeLoader extended with `!include path/to/other.yaml`."""

    def __init__(self, stream):  # type: ignore[no-untyped-def]
        self._root = Path(getattr(stream, "name", project_root() / "anon.yaml")).parent
        super().__init__(stream)


def _construct_include(loader: _IncludeLoader, node: yaml.Node) -> Any:
    rel = loader.construct_scalar(node)  # type: ignore[arg-type]
    target = (loader._root / rel).resolve()
    return load_yaml(target)


_IncludeLoader.add_constructor("!include", _construct_include)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file with `!include` support."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_IncludeLoader)
    return data or {}


def load_default_config() -> dict[str, Any]:
    """Load `configs/defaults.yaml`."""
    return load_yaml(configs_dir() / "defaults.yaml")


def load_backbones_config() -> dict[str, Any]:
    """Load `configs/backbones.yaml`."""
    return load_yaml(configs_dir() / "backbones.yaml")


def load_hardware_config() -> dict[str, Any]:
    """Load `configs/hardware.yaml`, with env overrides for cuda_devices."""
    cfg = load_yaml(configs_dir() / "hardware.yaml")
    if "CUDA_DEVICES" in os.environ:
        cfg["cuda_devices"] = [int(x) for x in os.environ["CUDA_DEVICES"].split(",") if x.strip()]
        cfg["max_parallel"] = len(cfg["cuda_devices"]) * cfg.get("jobs_per_gpu", 1)
    return cfg


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply CLI overrides like `train.lr=1e-4 batch.train_per_device=16`.

    Returns a new dict; does not mutate input.
    """
    import copy

    out = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        cursor: Any = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(raw)
    return out
