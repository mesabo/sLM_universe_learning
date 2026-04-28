"""Backbone loader — single entry point used by every class.

Validates the requested name against `configs/backbones.yaml`, picks the
right HF API per `kind`, and returns a normalized `Backbone` dataclass.

```python
from shared.backbones import load_backbone
bb = load_backbone("sentence-transformers/all-MiniLM-L6-v2")
print(bb.kind, bb.hidden_size, bb.params_m)
```
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import load_backbones_config
from .paths import hf_cache

Kind = Literal["sentence-encoder", "encoder", "decoder"]


@dataclass
class Backbone:
    name: str
    kind: Kind
    model: Any
    tokenizer: Any
    hidden_size: int
    max_len: int
    params_m: int
    extra: dict[str, Any] = field(default_factory=dict)


def list_backbones() -> list[str]:
    """All backbone ids registered in configs/backbones.yaml."""
    cfg = load_backbones_config()
    return [c["id"] for c in cfg["candidates"]]


def get_backbone_spec(name: str) -> dict[str, Any]:
    cfg = load_backbones_config()
    for entry in cfg["candidates"]:
        if entry["id"] == name:
            return entry
    raise ValueError(
        f"Unknown backbone {name!r}. Known: {list_backbones()}. "
        "Edit configs/backbones.yaml to register a new one."
    )


def _resolve_dtype(dtype: str | None, kind: Kind) -> Any:
    """Convert dtype string to torch dtype; fall back to float32 on CPU."""
    import torch

    if dtype is None or dtype == "auto":
        dtype = "float32" if kind != "decoder" else "bfloat16"
    if not torch.cuda.is_available():
        return torch.float32
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in table:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {list(table)}")
    return table[dtype]


def load_backbone(
    name: str,
    *,
    dtype: str | None = None,
    device: str | None = None,
    revision: str | None = None,
) -> Backbone:
    """Load a registered backbone with the right HF API for its kind.

    `device` defaults to "cuda" when available, else "cpu".
    """
    import torch

    spec = get_backbone_spec(name)
    kind: Kind = spec["kind"]
    torch_dtype = _resolve_dtype(dtype or spec.get("dtype"), kind)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Force project-local HF cache (safety net even if HF_HOME wasn't sourced).
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    if kind == "sentence-encoder":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name, device=device, revision=revision)
        tokenizer = model.tokenizer
        # `get_embedding_dimension` (sentence-transformers >= 5.4) supersedes
        # `get_sentence_embedding_dimension`; fall back for older releases.
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        hidden_size = int(getter() or 0)
        return Backbone(
            name=name,
            kind=kind,
            model=model,
            tokenizer=tokenizer,
            hidden_size=hidden_size,
            max_len=int(spec.get("max_len", 256)),
            params_m=int(spec.get("params_m", 0)),
            extra={"pooling": spec.get("pooling")},
        )

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if kind == "decoder":
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch_dtype, revision=revision
        ).to(device)
        hidden_size = int(getattr(model.config, "hidden_size", 0))
    else:  # encoder
        model = AutoModel.from_pretrained(name, torch_dtype=torch_dtype, revision=revision).to(
            device
        )
        hidden_size = int(getattr(model.config, "hidden_size", 0))

    return Backbone(
        name=name,
        kind=kind,
        model=model,
        tokenizer=tokenizer,
        hidden_size=hidden_size,
        max_len=int(spec.get("max_len", 512)),
        params_m=int(spec.get("params_m", 0)),
    )
