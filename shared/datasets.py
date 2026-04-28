"""Dataset spec layer.

A class never calls `datasets.load_dataset` directly; it loads a `DatasetSpec`
from YAML and uses one of the adapters below. This way swapping datasets
is a config change.

YAML spec example:

```yaml
dataset:
  hf_id: ag_news
  split:
    train: train
    eval: test
  text_field: text
  label_field: label
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import load_yaml


@dataclass
class DatasetSpec:
    hf_id: str
    split: dict[str, str]
    text_field: str = "text"
    label_field: str | None = None
    prompt_template: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_spec(path_or_dict: str | dict[str, Any]) -> DatasetSpec:
    """Load a DatasetSpec from a YAML path or an already-loaded dict."""
    if isinstance(path_or_dict, str):
        raw = load_yaml(path_or_dict)
        if "dataset" in raw:
            raw = raw["dataset"]
    else:
        raw = path_or_dict
    return DatasetSpec(
        hf_id=raw["hf_id"],
        split=raw.get("split", {"train": "train", "eval": "test"}),
        text_field=raw.get("text_field", "text"),
        label_field=raw.get("label_field"),
        prompt_template=raw.get("prompt_template"),
        extra={k: v for k, v in raw.items() if k not in {
            "hf_id", "split", "text_field", "label_field", "prompt_template",
        }},
    )


def to_classification(spec: DatasetSpec):
    """Return train/eval HF Datasets with `text` and `label` columns."""
    from datasets import load_dataset

    if spec.label_field is None:
        raise ValueError(f"label_field required for classification, got {spec}")
    out = {}
    for role, split_name in spec.split.items():
        ds = load_dataset(spec.hf_id, split=split_name)
        ds = ds.rename_columns(
            {c: t for c, t in [(spec.text_field, "text"), (spec.label_field, "label")] if c != t}
        )
        out[role] = ds
    return out


def to_chat(spec: DatasetSpec):
    """Render each row through `prompt_template` to a `text` field for SFT."""
    from datasets import load_dataset

    if spec.prompt_template is None:
        raise ValueError(f"prompt_template required for chat, got {spec}")

    def _render(example: dict) -> dict:
        return {"text": spec.prompt_template.format(**example)}

    out = {}
    for role, split_name in spec.split.items():
        ds = load_dataset(spec.hf_id, split=split_name)
        ds = ds.map(_render)
        out[role] = ds
    return out


def to_pairs(spec: DatasetSpec):
    """For embedding / contrastive training: yields `anchor`, `positive` columns.

    Expects spec.extra['anchor_field'] and spec.extra['positive_field'].
    """
    from datasets import load_dataset

    anchor = spec.extra.get("anchor_field")
    positive = spec.extra.get("positive_field")
    if not (anchor and positive):
        raise ValueError(f"to_pairs requires anchor_field/positive_field in spec.extra, got {spec}")
    out = {}
    for role, split_name in spec.split.items():
        ds = load_dataset(spec.hf_id, split=split_name)
        ds = ds.rename_columns({anchor: "anchor", positive: "positive"})
        out[role] = ds
    return out
