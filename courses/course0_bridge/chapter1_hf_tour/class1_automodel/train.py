"""Course 0 / ch1 / class 1 — HF ecosystem sanity check.

This class has no real training step; "training" is just a proof that the
HF stack is wired correctly. We load a backbone via `shared.backbones`,
run a single forward pass, and persist a result JSON.

Run via `run.sh`; do not invoke directly without the right working dir.
"""

from __future__ import annotations

import argparse

from shared.backbones import load_backbone
from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to class config YAML")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted overrides like backbone=BAAI/bge-small-en-v1.5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = get_logger("course0.ch1.class1")

    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg.get("seed", 0))

    backbone_name = cfg["backbone"]
    log.info("loading backbone: %s", backbone_name)
    bb = load_backbone(backbone_name)
    log.info("kind=%s hidden=%d max_len=%d params_m=%d",
             bb.kind, bb.hidden_size, bb.max_len, bb.params_m)

    prompt = cfg["prompt"]
    forward_ok = _run_forward(bb, prompt)
    log.info("forward_ok=%s", bool(forward_ok))

    run_eval(
        method=cfg["method"],
        backbone=backbone_name,
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics={
            "forward_ok": int(bool(forward_ok)),
            "hidden_size_ok": int(bb.hidden_size > 0),
        },
        expected_band=cfg.get("expected_band"),
    )
    log.info("done")


def _run_forward(bb, prompt: str) -> bool:
    """Run a single forward pass appropriate for the backbone's kind."""
    if bb.kind == "sentence-encoder":
        emb = bb.model.encode(prompt, convert_to_tensor=True)
        return emb.numel() > 0
    if bb.kind == "decoder":
        import torch

        inputs = bb.tokenizer(prompt, return_tensors="pt").to(bb.model.device)
        with torch.no_grad():
            out = bb.model(**inputs)
        return out.logits.numel() > 0
    # encoder
    import torch

    inputs = bb.tokenizer(prompt, return_tensors="pt").to(bb.model.device)
    with torch.no_grad():
        out = bb.model(**inputs)
    return out.last_hidden_state.numel() > 0


if __name__ == "__main__":
    main()
