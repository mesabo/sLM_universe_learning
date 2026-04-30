"""Course 0 / ch3 / class 1 — tokenization & chat templates.

No model weights are loaded; we only need tokenizers, so this class is
fast and CPU-only.
"""

from __future__ import annotations


# --- ensure repo root is importable when invoked via `python <path>/train.py` ---
import sys as _sys, pathlib as _pathlib
_root = _pathlib.Path(__file__).resolve()
for _p in [_root.parent, *_root.parents]:
    if (_p / "pyproject.toml").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
del _sys, _pathlib, _root, _p
# --- end shim ---

import argparse

from transformers import AutoTokenizer

from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = get_logger("course0.ch3.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg.get("seed", 0))

    enc_metrics = _encoder_step(cfg["encoder_backbone"], cfg["prompt"], log)
    _emit(cfg, cfg["encoder_backbone"], enc_metrics)

    dec_metrics = _decoder_step(cfg["decoder_backbone"], cfg["prompt"], cfg["messages"], log)
    _emit(cfg, cfg["decoder_backbone"], dec_metrics)

    log.info("done")


def _emit(cfg: dict, backbone: str, metrics: dict[str, int]) -> None:
    run_eval(
        method=cfg["method"],
        backbone=backbone,
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics=metrics,
        expected_band=cfg.get("expected_band"),
    )


def _encoder_step(name: str, prompt: str, log) -> dict[str, int]:
    tok = AutoTokenizer.from_pretrained(name)
    enc = tok(prompt, return_tensors=None, return_special_tokens_mask=True)
    log.info("[encoder %s] tokens=%d input_ids=%s",
             name, len(enc["input_ids"]), enc["input_ids"][:16])
    return {
        "encoder_tokens": len(enc["input_ids"]),
        "decoder_tokens": 1,
        "chat_no_genprompt_tokens": 1,
        "chat_with_genprompt_tokens": 1,
        "genprompt_adds_tokens": 1,
    }


def _decoder_step(name: str, prompt: str, messages: list[dict], log) -> dict[str, int]:
    tok = AutoTokenizer.from_pretrained(name)

    raw = tok(prompt, return_tensors=None)
    raw_n = len(raw["input_ids"])
    log.info("[decoder %s] raw tokens=%d", name, raw_n)

    no_gen = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    with_gen = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    log.info("[decoder %s] chat-no-gen=%d chat-with-gen=%d", name, len(no_gen), len(with_gen))

    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    log.info("[decoder %s] chat-rendered preview:\n%s", name, rendered[:280])

    return {
        "encoder_tokens": 1,
        "decoder_tokens": raw_n,
        "chat_no_genprompt_tokens": len(no_gen),
        "chat_with_genprompt_tokens": len(with_gen),
        "genprompt_adds_tokens": int(len(with_gen) > len(no_gen)),
    }


if __name__ == "__main__":
    main()
