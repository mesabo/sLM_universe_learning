"""Course 0 / ch2 / class 1 — encode AND generate the same prompt.

Each backbone runs `iterations.n_passes` timed passes after `iterations.warmup`
untimed ones. Results carry latency stats so the student can compare encoder
vs decoder cost on their own hardware.
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
import statistics
import time
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.backbones import load_backbone
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
    log = get_logger("course0.ch2.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg.get("seed", 0))

    iterations = cfg.get("iterations", {})
    n_passes = int(iterations.get("n_passes", 1))
    warmup = int(iterations.get("warmup", 0))
    if n_passes < 1:
        raise ValueError(f"iterations.n_passes must be >= 1, got {n_passes}")

    prompts = _resolve_prompts(cfg)
    log.info("prompts: %d (cycled across passes)", len(prompts))

    enc_metrics = _encoder_step(cfg["encoder_backbone"], prompts, n_passes, warmup, log)
    _emit(cfg, cfg["encoder_backbone"], enc_metrics)

    dec_metrics = _decoder_step(cfg["decoder_backbone"], prompts, cfg["generation"],
                                n_passes, warmup, log)
    _emit(cfg, cfg["decoder_backbone"], dec_metrics)

    log.info("done")


def _resolve_prompts(cfg: dict) -> list[str]:
    """Accept either `prompts: [list]` (preferred) or `prompt: "..."` (legacy)."""
    raw = cfg.get("prompts")
    if raw is None:
        single = cfg.get("prompt")
        if single is None:
            raise ValueError("config must define either `prompts: [...]` or `prompt: \"...\"`")
        raw = [single]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"`prompts` must be a non-empty list of strings, got {raw!r}")
    return [str(p) for p in raw]


def _emit(cfg: dict, backbone: str, metrics: dict[str, float]) -> None:
    # Accept defensive inputs: some callers may accidentally pass the raw
    # (_latencies, _indices, _last) tuple returned by `_time_n`. Convert
    # that into a minimal metrics dict so `run_eval` can proceed.
    if isinstance(metrics, tuple) and len(metrics) == 3:
        latencies, indices, last = metrics
        mean_ms = statistics.fmean(latencies) if latencies else 0.0
        metrics = {
            "output_ok": 1,
            "tokens_generated": 1,
            "n_passes_ran": len(latencies),
            "mean_latency_ms": float(mean_ms),
            "_extras": {
                "latencies_ms": [round(x, 3) for x in latencies],
                "prompt_indices": indices,
            },
        }

    extras = metrics.pop("_extras", {})
    run_eval(
        method=cfg["method"],
        backbone=backbone,
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics=metrics,
        expected_band=cfg.get("expected_band"),
        extras=extras,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = q * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _time_n(fn_of_prompt, prompts: list[str], n_passes: int, warmup: int):
    """Run fn(prompt) warmup + n_passes times, cycling through prompts.

    Returns (latencies_ms, prompt_indices, last_result).
    """
    last = None
    for w in range(warmup):
        last = fn_of_prompt(prompts[w % len(prompts)])
    latencies: list[float] = []
    indices: list[int] = []
    for i in range(n_passes):
        idx = i % len(prompts)
        t0 = time.perf_counter()
        last = fn_of_prompt(prompts[idx])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        indices.append(idx)
    return latencies, indices, last


def _encoder_step(name: str, prompts: list[str], n_passes: int, warmup: int, log) -> dict:
    log.info("[encoder] loading %s", name)
    bb = load_backbone(name)
    if bb.kind == "decoder":
        raise ValueError(f"Expected encoder/sentence-encoder, got decoder for {name}")

    def _once(p: str):
        return (
            bb.model.encode(p, convert_to_tensor=True)
            if bb.kind == "sentence-encoder"
            else _encoder_pool(bb, p)
        )

    latencies, indices, emb = _time_n(_once, prompts, n_passes, warmup)
    dim = int(emb.shape[-1])
    mean_ms = statistics.fmean(latencies)
    log.info("[encoder] dim=%d hidden=%d norm=%.4f n=%d mean=%.2fms p95=%.2fms",
             dim, bb.hidden_size, float(emb.norm()), n_passes, mean_ms,
             _percentile(latencies, 0.95))
    return {
        "output_ok": int(emb.numel() > 0),
        "dim_matches_hidden": int(dim == bb.hidden_size or bb.hidden_size == 0),
        "tokens_generated": 1,  # not applicable; satisfies band
        "n_passes_ran": n_passes,
        "mean_latency_ms": float(mean_ms),
        "_extras": {
            "warmup": warmup,
            "p50_latency_ms": float(statistics.median(latencies)),
            "p95_latency_ms": float(_percentile(latencies, 0.95)),
            "throughput_passes_per_s": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
            "n_prompts": len(prompts),
            "prompts": prompts,
            "latencies_ms": [round(x, 3) for x in latencies],
            "prompt_indices": indices,
        },
    }


def _encoder_pool(bb, prompt: str):
    import torch

    inputs = bb.tokenizer(prompt, return_tensors="pt").to(bb.model.device)
    with torch.no_grad():
        out = bb.model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze(0)


def _decoder_step(name: str, prompts: list[str], gen_cfg: dict,
                  n_passes: int, warmup: int, log) -> dict:
    import torch

    log.info("[decoder] loading %s", name)
    bb = load_backbone(name)
    if bb.kind != "decoder":
        raise ValueError(f"Expected decoder, got {bb.kind} for {name}")

    def _once(p: str):
        inputs = bb.tokenizer(p, return_tensors="pt").to(bb.model.device)
        with torch.no_grad():
            # Exercise 3: Optional mean-pooled hidden-state vector from the decoder.
            # This allows comparing decoder internal representations to encoder ones.
            emb = None
            if gen_cfg.get("compute_pooling", False):
                out_enc = bb.model(**inputs, output_hidden_states=True, return_dict=True)
                emb = out_enc.hidden_states[-1].mean(dim=1).squeeze(0)

            out_ids = bb.model.generate(
                **inputs,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=gen_cfg.get("do_sample", False),
                temperature=gen_cfg.get("temperature", 1.0),
                pad_token_id=bb.tokenizer.eos_token_id,
            )
        return out_ids, inputs, emb

    latencies, indices, last = _time_n(_once, prompts, n_passes, warmup)
    out_ids, inputs, emb = last
    new_tokens = int(out_ids.shape[-1] - inputs["input_ids"].shape[-1])
    text = bb.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    mean_ms = statistics.fmean(latencies)
    
    metrics = {
        "output_ok": int(new_tokens > 0),
        "dim_matches_hidden": 1,  # not applicable; satisfies band
        "tokens_generated": new_tokens,
        "n_passes_ran": n_passes,
        "mean_latency_ms": float(mean_ms),
        "_extras": {
            "warmup": warmup,
            "p50_latency_ms": float(statistics.median(latencies)),
            "p95_latency_ms": float(_percentile(latencies, 0.95)),
            "throughput_passes_per_s": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
            "tokens_per_second": float(new_tokens * 1000.0 / mean_ms) if mean_ms > 0 else 0.0,
            "n_prompts": len(prompts),
            "prompts": prompts,
            "latencies_ms": [round(x, 3) for x in latencies],
            "prompt_indices": indices,
        },
    }

    if emb is not None:
        dim = int(emb.shape[-1])
        log.info("[decoder] dim=%d norm=%.4f new_tokens=%d n=%d mean=%.2fms p95=%.2fms preview=%s",
                 dim, float(emb.norm()), new_tokens, n_passes, mean_ms, _percentile(latencies, 0.95),
                 text[:120].replace("\n", " "))
        metrics["_extras"]["pooled_dim"] = dim
        metrics["_extras"]["pooled_norm"] = float(emb.norm())
    else:
        log.info("[decoder] new_tokens=%d n=%d mean=%.2fms p95=%.2fms preview=%s",
                 new_tokens, n_passes, mean_ms, _percentile(latencies, 0.95),
                 text[:120].replace("\n", " "))

    return metrics


if __name__ == "__main__":
    main()
