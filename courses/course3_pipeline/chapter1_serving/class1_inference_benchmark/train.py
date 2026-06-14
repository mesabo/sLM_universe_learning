"""Course 3 / ch1 / class 1 — inference fundamentals benchmark.

No training. Three benchmarks against SmolLM2-Instruct, in order:
  1. Single-request latency (one prompt at a time, n_passes timed).
  2. Batched latency (batch_size copies of one prompt, batched_n_passes timed).
  3. Prefill / decode split (forward-only on prompt vs full generate).

Persists a result JSON with mean / p50 / p95 latencies and tokens/sec.
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
import os
import statistics
import time

import torch

from shared.backbones import load_backbone
from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.logging_utils import get_logger
from shared.paths import hf_cache
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = q * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_prompt(bb, prompt: str, gen_cfg: dict) -> str:
    """Optionally render through the chat template so instruct models actually answer."""
    if gen_cfg.get("use_chat_template"):
        return bb.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    return prompt


def _gen_kwargs(gen_cfg: dict, eos_id: int) -> dict:
    kw = dict(
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg.get("do_sample", False),
        temperature=gen_cfg.get("temperature", 1.0),
        pad_token_id=eos_id,
    )
    min_nt = gen_cfg.get("min_new_tokens")
    if min_nt:
        kw["min_new_tokens"] = int(min_nt)
    return kw


def _generate_once(bb, prompt: str, gen_cfg: dict) -> tuple[float, int]:
    """One generate call. Returns (elapsed_ms, n_new_tokens)."""
    rendered = _format_prompt(bb, prompt, gen_cfg)
    inputs = bb.tokenizer(rendered, return_tensors="pt").to(bb.model.device)
    _gpu_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = bb.model.generate(**inputs, **_gen_kwargs(gen_cfg, bb.tokenizer.eos_token_id))
    _gpu_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    n_new = int(out_ids.shape[-1] - inputs["input_ids"].shape[-1])
    return elapsed_ms, n_new


def _generate_batch_once(bb, prompts: list[str], gen_cfg: dict) -> tuple[float, int]:
    """One batched generate call. Returns (elapsed_ms, total_new_tokens)."""
    if bb.tokenizer.padding_side != "left":
        bb.tokenizer.padding_side = "left"  # required for left-padded batched generation
    rendered = [_format_prompt(bb, p, gen_cfg) for p in prompts]
    inputs = bb.tokenizer(rendered, return_tensors="pt", padding=True).to(bb.model.device)
    _gpu_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = bb.model.generate(**inputs, **_gen_kwargs(gen_cfg, bb.tokenizer.eos_token_id))
    _gpu_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # total_new_tokens = sum across batch of (output_len - input_len_per_row).
    # With left-padding, input_len is the same for all rows; new tokens per row
    # = out_ids.shape[-1] - inputs["input_ids"].shape[-1].
    n_new_per = int(out_ids.shape[-1] - inputs["input_ids"].shape[-1])
    total_new = n_new_per * out_ids.shape[0]
    return elapsed_ms, total_new


def _time_prefill(bb, prompt: str, gen_cfg: dict) -> float:
    """Forward pass on the rendered prompt only (no generation). Returns elapsed_ms."""
    rendered = _format_prompt(bb, prompt, gen_cfg)
    inputs = bb.tokenizer(rendered, return_tensors="pt").to(bb.model.device)
    _gpu_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        bb.model(**inputs)
    _gpu_sync()
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    args = parse_args()
    log = get_logger("course3.ch1.class1")
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    set_seed(cfg["seed"])
    os.environ.setdefault("HF_HOME", str(hf_cache()))

    # Auto-derive method tag from batch size.
    batch = int(cfg["batches"]["batch_size"])
    if cfg.get("method", "").startswith("bench-b"):
        cfg["method"] = f"bench-b{batch}"

    log.info("loading backbone: %s", cfg["backbone"])
    bb = load_backbone(cfg["backbone"])
    if bb.kind != "decoder":
        raise ValueError(f"this benchmark requires a decoder backbone, got {bb.kind}")

    prompts = cfg["prompts"]
    if not prompts:
        raise ValueError("config must declare at least one prompt")
    prompt = prompts[0]
    gen_cfg = cfg["generation"]
    warmup = int(cfg["batches"]["warmup"])

    # --- Single-request benchmark ------------------------------------------
    n_single = int(cfg["batches"]["single_n_passes"])
    log.info("[single] warmup=%d n_passes=%d", warmup, n_single)
    for _ in range(warmup):
        _generate_once(bb, prompt, gen_cfg)
    single_lats: list[float] = []
    single_new_tokens = 0
    for i in range(n_single):
        ms, n_new = _generate_once(bb, prompt, gen_cfg)
        single_lats.append(ms)
        single_new_tokens += n_new
        if i == 0 or (i + 1) % max(1, n_single // 4) == 0:
            log.info("[single] pass %d/%d latency_ms=%.2f n_new=%d", i+1, n_single, ms, n_new)
    single_mean = statistics.fmean(single_lats)
    single_p50 = statistics.median(single_lats)
    single_p95 = _percentile(single_lats, 0.95)

    # --- Batched benchmark -------------------------------------------------
    n_batched = int(cfg["batches"]["batched_n_passes"])
    log.info("[batched] batch_size=%d n_passes=%d", batch, n_batched)
    for _ in range(warmup):
        _generate_batch_once(bb, [prompt] * batch, gen_cfg)
    batched_lats: list[float] = []
    batched_new_tokens = 0
    for i in range(n_batched):
        ms, total_new = _generate_batch_once(bb, [prompt] * batch, gen_cfg)
        batched_lats.append(ms)
        batched_new_tokens += total_new
        log.info("[batched] pass %d/%d latency_ms=%.2f total_new=%d", i+1, n_batched, ms, total_new)
    batched_mean = statistics.fmean(batched_lats)
    batched_total_time_s = sum(batched_lats) / 1000.0
    tokens_per_sec = (batched_new_tokens / batched_total_time_s) if batched_total_time_s > 0 else 0.0

    # --- Prefill / decode split -------------------------------------------
    log.info("[prefill/decode] measuring split on a single request")
    prefill_lats: list[float] = []
    for _ in range(warmup):
        _time_prefill(bb, prompt, gen_cfg)
    for _ in range(n_single):
        prefill_lats.append(_time_prefill(bb, prompt, gen_cfg))
    prefill_mean = statistics.fmean(prefill_lats)

    # full generate cost (single) - prefill_mean = decode cost for n_new tokens
    n_new_per_pass = single_new_tokens // max(n_single, 1)
    decode_total_ms = single_mean - prefill_mean
    mean_decode_per_token_ms = (decode_total_ms / n_new_per_pass) if n_new_per_pass > 0 else 0.0
    prefill_fraction = (prefill_mean / single_mean) if single_mean > 0 else 0.0
    log.info("[summary] single=%.2fms batched=%.2fms tok/s=%.1f prefill=%.2fms "
             "decode/tok=%.3fms prefill_frac=%.3f",
             single_mean, batched_mean, tokens_per_sec,
             prefill_mean, mean_decode_per_token_ms, prefill_fraction)

    metrics = {
        "single_request_latency_ms": float(single_mean),
        "batched_latency_ms": float(batched_mean),
        "tokens_per_second": float(tokens_per_sec),
        "prefill_time_ms": float(prefill_mean),
        "mean_decode_per_token_ms": float(mean_decode_per_token_ms),
        "prefill_fraction": float(prefill_fraction),
    }

    # Industry terminology aliases
    metrics["ttft_ms"] = metrics.get("prefill_time_ms", metrics.get("single_latency_ms", 0))
    metrics["tpot_ms"] = metrics.get("mean_decode_per_token_ms", 0)

    # p95 latency across single-request passes
    if single_lats:
        sorted_lats = sorted(single_lats)
        p95_idx = int(0.95 * len(sorted_lats))
        metrics["p95_latency_ms"] = sorted_lats[p95_idx]

    run_eval(
        method=cfg["method"],
        backbone=cfg["backbone"],
        course=cfg["course"], klass=cfg["class_id"], task=cfg["task"],
        config=cfg, metrics=metrics,
        expected_band=cfg["expected_band"][cfg["mode"]],
        extras={
            "single_p50_ms": float(single_p50),
            "single_p95_ms": float(single_p95),
            "single_latencies_ms": [round(x, 3) for x in single_lats],
            "batched_latencies_ms": [round(x, 3) for x in batched_lats],
            "prefill_latencies_ms": [round(x, 3) for x in prefill_lats],
            "n_new_tokens_per_pass": n_new_per_pass,
            "batch_size": batch,
            "single_throughput_tokens_per_s": float(
                (single_new_tokens / (sum(single_lats)/1000.0)) if single_lats else 0.0
            ),
            "batched_speedup_vs_single_per_token": float(
                tokens_per_sec / max(1e-9,
                    (single_new_tokens / (sum(single_lats)/1000.0)) if single_lats else 0.0)
            ),
            "mode": cfg["mode"],
        },
    )


if __name__ == "__main__":
    main()
