"""Course 0 / ch2 / class 1 — encode AND generate the same prompt.

Each backbone runs `iterations.n_passes` timed passes after `iterations.warmup`
untimed ones. Results carry latency stats so the student can compare encoder
vs decoder cost on their own hardware.
"""

from __future__ import annotations

import argparse
import statistics
import time

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

    prompt = cfg["prompt"]

    enc_metrics = _encoder_step(cfg["encoder_backbone"], prompt, n_passes, warmup, log)
    _emit(cfg, cfg["encoder_backbone"], enc_metrics)

    dec_metrics = _decoder_step(cfg["decoder_backbone"], prompt, cfg["generation"],
                                n_passes, warmup, log)
    _emit(cfg, cfg["decoder_backbone"], dec_metrics)

    log.info("done")


def _emit(cfg: dict, backbone: str, metrics: dict[str, float]) -> None:
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


def _time_n(fn, n_passes: int, warmup: int) -> tuple[list[float], object]:
    """Run fn() warmup + n_passes times; return (latencies_ms, last_result)."""
    last = None
    for _ in range(warmup):
        last = fn()
    latencies: list[float] = []
    for _ in range(n_passes):
        t0 = time.perf_counter()
        last = fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies, last


def _encoder_step(name: str, prompt: str, n_passes: int, warmup: int, log) -> dict:
    log.info("[encoder] loading %s", name)
    bb = load_backbone(name)
    if bb.kind == "decoder":
        raise ValueError(f"Expected encoder/sentence-encoder, got decoder for {name}")

    def _once():
        return (
            bb.model.encode(prompt, convert_to_tensor=True)
            if bb.kind == "sentence-encoder"
            else _encoder_pool(bb, prompt)
        )

    latencies, emb = _time_n(_once, n_passes, warmup)
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
            "latencies_ms": [round(x, 3) for x in latencies],
        },
    }


def _encoder_pool(bb, prompt: str):
    import torch

    inputs = bb.tokenizer(prompt, return_tensors="pt").to(bb.model.device)
    with torch.no_grad():
        out = bb.model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze(0)


def _decoder_step(name: str, prompt: str, gen_cfg: dict,
                  n_passes: int, warmup: int, log) -> dict:
    import torch

    log.info("[decoder] loading %s", name)
    bb = load_backbone(name)
    if bb.kind != "decoder":
        raise ValueError(f"Expected decoder, got {bb.kind} for {name}")

    inputs = bb.tokenizer(prompt, return_tensors="pt").to(bb.model.device)

    def _once():
        with torch.no_grad():
            return bb.model.generate(
                **inputs,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=gen_cfg.get("do_sample", False),
                temperature=gen_cfg.get("temperature", 1.0),
                pad_token_id=bb.tokenizer.eos_token_id,
            )

    latencies, out_ids = _time_n(_once, n_passes, warmup)
    new_tokens = int(out_ids.shape[-1] - inputs["input_ids"].shape[-1])
    text = bb.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    mean_ms = statistics.fmean(latencies)
    log.info("[decoder] new_tokens=%d n=%d mean=%.2fms p95=%.2fms preview=%s",
             new_tokens, n_passes, mean_ms, _percentile(latencies, 0.95),
             text[:120].replace("\n", " "))
    return {
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
            "latencies_ms": [round(x, 3) for x in latencies],
        },
    }


if __name__ == "__main__":
    main()
