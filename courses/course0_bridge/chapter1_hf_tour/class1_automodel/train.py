"""Course 0 / ch1 / class 1 — HF ecosystem sanity check.

This class has no real training step; "training" is just a proof that the
HF stack is wired correctly. We load a backbone via `shared.backbones`,
run `iterations.n_passes` forward passes (after `iterations.warmup`
untimed ones), and persist a result JSON with latency stats.

Run via `run.sh`; do not invoke directly without the right working dir.
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
    parser.add_argument("--config", required=True, help="Path to class config YAML")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted overrides like backbone=BAAI/bge-small-en-v1.5 iterations.n_passes=10",
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
    iterations = cfg.get("iterations", {})
    n_passes = int(iterations.get("n_passes", 1))
    warmup = int(iterations.get("warmup", 0))
    if n_passes < 1:
        raise ValueError(f"iterations.n_passes must be >= 1, got {n_passes}")

    # Warmup — discard timing on these (cuDNN init, weight upload, etc.).
    for _ in range(warmup):
        _run_forward(bb, prompt)

    # Timed iterations.
    latencies_ms: list[float] = []
    forward_ok = False
    for i in range(n_passes):
        t0 = time.perf_counter()
        forward_ok = _run_forward(bb, prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(latency_ms)
        if i == 0 or (i + 1) % max(1, n_passes // 5) == 0:
            log.info("[pass %d/%d] forward_ok=%s latency_ms=%.2f",
                     i + 1, n_passes, bool(forward_ok), latency_ms)

    mean_ms = statistics.fmean(latencies_ms)
    p50_ms = statistics.median(latencies_ms)
    p95_ms = _percentile(latencies_ms, 0.95)
    log.info("summary: n=%d mean=%.2fms p50=%.2fms p95=%.2fms throughput=%.2f passes/s",
             n_passes, mean_ms, p50_ms, p95_ms, 1000.0 / mean_ms if mean_ms > 0 else 0.0)

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
            "n_passes_ran": n_passes,
            "mean_latency_ms": float(mean_ms),
        },
        expected_band=cfg.get("expected_band"),
        extras={
            "warmup": warmup,
            "p50_latency_ms": float(p50_ms),
            "p95_latency_ms": float(p95_ms),
            "throughput_passes_per_s": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
            "latencies_ms": [round(x, 3) for x in latencies_ms],
        },
    )
    log.info("done")


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1]). No external deps."""
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
