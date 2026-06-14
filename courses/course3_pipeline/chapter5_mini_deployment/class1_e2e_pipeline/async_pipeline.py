"""
Async-native version of the four pipeline loops using asyncio.

Production pattern: each subsystem is an asyncio coroutine.
asyncio.gather() runs them concurrently on the same event loop — no
thread overhead, no GIL for I/O-bound work.

CPU-bound work (model inference): use asyncio.run_in_executor()
with ThreadPoolExecutor to avoid blocking the event loop.

Run: python async_pipeline.py
"""
from __future__ import annotations
import asyncio
import logging
import random
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("async_pipeline")


async def serving_loop(stats_queue: asyncio.Queue, n_requests: int = 10) -> None:
    """Simulate async serving — each request is a coroutine."""
    for i in range(n_requests):
        await asyncio.sleep(random.uniform(0.01, 0.04))
        latency_ms = random.gauss(120, 15)
        await stats_queue.put({"type": "latency", "value": latency_ms, "idx": i})
        log.info("served request %02d  latency=%.1f ms", i, latency_ms)


async def monitoring_loop(stats_queue: asyncio.Queue, poll_interval: float = 0.05) -> None:
    """Consume stats from queue, compute drift, emit alerts."""
    p95_window: list[float] = []
    while True:
        try:
            item = await asyncio.wait_for(stats_queue.get(), timeout=poll_interval)
            if item["type"] == "latency":
                p95_window.append(item["value"])
                if len(p95_window) > 50:
                    p95_window = p95_window[-50:]
                p95 = sorted(p95_window)[int(0.95 * len(p95_window))]
                if p95 > 200:
                    log.warning("SLO BREACH: p95 latency = %.1f ms > 200 ms threshold", p95)
            stats_queue.task_done()
        except asyncio.TimeoutError:
            log.info("monitoring_loop: queue drained, exiting")
            break


async def active_learning_loop(n_queries: int = 5) -> list[float]:
    """Select uncertain samples for human labeling."""
    scores = []
    for i in range(n_queries):
        await asyncio.sleep(random.uniform(0.02, 0.06))
        uncertainty = random.uniform(0.5, 1.0)
        scores.append(uncertainty)
        log.info("AL query %d: uncertainty=%.3f  %s", i, uncertainty,
                 "[SELECTED]" if uncertainty > 0.75 else "")
    return scores


async def auto_update_loop(versions: list[str] | None = None) -> None:
    """Poll model registry and promote new versions."""
    if versions is None:
        versions = ["v1", "v2"]
    for version in versions:
        await asyncio.sleep(random.uniform(0.05, 0.15))
        log.info("auto-update: promoting model %s to production (blue-green flip)", version)


async def main() -> None:
    """
    Run all four subsystems concurrently via asyncio.gather().

    Wall-clock time is approximately the slowest single coroutine, NOT the sum of all.
    This is the key benefit over threading for I/O-bound workloads.
    """
    queue: asyncio.Queue = asyncio.Queue()
    t0 = time.perf_counter()

    await asyncio.gather(
        serving_loop(queue, n_requests=10),
        monitoring_loop(queue),
        active_learning_loop(n_queries=5),
        auto_update_loop(),
    )

    elapsed = time.perf_counter() - t0
    log.info("All four subsystems completed in %.2f s (concurrent execution)", elapsed)


if __name__ == "__main__":
    asyncio.run(main())
