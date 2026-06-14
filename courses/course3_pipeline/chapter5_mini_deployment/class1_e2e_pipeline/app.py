"""FastAPI service wrapping the e2e ML pipeline.

Production pattern: each subsystem (serving, monitoring, AL, auto-update)
runs as a BackgroundTask or asyncio coroutine — never blocking the HTTP event loop.

Run locally:
  uvicorn app:app --reload --host 0.0.0.0 --port 8080

Key patterns demonstrated:
  - lifespan context manager for startup/shutdown
  - BackgroundTasks for fire-and-forget monitoring
  - StreamingResponse for SSE token streaming
  - /metrics in Prometheus text format (p95/p99 vocabulary)
  - TTFT measurement in streaming endpoint
"""
from __future__ import annotations
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# In-memory state (production: Redis or PostgreSQL)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "model_version": "v0",
    "n_queries": 0,
    "n_cache_hits": 0,
    "drift_score": 0.0,
    "latencies_ms": [],  # rolling window for p95/p99
    "monitoring_active": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load/warm up on startup; release on shutdown."""
    _state["started_at"] = time.time()
    _state["monitoring_active"] = True
    yield
    _state["monitoring_active"] = False


app = FastAPI(
    title="E2E Pipeline Service",
    version="1.0.0",
    description="""
## End-to-end ML pipeline as a production FastAPI service

Demonstrates the core production patterns:

| Pattern | Implementation |
|---------|----------------|
| **Async endpoints** | `async def` + `await asyncio.sleep()` — never block event loop |
| **BackgroundTasks** | Monitoring tick fires after every prediction (non-blocking) |
| **StreamingResponse** | SSE token stream — measure TTFT (time to first token) |
| **Prometheus metrics** | `/metrics` in text exposition format — p50/p95/p99 latency |
| **lifespan** | Model warm-up on startup, graceful shutdown |
| **Blue-green update** | `/trigger-update` flips model version pointer atomically |
""",
    lifespan=lifespan,
    swagger_ui_parameters={"tryItOutEnabled": True, "displayRequestDuration": True},
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512,
                      description="Input text to run through the model.")

class PredictResponse(BaseModel):
    prediction: str
    model_version: str
    latency_ms: float
    cache_hit: bool


# ---------------------------------------------------------------------------
# Background subsystems
# ---------------------------------------------------------------------------
async def _monitoring_tick(text: str):
    """Async monitoring tick — runs after each prediction via BackgroundTasks.

    Production: compute PSI, log to CloudWatch/Datadog, alert if drift > threshold.
    Does NOT block the HTTP response — client gets the prediction immediately.
    """
    await asyncio.sleep(0)  # yield to event loop
    _state["drift_score"] = min(1.0, _state["drift_score"] + 0.001)


def _compute_percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, int(p * len(sorted_v)) - 1)
    return sorted_v[idx]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health():
    """Liveness probe."""
    return {"status": "ok", "service": "e2e-pipeline"}


@app.get("/metrics", tags=["System"])
async def metrics():
    """
    Prometheus-compatible metrics endpoint.

    Production: add prometheus_fastapi_instrumentator for automatic
    request_count, request_duration_seconds histogram (p50/p95/p99).

    p95 latency is the key SLO metric — never use mean.
    """
    lats = _state["latencies_ms"][-200:]  # rolling window
    p50 = _compute_percentile(lats, 0.50)
    p95 = _compute_percentile(lats, 0.95)
    p99 = _compute_percentile(lats, 0.99)
    lines = [
        "# HELP pipeline_n_queries_total Total prediction requests served",
        "# TYPE pipeline_n_queries_total counter",
        f"pipeline_n_queries_total {_state['n_queries']}",
        "",
        "# HELP pipeline_request_latency_ms Request latency percentiles",
        "# TYPE pipeline_request_latency_ms summary",
        f'pipeline_request_latency_ms{{quantile="0.5"}} {p50:.2f}',
        f'pipeline_request_latency_ms{{quantile="0.95"}} {p95:.2f}',
        f'pipeline_request_latency_ms{{quantile="0.99"}} {p99:.2f}',
        "",
        "# HELP pipeline_drift_score Current data drift PSI score",
        "# TYPE pipeline_drift_score gauge",
        f"pipeline_drift_score {_state['drift_score']:.4f}",
        "",
        "# HELP pipeline_model_info Current deployed model",
        "# TYPE pipeline_model_info gauge",
        f'pipeline_model_info{{version="{_state["model_version"]}"}} 1',
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    """
    Run prediction with fire-and-forget async monitoring.

    **BackgroundTasks pattern**: monitoring tick is added as a background task.
    The HTTP response is sent to the client BEFORE monitoring completes.
    This keeps p95 latency low even when monitoring is expensive.
    """
    t0 = time.perf_counter()
    _state["n_queries"] += 1

    # Non-blocking yield (replace with: result = await model.ainvoke(req.text))
    await asyncio.sleep(0)
    prediction = f"class_{abs(hash(req.text)) % 4}"

    latency_ms = (time.perf_counter() - t0) * 1000
    _state["latencies_ms"].append(latency_ms)
    if len(_state["latencies_ms"]) > 1000:
        _state["latencies_ms"] = _state["latencies_ms"][-500:]

    # Fire monitoring AFTER response — never block the client
    background_tasks.add_task(_monitoring_tick, req.text)

    return PredictResponse(
        prediction=prediction,
        model_version=_state["model_version"],
        latency_ms=round(latency_ms, 2),
        cache_hit=False,
    )


@app.get("/predict/stream", tags=["Inference"])
async def predict_stream(text: str = "Hello world"):
    """
    Stream predictions as Server-Sent Events (SSE).

    **TTFT (Time to First Token)**: the time from request to first `data:` event.
    This is what users perceive as "lag". Minimize by starting generation immediately.

    **TPOT (Time Per Output Token)**: interval between consecutive `data:` events.
    Determines streaming smoothness. Target: < 50ms per token.

    Production: replace the sleep loop with `async for chunk in llm.astream(text)`.
    """
    async def _generate():
        words = text.split() or ["hello"]
        for i, word in enumerate(words):
            await asyncio.sleep(0.02)  # simulate TPOT
            yield f"data: {word}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.get("/status", tags=["System"])
async def status():
    """Pipeline status: model version, query count, drift score, uptime."""
    lats = _state["latencies_ms"][-200:]
    return {
        "model_version": _state["model_version"],
        "n_queries": _state["n_queries"],
        "drift_score": round(_state["drift_score"], 4),
        "monitoring_active": _state["monitoring_active"],
        "uptime_s": round(time.time() - _state.get("started_at", time.time()), 1),
        "p95_latency_ms": round(_compute_percentile(lats, 0.95), 2),
    }


@app.post("/trigger-update", tags=["MLOps"])
async def trigger_update():
    """
    Atomically swap to next model version (blue-green flip).

    Production: this endpoint is called by your CI/CD pipeline after
    the new model passes shadow-mode A/B evaluation. Rollback = re-flip.
    In Kubernetes: update the Deployment image tag instead.
    """
    import re
    old = _state["model_version"]
    n = int(re.search(r"\d+", old).group()) + 1 if re.search(r"\d+", old) else 1
    _state["model_version"] = f"v{n}"
    _state["last_update"] = time.time()
    return {"old_version": old, "new_version": _state["model_version"], "status": "swapped"}
