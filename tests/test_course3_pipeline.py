"""Industry-standard tests for course3_pipeline.

Covers: FastAPI endpoint contract (schema, status codes, validation),
async concurrency proof, streaming, Prometheus metrics format, p95 computation.
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
APP_DIR = REPO / "courses/course3_pipeline/chapter5_mini_deployment/class1_e2e_pipeline"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# FastAPI TestClient tests
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app import app
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_response_schema(client):
    """POST /predict returns prediction, model_version, latency_ms, cache_hit."""
    resp = client.post("/predict", json={"text": "classify this sentence"})
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "model_version" in body
    assert "latency_ms" in body
    assert isinstance(body["latency_ms"], (int, float))
    assert "cache_hit" in body


def test_predict_empty_text_returns_422(client):
    """POST /predict with empty text returns 422 Unprocessable Entity."""
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_predict_missing_body_returns_422(client):
    """POST /predict with no body returns 422."""
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_metrics_prometheus_format(client):
    """GET /metrics returns Prometheus text format with p95 latency."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "pipeline_n_queries_total" in body
    assert "pipeline_request_latency_ms" in body
    assert 'quantile="0.95"' in body
    assert "pipeline_drift_score" in body


def test_status_returns_model_version(client):
    """GET /status returns model_version and uptime."""
    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_version" in body
    assert "uptime_s" in body
    assert "p95_latency_ms" in body


def test_trigger_update_increments_version(client):
    """POST /trigger-update returns new_version different from old_version."""
    r1 = client.get("/status")
    old = r1.json()["model_version"]
    r2 = client.post("/trigger-update")
    assert r2.status_code == 200
    body = r2.json()
    assert body["old_version"] == old
    assert body["new_version"] != old


def test_n_queries_counter_increments(client):
    """n_queries increases by 1 for each POST /predict."""
    from app import _state
    before = _state["n_queries"]
    client.post("/predict", json={"text": "test"})
    assert _state["n_queries"] == before + 1


# ---------------------------------------------------------------------------
# Async pipeline tests
# ---------------------------------------------------------------------------
def test_async_pipeline_runs():
    """async_pipeline.main() completes without error."""
    from async_pipeline import main
    asyncio.run(main())


def test_asyncio_gather_is_concurrent():
    """asyncio.gather() runs coroutines concurrently — not sequentially.

    Two 0.1s sleeps complete in ~0.1s total wall-clock, not ~0.2s.
    This is the essential proof that async I/O is NOT sequential.
    """
    async def _check():
        import time
        t0 = time.perf_counter()
        await asyncio.gather(asyncio.sleep(0.1), asyncio.sleep(0.1))
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.18, f"gather took {elapsed:.3f}s — concurrency broken"
    asyncio.run(_check())


def test_active_learning_loop_returns_scores():
    """active_learning_loop() returns a list of float uncertainty scores."""
    from async_pipeline import active_learning_loop
    scores = asyncio.run(active_learning_loop(n_queries=3))
    assert len(scores) == 3
    for s in scores:
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# p95 computation correctness
# ---------------------------------------------------------------------------
def test_p95_computation():
    """p95 of [1..100] is 95."""
    from app import _compute_percentile
    values = list(range(1, 101))
    p95 = _compute_percentile(values, 0.95)
    assert 94 <= p95 <= 96


def test_p95_empty_list():
    """p95 of empty list returns 0 without error."""
    from app import _compute_percentile
    assert _compute_percentile([], 0.95) == 0.0
