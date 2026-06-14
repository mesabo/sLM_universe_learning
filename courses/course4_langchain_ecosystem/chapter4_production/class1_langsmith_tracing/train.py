"""Course 4 / ch4 / class 1 — LangSmith tracing and evaluation.

Demonstrates:
  - LangSmith tracing via LANGCHAIN_TRACING_V2 env var
  - Offline fallback: writing trace records to a local JSONL file when API key not set
  - Unique run IDs: each chain invocation gets a distinct trace ID
  - Structured trace payload: input, output, run_id, latency
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.llm_client import get_llm
from shared.logging_utils import get_logger
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


PROMPTS = [
    "What is LangSmith?",
    "Explain the difference between a chain and an agent in LangChain.",
    "How does FAISS work?",
    "What is context recall in RAG evaluation?",
    "Describe LangGraph in one sentence.",
    "What are the benefits of structured output?",
    "How does ReAct reasoning work?",
    "What is semantic caching?",
    "Explain LoRA briefly.",
    "What is a vector store?",
    "How does streaming work in LCEL?",
    "What is MultiQueryRetriever?",
    "Describe the purpose of a document compressor.",
    "What is the difference between FAISS and ChromaDB?",
    "How does LangSmith help with debugging?",
]


def run_with_offline_trace(llm, prompts: list[str], trace_path: Path, log) -> list[str]:
    """Invoke llm on each prompt, write trace record to JSONL, return run_ids."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    chain = ChatPromptTemplate.from_template("{question}") | llm | StrOutputParser()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    run_ids = []

    with trace_path.open("a") as fh:
        for prompt in prompts:
            run_id = str(uuid.uuid4())
            t0 = time.monotonic()
            try:
                output = chain.invoke({"question": prompt})
            except Exception as exc:
                output = f"ERROR: {exc}"
            latency_ms = (time.monotonic() - t0) * 1000
            record = {
                "run_id": run_id,
                "input": prompt,
                "output": output[:200],
                "latency_ms": round(latency_ms, 1),
            }
            fh.write(json.dumps(record) + "\n")
            run_ids.append(run_id)
            log.info("Traced", run_id=run_id[:8], latency_ms=round(latency_ms))

    return run_ids


def run_with_langsmith(llm, prompts: list[str], log) -> list[str]:
    """Try real LangSmith tracing; return list of run_ids (may be empty)."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    api_key = os.environ.get("LANGSMITH_API_KEY", "")
    if not api_key:
        log.info("LANGSMITH_API_KEY not set; skipping live tracing")
        return []

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = api_key

    from langsmith import Client
    from langchain_core.tracers import LangChainTracer

    client = Client()
    tracer = LangChainTracer()
    chain = ChatPromptTemplate.from_template("{question}") | llm | StrOutputParser()
    run_ids = []

    for prompt in prompts:
        try:
            run_id = str(uuid.uuid4())
            chain.invoke({"question": prompt}, config={"callbacks": [tracer], "run_name": run_id})
            run_ids.append(run_id)
        except Exception as exc:
            log.warning("LangSmith trace failed", error=str(exc))

    return run_ids


def run_traceable_demo(log) -> dict:
    """
    @traceable — instrument arbitrary Python functions for LangSmith.

    Use for functions that are NOT LangChain Runnables (preprocessing,
    post-processing, custom logic). LangSmith captures inputs, outputs,
    latency, and exceptions automatically.

    With LANGCHAIN_API_KEY set: traces appear in LangSmith UI automatically.
    Without it: @traceable is a no-op — safe for all environments.

    LangChain Hub (prompt versioning):
      hub.push("my-org/my-prompt", prompt_object)   # save v1
      prompt = hub.pull("my-org/my-prompt:v2")       # load specific version
    This is how teams A/B test prompts without code deploys.
    """
    try:
        from langsmith import traceable

        @traceable(name="text_normalizer", tags=["preprocessing", "course4"])
        def normalize_query(query: str) -> str:
            """Normalize a user query before embedding."""
            return " ".join(query.strip().lower().split())

        result = normalize_query("  What IS  LoRA fine-tuning?  ")
        assert result == "what is lora fine-tuning?"
        log.info("@traceable demo ok", result=result)
        return {"traceable_ok": 1}
    except ImportError:
        log.info("langsmith not installed — @traceable skipped (expected in offline env)")
        return {"traceable_ok": -1}
    except Exception as exc:
        log.warning("traceable demo failed", error=str(exc)[:80])
        return {"traceable_ok": 0}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch4.class1")
    mode = cfg.get("mode", "smoke")
    n_traces = cfg["limits"][mode]["n_traces"]

    llm = get_llm(cfg)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"))

    prompts = PROMPTS[:n_traces]
    trace_path = Path(cfg.get("offline_trace_path", "results/course4_langchain_ecosystem/traces.jsonl"))
    if trace_path.exists():
        trace_path.unlink()

    # Always write offline traces (primary path, works without API key)
    run_ids = run_with_offline_trace(llm, prompts, trace_path, log)

    # Optionally attempt live LangSmith tracing on top
    live_ids = run_with_langsmith(llm, prompts[:2], log)
    if live_ids:
        log.info("LangSmith live traces", n=len(live_ids))

    # @traceable decorator demo
    traceable_metrics = run_traceable_demo(log)

    # Verify offline JSONL
    traces_written_ok = 0
    run_ids_unique_ok = 0
    if trace_path.exists():
        records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        if len(records) == n_traces:
            traces_written_ok = 1
        ids_in_file = [r["run_id"] for r in records]
        if len(set(ids_in_file)) == len(ids_in_file) == n_traces:
            run_ids_unique_ok = 1
        log.info("Trace file verified", n=len(records), unique_ids=len(set(ids_in_file)))

    metrics = {
        "traces_written_ok": float(traces_written_ok),
        "run_ids_unique_ok": float(run_ids_unique_ok),
    }
    metrics.update(traceable_metrics)
    run_eval(
        method=cfg["method"],
        backbone=cfg.get("backbone", "local"),
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics=metrics,
        expected_band=cfg.get("expected_band", {}),
        extras={"mode": mode, "n_traces": n_traces, "trace_path": str(trace_path)},
    )


if __name__ == "__main__":
    main()
