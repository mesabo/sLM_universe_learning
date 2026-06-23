"""Correction for Course 4 / ch4 / class 2 exercises.

This file follows the same LCEL service-pattern theme as ``train.py`` and adds
the three concrete exercise implementations.

What is new relative to ``train.py``:
  1. TTFT measurement for streamed responses.
  2. A simple semantic-similarity cache using embeddings.
  3. An async FastAPI streaming endpoint example.

Sections marked ``NEW vs train.py`` highlight the corrections.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import apply_overrides, load_yaml
from shared.llm_client import get_embedding_model, get_llm
from shared.logging_utils import get_logger
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_chain(llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    return PromptTemplate.from_template("Answer briefly: {text}") | llm | StrOutputParser()


def measure_ttft(chain, prompts: list[str]) -> list[dict]:
    """Exercise 1 solution.

    NEW vs train.py:
    the baseline only checks that streaming yields at least one chunk. Here we
    split the latency into TTFT and total generation time.
    """
    report = []
    for prompt in prompts:
        start = time.perf_counter()
        iterator = iter(chain.stream({"text": prompt}))
        try:
            first_chunk = next(iterator)
            ttft_ms = (time.perf_counter() - start) * 1000
            chunks = [first_chunk]
            for chunk in iterator:
                chunks.append(chunk)
            total_ms = (time.perf_counter() - start) * 1000
        except StopIteration:
            ttft_ms = 0.0
            total_ms = 0.0
            chunks = []
        report.append(
            {
                "prompt": prompt,
                "ttft_ms": round(ttft_ms, 2),
                "total_ms": round(total_ms, 2),
                "n_chunks": len(chunks),
            }
        )
    return report


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """Exercise 2 in-memory semantic cache."""

    def __init__(self, embeddings, threshold: float = 0.95) -> None:
        self.embeddings = embeddings
        self.threshold = threshold
        self.rows: list[tuple[list[float], str]] = []

    def lookup(self, query: str) -> str | None:
        query_vector = self.embeddings.embed_query(query)
        for cached_vector, cached_response in self.rows:
            if cosine_similarity(query_vector, cached_vector) > self.threshold:
                return cached_response
        return None

    def store(self, query: str, response: str) -> None:
        self.rows.append((self.embeddings.embed_query(query), response))


def run_semantic_cache_demo(chain, embeddings) -> dict:
    """Exercise 2 solution for semantic caching."""
    cache = SemanticCache(embeddings=embeddings, threshold=0.95)
    queries = [
        "What is LCEL?",
        "Explain LangGraph.",
        "What is a vector store?",
        "Describe semantic caching.",
        "What does ReAct mean?",
        "Define LCEL.",
        "How would you explain LangGraph?",
        "What is a vector database for embeddings?",
        "What does semantic cache mean?",
        "Expand the acronym ReAct.",
    ]
    hits = 0
    results = []
    for query in queries:
        cached = cache.lookup(query)
        if cached is not None:
            hits += 1
            results.append({"query": query, "source": "cache", "response": cached[:60]})
            continue
        response = chain.invoke({"text": query})
        cache.store(query, response)
        results.append({"query": query, "source": "llm", "response": response[:60]})
    return {"hit_rate": round(hits / len(queries), 4), "results": results}


def build_fastapi_streaming_example(chain) -> str:
    """Exercise 3 endpoint snippet.

    NEW vs train.py:
    the baseline demonstrates async LangChain primitives, but not the actual
    FastAPI route shape that wires them into an HTTP streaming endpoint.
    """
    _ = chain
    return """from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
async def stream_response(q: str):
    async def generate():
        async for chunk in chain.astream({"text": q}):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")
"""


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch4.class2.correction")

    llm = get_llm(cfg)
    embeddings = get_embedding_model(cfg)
    chain = build_chain(llm)
    log.info("Models loaded", llm=cfg.get("backbone", "?"), embed=cfg.get("embed_backbone", "?"))

    print("\n=== Exercise 1: Measure TTFT ===")
    print(
        "NEW vs train.py: streaming latency is decomposed into TTFT and total "
        "generation time instead of only checking whether chunks exist."
    )
    print("Expected output tip: TTFT should be less than or equal to total latency, and local models may emit only 1 chunk.")
    ttft_report = measure_ttft(
        chain,
        [
            "What is LCEL?",
            "Explain LangGraph.",
            "What is RAG?",
            "What is LangSmith?",
            "What is a tool call?",
        ],
    )
    print("Exercise 1 - TTFT report:", ttft_report)

    print("\n=== Exercise 2: Semantic similarity cache ===")
    print(
        "NEW vs train.py: cache hits are based on embedding similarity instead "
        "of exact prompt equality."
    )
    print("Expected output tip: paraphrased later queries should increase the hit rate above zero if embeddings behave reasonably.")
    cache_report = run_semantic_cache_demo(chain, embeddings)
    print("Exercise 2 - hit rate:", cache_report["hit_rate"])
    print("Exercise 2 - sample results:", cache_report["results"][:4])

    print("\n=== Exercise 3: Async streaming endpoint ===")
    print(
        "NEW vs train.py: the async primitives are assembled into a concrete "
        "FastAPI endpoint shape suitable for `uvicorn`."
    )
    print("Expected output tip: the printed code should define `/stream` and yield chunks from `chain.astream(...)`.")
    endpoint_code = build_fastapi_streaming_example(chain)
    print("Exercise 3 - endpoint code:\n", endpoint_code)


if __name__ == "__main__":
    main()
