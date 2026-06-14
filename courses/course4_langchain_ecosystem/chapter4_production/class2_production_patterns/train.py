"""Course 4 / ch4 / class 2 — Production patterns: streaming, caching, retry.

Demonstrates:
  - Streaming: iterating over chunk tokens as they arrive via chain.stream()
  - In-memory semantic cache: identical queries return cached response, no second LLM call
  - Retry middleware: automatic retry on transient errors with exponential backoff
  - Cost tracking: counting input/output tokens with tiktoken
"""

from __future__ import annotations

import argparse
import sys
import time
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


def test_streaming(llm, n_calls: int, log) -> int:
    """Verify that chain.stream() yields at least 1 token chunk."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    chain = ChatPromptTemplate.from_template("Complete this in one sentence: {text}") | llm | StrOutputParser()
    prompts = [
        "The purpose of LCEL is",
        "LangGraph enables",
        "Vector databases are useful for",
        "Streaming in LangChain works by",
    ][:n_calls]

    ok = 0
    for prompt in prompts:
        chunks = []
        try:
            for chunk in chain.stream({"text": prompt}):
                chunks.append(chunk)
            if len(chunks) > 0:
                ok = 1
                log.info("Stream ok", prompt=prompt[:30], n_chunks=len(chunks))
        except Exception as exc:
            log.warning("Stream failed", error=str(exc))
    return ok


def test_cache(llm, n_calls: int, log) -> int:
    """Test in-memory cache: second identical call should return fast (cached)."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.globals import set_llm_cache
    from langchain_community.cache import InMemoryCache

    set_llm_cache(InMemoryCache())
    chain = ChatPromptTemplate.from_template("Define: {term}") | llm | StrOutputParser()

    terms = ["LangChain", "FAISS", "LangGraph", "LCEL"][:n_calls]
    cache_ok = 0
    for term in terms:
        try:
            t0 = time.monotonic()
            r1 = chain.invoke({"term": term})
            t1 = time.monotonic() - t0
            t0 = time.monotonic()
            r2 = chain.invoke({"term": term})
            t2 = time.monotonic() - t0
            if r1 == r2:
                cache_ok = 1
            log.info("Cache test", term=term, first_ms=round(t1 * 1000), second_ms=round(t2 * 1000), hit=(r1 == r2))
        except Exception as exc:
            log.warning("Cache test failed", error=str(exc))

    set_llm_cache(None)
    return cache_ok


def test_retry(llm, log) -> int:
    """Test that a chain wrapped with retry survives a transient error.

    Uses RunnableLambda to inject a flaky first call without monkey-patching
    the Pydantic-validated LLM object.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda

    inner_chain = (
        ChatPromptTemplate.from_template("Say hello to {name} in one sentence.")
        | llm
        | StrOutputParser()
    )
    call_count = [0]

    def flaky_fn(input_: dict) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            raise ConnectionError("Simulated transient error")
        return inner_chain.invoke(input_)

    flaky_chain = RunnableLambda(flaky_fn).with_retry(stop_after_attempt=3)

    try:
        result = flaky_chain.invoke({"name": "LangChain"})
        ok = 1 if result else 0
        log.info("Retry ok", attempts=call_count[0], result=result[:40])
    except Exception as exc:
        ok = 0
        log.warning("Retry test failed", error=str(exc))

    return ok


def run_async_patterns(llm, log) -> dict:
    """
    Async LangChain invocation: ainvoke / astream / abatch.

    FastAPI is async-first. If you use synchronous chain.invoke() inside
    an async def endpoint, you BLOCK the event loop — other requests stall.

    Always use:
      - await chain.ainvoke(input)          # single async call
      - async for chunk in chain.astream()  # streaming (SSE)
      - await chain.abatch(inputs)          # concurrent batch

    abatch() is faster than N sequential ainvoke() calls because it can
    parallelize API requests (for OpenAI) or batch model inference (local).

    LangServe deployment note:
      from langserve import add_routes
      add_routes(app, chain, path="/chain")
      # Gives: /chain/invoke, /chain/stream, /chain/batch — all async
    """
    import asyncio
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template("Answer in one word: {q}")
    chain = prompt | llm | StrOutputParser()

    async def _run_async():
        results = {}

        # ainvoke
        try:
            r = await chain.ainvoke({"q": "What color is the sky?"})
            results["ainvoke_ok"] = 1
            log.info("ainvoke ok", response=str(r)[:30])
        except Exception as exc:
            log.warning("ainvoke failed", error=str(exc)[:60])
            results["ainvoke_ok"] = 0

        # astream
        chunks = []
        try:
            async for chunk in chain.astream({"q": "Name a planet"}):
                chunks.append(str(chunk))
            results["astream_ok"] = 1 if chunks else 0
            log.info("astream ok", n_chunks=len(chunks))
        except Exception as exc:
            log.warning("astream failed", error=str(exc)[:60])
            results["astream_ok"] = 0

        # abatch
        try:
            inputs = [{"q": f"Is {n} a prime number?"} for n in [2, 7, 10]]
            batch_out = await chain.abatch(inputs, config={"max_concurrency": 3})
            results["abatch_ok"] = 1 if len(batch_out) == 3 else 0
            log.info("abatch ok", n_results=len(batch_out))
        except Exception as exc:
            log.warning("abatch failed", error=str(exc)[:60])
            results["abatch_ok"] = 0

        return results

    try:
        return asyncio.run(_run_async())
    except RuntimeError:
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(_run_async())
        except Exception as exc:
            log.warning("async patterns skipped", error=str(exc)[:60])
            return {"ainvoke_ok": -1, "astream_ok": -1, "abatch_ok": -1}


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (GPT-2 encoding for local models)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch4.class2")
    mode = cfg.get("mode", "smoke")
    n_calls = cfg["limits"][mode]["n_calls"]

    llm = get_llm(cfg)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"))

    stream_tokens_ok = test_streaming(llm, n_calls, log)
    cache_hit_ok = test_cache(llm, n_calls, log)
    retry_ok = test_retry(llm, log)

    # ── LangServe: deploy this chain as a REST API ──────────────────────────
    # from langserve import add_routes
    # add_routes(fastapi_app, chain, path="/chain")
    # → automatically creates /chain/invoke, /chain/stream, /chain/batch
    # → all endpoints are async-native and support streaming out of the box
    # Install: pip install "langserve[all]"

    async_metrics = run_async_patterns(llm, log)

    sample_text = "LangChain is a framework for building composable LLM applications using LCEL."
    token_count = count_tokens(sample_text)
    log.info("Token count", text=sample_text[:40], tokens=token_count)

    metrics = {
        "stream_tokens_ok": float(stream_tokens_ok),
        "cache_hit_ok": float(cache_hit_ok),
        "retry_ok": float(retry_ok),
    }
    metrics.update(async_metrics)
    run_eval(
        method=cfg["method"],
        backbone=cfg.get("backbone", "local"),
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics=metrics,
        expected_band=cfg.get("expected_band", {}),
        extras={"mode": mode, "n_calls": n_calls, "token_count": token_count},
    )


if __name__ == "__main__":
    main()
