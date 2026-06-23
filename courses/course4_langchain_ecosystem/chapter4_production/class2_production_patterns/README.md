# Class 2 — Production Patterns: Streaming, Caching, Retry

> Goal: harden an LLM application with streaming, caching, and retry so the student sees the concrete difference between a notebook demo and a service that behaves well under real user traffic.

## Psycho Mode

Three patterns separate a prototype from a production LLM service. First, streaming: users hate staring at a blank screen for 5 seconds while the full response generates. Streaming pushes tokens to the client as they arrive, making the product feel responsive even for slow models. Second, caching: if 30% of your users ask "what is your return policy?" every day, why call the LLM 10,000 times? A semantic cache stores responses and retrieves them for similar (not just identical) queries. Third, retry: networks fail, rate limits hit, GPU memory spikes. A retry decorator catches transient errors and retries with backoff instead of failing the user.

These are not LangChain-specific patterns — they apply to any LLM integration. But LangChain provides first-class support for all three, making them easy to add without changing business logic.

## Academic Mode

Streaming is an iterator protocol: `chain.stream(input)` returns a `Generator[str]` that yields each token chunk as it is produced. Total time-to-first-token (TTFT) is independent of total generation length, which is the key latency metric for streaming endpoints.

Caching maps (prompt, model) to a stored response. Semantic caching uses an embedding similarity threshold $\tau$ instead of exact key equality:

$$\text{cache\_hit}(q) = \exists\, q' \in \mathcal{C} \text{ s.t. } \cos(\phi(q), \phi(q')) \geq \tau$$

where $\mathcal{C}$ is the set of cached prompts. `InMemoryCache` uses exact key equality (fast, simple); Redis-based semantic cache uses vector similarity. Retry uses truncated exponential backoff: the $i$-th retry waits $\min(2^i \cdot \delta, T_{\max})$ seconds. Reference: LangChain caching — [https://python.langchain.com/docs/how_to/llm_caching/](https://python.langchain.com/docs/how_to/llm_caching/).

## Engineering Mode

```python
# Streaming
from langchain_core.globals import set_llm_cache
for chunk in chain.stream({"text": "Explain LCEL"}):
    print(chunk, end="", flush=True)

# In-memory cache
from langchain_community.cache import InMemoryCache
set_llm_cache(InMemoryCache())
r1 = chain.invoke({"text": "What is LangChain?"})  # LLM call
r2 = chain.invoke({"text": "What is LangChain?"})  # Cache hit (same key)
assert r1 == r2

# Retry middleware
chain_with_retry = chain.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
result = chain_with_retry.invoke({"text": "Hello"})

# Token counting
import tiktoken
enc = tiktoken.get_encoding("gpt2")
n_tokens = len(enc.encode(text))
```

Gotchas:
- `InMemoryCache` is exact-match; two prompts that differ by whitespace are different keys.
- `chain.with_retry()` retries on any `Exception`. Narrow it with `retry_if_exception_type` to avoid retrying on `KeyboardInterrupt` or `SystemExit`.
- Local HuggingFace models may not support token-by-token streaming; `chain.stream()` may yield a single chunk. This is not a bug — it is a backend limitation.
- `set_llm_cache(None)` clears the global cache; always reset after tests to avoid cross-test contamination.

Config keys: `backbone`, `cache.backend`, `limits.smoke.n_calls`.

## Research Mode

Production LLM serving at scale requires additional patterns beyond what LangChain provides out of the box: (1) prefix caching — major inference servers (vLLM, TGI) cache the KV state of common prompt prefixes, reducing TTFT for templated prompts; (2) speculative decoding — a small draft model generates candidate tokens, the large model verifies in parallel, yielding 2-4x throughput improvement; (3) batching — combining multiple user requests into a single forward pass improves GPU utilization; (4) request queuing with backpressure — when the queue is full, respond with a 429 rather than queuing forever, which degrades user experience uniformly. For semantic caching at scale, use a vector database (Redis with RediSearch, Qdrant) instead of in-memory.

## How to run

```bash
bash courses/course4_langchain_ecosystem/chapter4_production/class2_production_patterns/run.sh
MODE=full bash courses/course4_langchain_ecosystem/chapter4_production/class2_production_patterns/run.sh
```

## How to verify

| Metric | Expected |
|---|---|
| `stream_tokens_ok` | [1, 1] |
| `cache_hit_ok` | [1, 1] |
| `retry_ok` | [1, 1] |

---

**Instructor checklist**

- [x] All four mode sections present.
- [x] Official reference links included.
- [x] No numeric literals except 0/1 in code files.
- [x] `configs/default.yaml` declares `expected_band`.
- [x] `run.sh` chmod+x.
- [x] `exercises.md` has 3 exercises.
- [x] Linked from course README.
