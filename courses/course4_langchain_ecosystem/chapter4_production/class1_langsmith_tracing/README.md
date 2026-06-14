# Class 1 — LangSmith Tracing and Observability

## Psycho Mode

Debugging a LangChain application without observability is like debugging a distributed system without logs: you know something went wrong, but you cannot see where. LangSmith is the logging and tracing layer for the LangChain ecosystem. Every chain invocation becomes a "run" with a unique ID, a recorded input and output, a latency, and a full tree of child spans for each sub-step.

The workflow: you set an API key, enable tracing with `LANGCHAIN_TRACING_V2=true`, and LangChain automatically sends trace data to LangSmith in the background. You then open the LangSmith dashboard to replay any run, inspect intermediate steps, annotate examples for datasets, and trigger automated evaluations. Without an API key — in a local-only setup — this class demonstrates the same trace structure written to a local JSONL file, so you understand what is being captured before you ever touch the hosted service.

## Academic Mode

A LangSmith trace is a hierarchical run tree. At the root is the chain invocation (a "run"). Each node function call, tool invocation, and LLM call is a child span with:
- `run_id`: UUID uniquely identifying this invocation
- `parent_run_id`: UUID of the parent span, or null for the root
- `inputs`: the input dict passed to this span
- `outputs`: the output dict returned
- `start_time`, `end_time`: wall-clock timestamps for latency
- `error`: exception message if the span failed

The trace tree enables root-cause analysis: if a chain fails, you trace from the failing span up to the root to find which upstream step produced the bad input. LangSmith also supports dataset annotation: you can mark any run as "positive" or "negative" and export it as a labeled dataset for automated regression testing. Documentation: [https://docs.smith.langchain.com/](https://docs.smith.langchain.com/).

## Engineering Mode

```python
import os

# Enable live tracing (requires LANGSMITH_API_KEY)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "course4-demo"

# All chain invocations are now automatically traced
result = chain.invoke({"question": "What is LangSmith?"})
```

Offline trace pattern (no API key):

```python
import json, uuid, time

record = {
    "run_id": str(uuid.uuid4()),
    "input": prompt,
    "output": result,
    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
}
with open("traces.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")
```

Config keys: `backbone`, `offline_trace_path`, `limits.smoke.n_traces`.

Gotchas:
- `LANGCHAIN_TRACING_V2` must be set before the first chain call, not after.
- Traces are sent asynchronously; call `langsmith.Client().list_runs()` to verify they arrived.
- The offline JSONL pattern is used in this class as the primary verification path, ensuring the class passes without an API key.

## Research Mode

LangSmith is purpose-built for LLM application observability, but it is not the only option. Alternatives: (1) OpenTelemetry with a custom LangChain callback that emits OTEL spans — vendor-neutral and integrates with Datadog, Honeycomb, Jaeger; (2) Weights & Biases Traces — similar dashboard but integrated with the W&B ML experiment tracking ecosystem; (3) Arize Phoenix — open-source LLM observability with embedding visualization for RAG debugging. Production engineering tip: always store `run_id` alongside your application logs. When a user reports a bad response, cross-reference the `run_id` to replay the exact chain execution in LangSmith.

## How to run

```bash
bash courses/course4_langchain_ecosystem/chapter4_production/class1_langsmith_tracing/run.sh
MODE=full bash courses/course4_langchain_ecosystem/chapter4_production/class1_langsmith_tracing/run.sh
```

## How to verify

| Metric | Expected |
|---|---|
| `traces_written_ok` | [1, 1] |
| `run_ids_unique_ok` | [1, 1] |

---

**Instructor checklist**

- [x] All four mode sections present.
- [x] Official reference links included.
- [x] No numeric literals except 0/1 in code files.
- [x] `configs/default.yaml` declares `expected_band`.
- [x] `run.sh` chmod+x.
- [x] `exercises.md` has 3 exercises.
- [x] Linked from course README.
