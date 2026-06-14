# Class 1 — LCEL Chains (PromptTemplate → LLM → OutputParser)

## Psycho Mode

Think of LCEL (LangChain Expression Language) as Unix pipes for LLM calls. Just as `cat file | grep pattern | wc -l` chains three programs by passing stdout to stdin, LCEL uses the `|` operator to chain a prompt, a model, and a parser. Each component is a "Runnable" — anything that accepts an input and returns an output. The beauty is composability: you can swap out any piece without touching the others. A chain is not a class you subclass; it is a pipeline you compose at the point of use.

The three-step mental model: (1) "What do I say to the model?" — that is your PromptTemplate. (2) "Which model do I call?" — that is your LLM or ChatModel. (3) "How do I parse the response?" — that is your OutputParser. LCEL wires them together so that `.invoke()`, `.batch()`, and `.stream()` all work uniformly on the composed pipeline.

## Academic Mode

Formally, a LangChain Runnable implements the interface:

$$R : \mathcal{I} \to \mathcal{O}$$

where $\mathcal{I}$ is the input type and $\mathcal{O}$ is the output type. The `|` operator composes two runnables $R_1 : A \to B$ and $R_2 : B \to C$ into $R_1 | R_2 : A \to C$ via function composition:

$$(R_1 | R_2)(x) = R_2(R_1(x))$$

Batch execution maps over a list with configurable concurrency:

$$\text{batch}(R, [x_1, \ldots, x_n]) = [R(x_1), \ldots, R(x_n)]$$

Streaming returns an iterator over partial outputs, enabling token-by-token display before the full response is complete. The LCEL specification is documented in the LangChain Expression Language reference: [https://python.langchain.com/docs/concepts/lcel/](https://python.langchain.com/docs/concepts/lcel/).

## Engineering Mode

The key chain pattern:

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = PromptTemplate.from_template("Summarize: {text}") | llm | StrOutputParser()

# Single call
result = chain.invoke({"text": "..."})

# Parallel batch (uses a thread pool internally)
results = chain.batch([{"text": "..."}, {"text": "..."}])

# Streaming (yields str chunks as they arrive)
for chunk in chain.stream({"text": "..."}):
    print(chunk, end="", flush=True)
```

Gotchas:
- Local HuggingFace models may not support true token-by-token streaming; `chain.stream()` may return a single chunk. This is a backend limitation, not an LCEL bug.
- `batch()` is synchronous by default. For async, use `abatch()`.
- `PromptTemplate` vs `ChatPromptTemplate`: use `ChatPromptTemplate` for chat models (HuggingFace chat-instruct variants), `PromptTemplate` for completion models.

Config keys: `backbone`, `max_new_tokens`, `temperature`, `limits.smoke.n_calls`.

## Research Mode

LCEL is architecturally inspired by functional reactive programming (FRP) and lazy evaluation graphs. Open questions: (1) Can LCEL chains be automatically parallelized across independent branches using static analysis? (2) How does LCEL compose with structured concurrency (Python `asyncio` task groups)? (3) The `RunnableParallel` combinator (a dict of Runnables) allows fan-out; how does its scheduling interact with GPU memory limits for batched local models?

Recent extension: LangChain 0.3 introduced `RunnablePassthrough.assign()` for in-place context enrichment — a common pattern in RAG pipelines where retrieved docs are appended to the input dict before the final LLM call.

## How to run

```bash
bash courses/course4_langchain_ecosystem/chapter1_core/class1_lcel_chains/run.sh
# Full mode:
MODE=full bash courses/course4_langchain_ecosystem/chapter1_core/class1_lcel_chains/run.sh
```

## How to verify

Expected metric band (from `configs/default.yaml`):

| Metric | Expected |
|---|---|
| `chain_ok` | [1, 1] |
| `batch_ok` | [1, 1] |
| `stream_ok` | [1, 1] |

`eval.py` exits non-zero if any metric falls outside the band.

---

**Instructor checklist**

- [x] All four mode sections present and >= 2 paragraphs each.
- [x] Official reference links included.
- [x] `train.py` / `eval.py`: no numeric literals except 0/1.
- [x] `configs/default.yaml` declares `expected_band` for every metric.
- [x] `run.sh` uses `HF_HOME` and is `chmod +x`.
- [x] `exercises.md` has exactly 3 exercises.
- [x] Linked from parent course README.
