# LangChain Ecosystem Mastery Guide

A comprehensive, end-to-end reference for `course4_langchain_ecosystem/`. Covers every concept, every pipeline, every function, and every flow — from zero knowledge to production proficiency.

---

## Table of Contents

1. [Setup](#1-setup)
2. [Core Concepts](#2-core-concepts)
3. [LCEL — Composing Pipelines](#3-lcel--composing-pipelines)
4. [Memory and Conversation State](#4-memory-and-conversation-state)
5. [Structured Output with Pydantic](#5-structured-output-with-pydantic)
6. [Vector Stores: FAISS and ChromaDB](#6-vector-stores-faiss-and-chromadb)
7. [Advanced RAG Strategies](#7-advanced-rag-strategies)
8. [RAG Evaluation with RAGAS](#8-rag-evaluation-with-ragas)
9. [Tools and Function Calling](#9-tools-and-function-calling)
10. [ReAct Agents](#10-react-agents)
11. [LangGraph: Stateful Workflows](#11-langgraph-stateful-workflows)
12. [LangSmith: Tracing and Observability](#12-langsmith-tracing-and-observability)
13. [Production Patterns](#13-production-patterns)
14. [End-to-End Flow Diagrams](#14-end-to-end-flow-diagrams)
15. [Verification Checklist](#15-verification-checklist)

---

## 1. Setup

### Environment

```bash
conda activate slm-gpu
# Verify LangChain installation:
python -c "import langchain; print(langchain.__version__)"  # >= 1.3.x
python -c "import langgraph; print(langgraph.__version__)"
python -c "import langsmith; print(langsmith.__version__)"
python -c "import faiss; print('faiss ok')"
python -c "import chromadb; print(chromadb.__version__)"
python -c "import ragas; print('ragas ok')"
```

### Environment variables

```bash
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=5,6,7            # GPU selection (lab policy)
# Optional:
export LANGSMITH_API_KEY="ls-..."            # enables live tracing
export OPENAI_API_KEY="sk-..."              # enables ChatOpenAI backend
```

### Shared utilities (used by all 11 classes)

| Module | Function | Purpose |
|---|---|---|
| `shared/llm_client.py` | `get_llm(cfg)` | Returns `HuggingFacePipeline` or `ChatOpenAI` |
| `shared/llm_client.py` | `get_embedding_model(cfg)` | Returns `HuggingFaceEmbeddings` |
| `shared/vector_store.py` | `build_store(docs, cfg, embeddings)` | FAISS or ChromaDB from documents |
| `shared/vector_store.py` | `query_store(store, query, k, search_type)` | similarity or MMR retrieval |
| `shared/logging_utils.py` | `get_logger(name)` | `StructLogger` with kwargs support |
| `shared/eval_harness.py` | `run_eval(...)` | Band-check + JSON result writer |
| `shared/config.py` | `load_yaml(path)`, `apply_overrides(cfg, overrides)` | YAML config + CLI overrides |

### Config structure (every class)

```yaml
course:   course4_langchain_ecosystem
class_id: chapter1_core_class1_lcel_chains
task:     chain_composition
method:   lcel_pipeline

provider:       huggingface           # or "openai"
backbone:       HuggingFaceTB/SmolLM2-135M-Instruct
embed_backbone: sentence-transformers/all-MiniLM-L6-v2
max_new_tokens: 128
temperature:    0.1

mode: smoke
limits:
  smoke: { n_calls: 3 }
  full:  { n_calls: 10 }

expected_band:
  chain_ok: [1, 1]
```

---

## 2. Core Concepts

### The LangChain abstraction hierarchy

```
Runnable (interface)
  ├── PromptTemplate / ChatPromptTemplate
  ├── BaseChatModel (ChatHuggingFace, ChatOpenAI)
  ├── OutputParser (StrOutputParser, JsonOutputParser)
  ├── RunnableLambda (any Python callable)
  ├── RunnableParallel (fan-out)
  └── RunnableWithMessageHistory (stateful)

Chain = Runnable | Runnable | Runnable
  └── composed via LCEL pipe operator
```

### Key abstractions

| Concept | What it is | When to use |
|---|---|---|
| **Runnable** | Protocol with `.invoke()`, `.batch()`, `.stream()` | Building block of all chains |
| **LCEL** | `|` operator composing Runnables into a chain | Connecting prompt → LLM → parser |
| **Memory** | `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` | Multi-turn conversations |
| **OutputParser** | Converts LLM string → structured Python object | Schema-validated generation |
| **VectorStore** | Indexed embedding corpus supporting ANN retrieval | Semantic search, RAG |
| **Retriever** | `VectorStore.as_retriever()` + variants | Pluggable in RAG chains |
| **Tool** | `@tool`-decorated function with schema | Agent capabilities |
| **Agent** | ReAct loop: model selects tool, observes result, repeats | Multi-step reasoning |
| **StateGraph** | LangGraph directed graph of state-transforming nodes | Stateful multi-actor workflows |
| **Tracer** | LangSmith callback — captures every invocation as a run | Observability, debugging |

---

## 3. LCEL — Composing Pipelines

**File:** `chapter1_core/class1_lcel_chains/train.py`

### The pipe operator

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_template("Summarize: {text}") | llm | StrOutputParser()
```

Each `|` creates a `RunnableSequence`. The output type of the left component must match the input type of the right:
- `ChatPromptTemplate` → `ChatPromptValue`
- `ChatModel` → `AIMessage`
- `StrOutputParser` → `str`

### Three invocation modes

| Mode | Call | Returns | Use case |
|---|---|---|---|
| `invoke` | `chain.invoke({"text": "..."})` | Single result | Synchronous, one shot |
| `batch` | `chain.batch([{"text": "..."}, ...])` | List of results | Parallel processing |
| `stream` | `chain.stream({"text": "..."})` | Token iterator | Streaming UI |

### LCEL fan-out with RunnableParallel

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
)
result = chain.invoke({"text": "..."})
# result == {"summary": "...", "keywords": "..."}
```

### Fallback on error

```python
chain_with_fallback = primary_chain.with_fallbacks([backup_chain])
```

---

## 4. Memory and Conversation State

**File:** `chapter1_core/class2_memory_conversation/train.py`

### InMemoryChatMessageHistory

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

history = InMemoryChatMessageHistory()
history.add_user_message("My name is Alice.")
history.add_ai_message("Hello Alice!")
print(len(history.messages))  # 2
```

### RunnableWithMessageHistory

Wraps a chain to automatically inject history into each invocation:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain, get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
chain_with_history.invoke(
    {"input": "Hello"},
    config={"configurable": {"session_id": "alice"}},
)
```

**Flow:** Each call → `get_session_history(session_id)` → append to `MessagesPlaceholder` → LLM sees full context → append AI response to history.

---

## 5. Structured Output with Pydantic

**File:** `chapter1_core/class3_structured_output/train.py`

### JsonOutputParser + Pydantic schema

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

class Movie(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year as integer")
    genre: str = Field(description="Primary genre")

parser = JsonOutputParser(pydantic_object=Movie)
prompt = PromptTemplate(
    template="Extract JSON:\n{format_instructions}\n\nText: {text}\n\nJSON:",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | llm | parser
result = chain.invoke({"text": "The Matrix (1999) is a sci-fi film."})
# result == {"title": "The Matrix", "year": 1999, "genre": "sci-fi"}
```

### Fallback extraction

When the LLM ignores format instructions (common for small models):

```python
import json
raw = (prompt | llm | StrOutputParser()).invoke({"text": text})
start, end = raw.find("{"), raw.rfind("}") + 1
if start >= 0 and end > start:
    obj = json.loads(raw[start:end])
```

---

## 6. Vector Stores: FAISS and ChromaDB

**File:** `chapter2_vector_rag/class1_vector_stores/train.py`

### Embedding → Index → Retrieve

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs = [Document(page_content=text, metadata={"id": doc_id}) for doc_id, text in corpus]

# FAISS (in-memory, exact ANN)
faiss_store = FAISS.from_documents(docs, embeddings)
results = faiss_store.similarity_search("What is FAISS?", k=3)

# ChromaDB (persistent, HNSW)
chroma_store = Chroma.from_documents(docs, embeddings, collection_name="demo")
results = chroma_store.similarity_search("What is FAISS?", k=3)

# MMR (Maximal Marginal Relevance — diverse results)
results = faiss_store.max_marginal_relevance_search("vector database", k=3, fetch_k=10)
```

### Recall@k metric

```python
def recall_at_k(retrieved_ids, relevant_ids):
    return sum(1 for r in retrieved_ids if r in relevant_ids) / max(len(relevant_ids), 1)
```

### `build_store` and `query_store` (shared utility)

```python
from shared.vector_store import build_store, query_store

cfg["vector_store"] = {"backend": "faiss"}   # or "chromadb"
store = build_store(docs, cfg, embeddings)
results = query_store(store, "What is FAISS?", k=3, search_type="mmr")
```

---

## 7. Advanced RAG Strategies

**File:** `chapter2_vector_rag/class2_advanced_rag/train.py`

### MultiQueryRetriever

Generates N paraphrased queries → unions results → deduplicates:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
mq = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
results = mq.invoke("What is multi-query retrieval?")
```

### ContextualCompressionRetriever + EmbeddingsFilter

Extracts only the relevant span from each retrieved document:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainExtractor

# Fast (embedding cosine threshold):
ef = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
cr = ContextualCompressionRetriever(base_compressor=ef, base_retriever=base_retriever)

# Accurate (LLM extracts relevant sentence):
lce = LLMChainExtractor.from_llm(llm)
cr = ContextualCompressionRetriever(base_compressor=lce, base_retriever=base_retriever)
```

### Hybrid BM25 + Dense (EnsembleRetriever)

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25 = BM25Retriever.from_documents(docs)
bm25.k = k
faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": k})
ensemble = EnsembleRetriever(retrievers=[bm25, faiss_retriever], weights=[0.5, 0.5])
results = ensemble.invoke(query)
```

---

## 8. RAG Evaluation with RAGAS

**File:** `chapter2_vector_rag/class3_rag_eval/train.py`

### Evaluation dataset format

```python
rows = [{
    "user_input": "What is LangChain?",
    "retrieved_contexts": ["LangChain is a framework..."],
    "response": "LangChain enables composable LLM pipelines.",
    "reference": "LangChain is a framework for building LLM applications.",
}]
```

### Running RAGAS

```python
from ragas import EvaluationDataset, evaluate
from ragas.metrics import ContextRecall, SemanticSimilarity
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

dataset = EvaluationDataset.from_list(rows)
result = evaluate(
    dataset,
    metrics=[ContextRecall(), SemanticSimilarity()],
    llm=LangchainLLMWrapper(llm),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
)
df = result.to_pandas()
print(df[["context_recall", "semantic_similarity"]].mean())
```

### Metric interpretation

| Metric | Low (<0.5) | Good (>=0.7) |
|---|---|---|
| `context_recall` | Retriever missing relevant docs | Retriever finding reference context |
| `semantic_similarity` | Generator producing off-topic answers | Generator answers match reference |
| `faithfulness` | Generator hallucinating | Generator stays grounded in context |

---

## 9. Tools and Function Calling

**File:** `chapter3_agents/class1_tools_function_calling/train.py`

### @tool decorator

```python
from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

# Schema inspection
print(add.name)          # "add"
print(add.description)   # "Add two numbers..."
print(add.args_schema)   # Pydantic BaseModel with a: float, b: float

# Direct invocation
result = add.invoke({"a": 3.0, "b": 4.0})   # 7.0
```

### Binding tools to the LLM

```python
llm_with_tools = llm.bind_tools([add, multiply, sqrt])
response = llm_with_tools.invoke("What is 3 + 4?")
if response.tool_calls:
    for call in response.tool_calls:
        result = add.invoke(call["args"])
```

**Flow:** LLM receives tool schemas in a special format → generates tool call JSON → your code executes the tool → pass result back to LLM.

---

## 10. ReAct Agents

**File:** `chapter3_agents/class2_react_agent/train.py`

### AgentExecutor with tool_calling_agent

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools to solve problems step by step."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent, tools=tools,
    max_iterations=5, handle_parsing_errors=True,
    return_intermediate_steps=True, verbose=False,
)
result = executor.invoke({"input": "What is sqrt(3 * 48)?"})
# result["output"] == "12.0"
# result["intermediate_steps"] == [(AgentAction(tool="multiply"), "144"), (AgentAction(tool="sqrt"), "12.0")]
```

### ReAct loop flow

```
input: "What is sqrt(3 * 48)?"
  │
  ▼ Thought: "I need to multiply 3 and 48 first."
  │ Action: multiply(a=3, b=48)
  ▼ Observation: 144
  │ Thought: "Now compute sqrt(144)."
  │ Action: sqrt(x=144)
  ▼ Observation: 12.0
  │ Thought: "I have the answer."
  ▼ Final Answer: "12.0"
```

---

## 11. LangGraph: Stateful Workflows

**File:** `chapter3_agents/class3_langgraph_stateful/train.py`

### StateGraph anatomy

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

class MyState(TypedDict):
    text: str
    label: str
    n_retries: int
    done: bool

def classify(state: MyState) -> MyState:
    # Call LLM, update state
    return {**state, "label": "positive"}

def finalize(state: MyState) -> MyState:
    return {**state, "done": True}

def route(state: MyState) -> Literal["finalize", "retry"]:
    return "finalize" if state["label"] else "retry"

graph = StateGraph(MyState)
graph.add_node("classify", classify)
graph.add_node("finalize", finalize)
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route)
graph.add_edge("finalize", END)

app = graph.compile()
result = app.invoke({"text": "Great!", "label": "", "n_retries": 0, "done": False})
```

### Node function contract

- Input: full `TypedDict` state
- Output: partial or full updated state dict
- Must not return `None`
- All TypedDict fields must be present in the initial state passed to `invoke()`

### Conditional edge routing

```python
def route(state) -> str:
    if state["label"] in {"positive", "negative", "neutral"}:
        return "finalize"
    return "retry"     # back to classify node (cycle)
graph.add_conditional_edges("classify", route)
```

---

## 12. LangSmith: Tracing and Observability

**File:** `chapter4_production/class1_langsmith_tracing/train.py`

### Live tracing setup

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "my-project"
# All chain.invoke() calls are now automatically traced.
```

### Offline JSONL fallback

```python
import json, uuid, time

record = {
    "run_id": str(uuid.uuid4()),
    "input": prompt,
    "output": response,
    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
}
with open("traces.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")
```

### Run ID uniqueness check

```python
records = [json.loads(line) for line in Path("traces.jsonl").read_text().splitlines()]
ids = [r["run_id"] for r in records]
assert len(set(ids)) == len(ids), "Duplicate run IDs detected"
```

---

## 13. Production Patterns

**File:** `chapter4_production/class2_production_patterns/train.py`

### Streaming

```python
for chunk in chain.stream({"text": "Explain LCEL"}):
    print(chunk, end="", flush=True)
# FastAPI:
return StreamingResponse((c for c in chain.stream(...)), media_type="text/plain")
```

### In-memory cache

```python
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

set_llm_cache(InMemoryCache())
r1 = chain.invoke({"text": "What is LangChain?"})
r2 = chain.invoke({"text": "What is LangChain?"})
assert r1 == r2   # cache hit
set_llm_cache(None)  # reset after test
```

### Retry middleware

```python
chain_with_retry = chain.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
```

### Token counting

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
n_tokens = len(enc.encode(text))
```

---

## 14. End-to-End Flow Diagrams

### RAG pipeline (class2 → class3)

```
Document corpus
    │ embed (all-MiniLM-L6-v2)
    ▼
FAISS / ChromaDB index
    │
User query ──► embed query
                │
                ▼ similarity_search / mmr / multi_query / compression
                │
           top-k documents
                │
     PromptTemplate.format(context, query)
                │
                ▼
         LLM (SmolLM2-135M)
                │
                ▼
        StrOutputParser → answer
                │
      RAGAS evaluation → context_recall, semantic_similarity
```

### Agent loop (class1 → class2 → class3)

```
User task
    │
    ▼
AgentExecutor
    │
    ├── Thought (LLM reasoning)
    ├── Tool selection (bind_tools schema)
    ├── Tool execution (ToolRegistry.dispatch)
    └── Observation → back to Thought until done
    │
Final answer
    │
LangGraph StateGraph (optional)
    ├── Node: run_agent → update state
    ├── Conditional edge: route based on state
    └── Node: end → return
    │
TraceWriter → JSONL or LangSmith
```

### Production chain (class2 production)

```
User request
    │
FastAPI endpoint
    │
    ├── SemanticCache lookup (cosine >= 0.95) ──► cached response
    │                                              (skip LLM call)
    │ (cache miss)
    ▼
chain.with_retry(3)
    │
LCEL: PromptTemplate | LLM | StrOutputParser
    │
Response
    │
    ├── CostTracker.record(n_tokens_in, n_tokens_out, latency_ms)
    ├── SemanticCache.set(query, response)
    └── TraceWriter → traces.jsonl
```

---

## 15. Verification Checklist

Run these commands in order. Each must exit 0.

```bash
BASE="courses/course4_langchain_ecosystem"

# Chapter 1: Core
bash ${BASE}/chapter1_core/class1_lcel_chains/run.sh
bash ${BASE}/chapter1_core/class2_memory_conversation/run.sh
bash ${BASE}/chapter1_core/class3_structured_output/run.sh

# Chapter 2: Vector RAG
bash ${BASE}/chapter2_vector_rag/class1_vector_stores/run.sh
bash ${BASE}/chapter2_vector_rag/class2_advanced_rag/run.sh
bash ${BASE}/chapter2_vector_rag/class3_rag_eval/run.sh

# Chapter 3: Agents
bash ${BASE}/chapter3_agents/class1_tools_function_calling/run.sh
bash ${BASE}/chapter3_agents/class2_react_agent/run.sh
bash ${BASE}/chapter3_agents/class3_langgraph_stateful/run.sh

# Chapter 4: Production
bash ${BASE}/chapter4_production/class1_langsmith_tracing/run.sh
bash ${BASE}/chapter4_production/class2_production_patterns/run.sh
```

Expected: all 11 exit 0 with metric bands satisfied.

### Production projects

```bash
BASE="courses/projects"
bash ${BASE}/01_smolsearch/run.sh
bash ${BASE}/02_ragify/run.sh
bash ${BASE}/03_agentflow/run.sh
bash ${BASE}/04_llmops_baseline/run.sh
```

Expected: all 4 exit 0.

### Shared unit tests

```bash
pytest tests/ -q    # all pre-existing shared module tests still green
```

---

## Key Terminology Reference

| Term | Definition |
|---|---|
| **LCEL** | LangChain Expression Language — `|` operator composing Runnables |
| **Runnable** | Any object with `.invoke()`, `.batch()`, `.stream()` interface |
| **RunnableSequence** | Chain of Runnables created by `|` composition |
| **RunnableParallel** | Dict of Runnables running concurrently (fan-out) |
| **RunnableWithMessageHistory** | Runnable that auto-injects conversation history |
| **PromptTemplate** | Template with `{variable}` placeholders for completion models |
| **ChatPromptTemplate** | Message-list template for chat models |
| **MessagesPlaceholder** | Dynamic slot in ChatPromptTemplate for history injection |
| **OutputParser** | Converts LLM string output to structured Python objects |
| **JsonOutputParser** | Extracts JSON matching a Pydantic schema |
| **InMemoryChatMessageHistory** | In-process message history store (per session) |
| **VectorStore** | Indexed embedding corpus with ANN retrieval |
| **FAISS** | Facebook AI Similarity Search — in-memory ANN index |
| **ChromaDB** | Persistent vector database with metadata filtering |
| **Retriever** | `VectorStore.as_retriever()` — LangChain retrieval interface |
| **MMR** | Maximal Marginal Relevance — diversity-aware retrieval |
| **MultiQueryRetriever** | Generates N query variants, unions results |
| **ContextualCompressionRetriever** | Extracts relevant span from retrieved docs |
| **EnsembleRetriever** | Fuses multiple retrievers (BM25 + dense) |
| **RAG** | Retrieval-Augmented Generation |
| **RAGAS** | RAG Assessment — metrics: context_recall, faithfulness, semantic_similarity |
| **@tool** | Decorator wrapping Python function as LangChain tool with schema |
| **bind_tools** | Attaches tool schemas to an LLM for function calling |
| **AgentExecutor** | Wraps ReAct agent loop with tools, iteration guard, error handling |
| **create_tool_calling_agent** | Builds agent using tool calling (preferred over ReAct text parsing) |
| **ReAct** | Reasoning + Acting — interleaved thought/action/observation loop |
| **StateGraph** | LangGraph graph with shared typed state |
| **Node** | State-transforming function in a StateGraph |
| **Conditional edge** | Routing function `(state) -> str` selecting the next node |
| **LangSmith** | Observability platform for LangChain — traces every run |
| **run_id** | UUID identifying a single chain invocation in LangSmith |
| **Streaming** | Token-by-token output via `chain.stream()` |
| **InMemoryCache** | Exact-key LLM response cache |
| **SemanticCache** | Cosine-similarity-based LLM response cache |
| **with_retry** | LCEL middleware retrying on exception with exponential backoff |
| **tiktoken** | OpenAI tokenizer for counting input/output tokens |
| **HuggingFacePipeline** | LangChain wrapper for local HuggingFace `pipeline()` |
| **ChatHuggingFace** | Chat-model interface wrapping `HuggingFacePipeline` |
| **HuggingFaceEmbeddings** | LangChain wrapper for sentence-transformers |
