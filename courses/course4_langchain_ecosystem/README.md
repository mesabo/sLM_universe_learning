# Course 4 — LangChain Ecosystem

**Target audience:** Engineers who already understand sLM fine-tuning (Courses 0–3) and need hands-on proficiency with the framework layer that appears in every modern LLM job description.

**What you build:** Four progressively deeper chapters covering the full LangChain / LangGraph / LangSmith stack, with local sLM backends (no API key required) and optional OpenAI integration.

---

## Chapter map

| Chapter | Class | Topic | CV keyword |
|---|---|---|---|
| **ch1 — Core** | class1_lcel_chains | LCEL: PromptTemplate → LLM → OutputParser; invoke / batch / stream | LangChain, LCEL |
| **ch1 — Core** | class2_memory_conversation | ConversationBuffer, RunnableWithMessageHistory, multi-turn sessions | Memory, stateful chat |
| **ch1 — Core** | class3_structured_output | PydanticOutputParser, JsonOutputParser, Pydantic schema validation | Structured output |
| **ch2 — Vector RAG** | class1_vector_stores | FAISS vs ChromaDB: embed, index, similarity_search, MMR | Vector DB, FAISS, ChromaDB |
| **ch2 — Vector RAG** | class2_advanced_rag | MultiQueryRetriever, ContextualCompressionRetriever, hybrid BM25+dense | Advanced RAG |
| **ch2 — Vector RAG** | class3_rag_eval | RAGAS: context_recall, semantic_similarity, evaluation dataset | RAG evaluation, RAGAS |
| **ch3 — Agents** | class1_tools_function_calling | @tool decorator, schema, bind_tools, direct invocation | Tool use, function calling |
| **ch3 — Agents** | class2_react_agent | ReAct loop, AgentExecutor, intermediate steps, iteration guard | ReAct agents |
| **ch3 — Agents** | class3_langgraph_stateful | StateGraph, nodes, conditional edges, cycles, shared state | LangGraph |
| **ch4 — Production** | class1_langsmith_tracing | LangSmith tracing, offline JSONL fallback, run IDs | LangSmith, observability |
| **ch4 — Production** | class2_production_patterns | Streaming, in-memory cache, retry middleware, token counting | Production LLM, streaming |

---

## How to run a class

```bash
# Smoke run (fast, minimal corpus)
bash courses/course4_langchain_ecosystem/chapter1_core/class1_lcel_chains/run.sh

# Full run
MODE=full bash courses/course4_langchain_ecosystem/chapter1_core/class1_lcel_chains/run.sh
```

All classes default to `mode: smoke` in `configs/default.yaml`. Override with `MODE=full`.

## Shared utilities added for Course 4

| Module | Purpose |
|---|---|
| `shared/llm_client.py` | `get_llm(cfg)`: returns `ChatHuggingFace` (local) or `ChatOpenAI` (when `OPENAI_API_KEY` set) |
| `shared/vector_store.py` | `build_store(docs, cfg, embeddings)`: FAISS or ChromaDB; `query_store(store, q, k)`: similarity or MMR |

## Environment

All Course 4 dependencies are in `env/slm-gpu.yml`. Install:

```bash
conda env update -n slm-gpu -f env/slm-gpu.yml
```

Key packages: `langchain>=0.3`, `langgraph>=0.2`, `langsmith>=0.1`, `faiss-cpu`, `chromadb`, `ragas`, `tiktoken`.

## Backbone strategy

Course 4 uses two backbone types:
- **Decoder** (default): `HuggingFaceTB/SmolLM2-135M-Instruct` — for all chain/agent/memory classes
- **Sentence encoder** (embedding): `sentence-transformers/all-MiniLM-L6-v2` — for all vector store / RAG classes

Both are declared in each class's `configs/default.yaml` under `backbone` and `embed_backbone`.
