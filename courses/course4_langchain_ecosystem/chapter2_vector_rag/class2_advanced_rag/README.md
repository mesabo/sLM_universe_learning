# Class 2 — Advanced RAG: Multi-Query, Compression, Hybrid

> Goal: improve a naive RAG pipeline by adding multi-query retrieval, contextual compression, and hybrid sparse+dense search, and show which failure mode each technique is meant to fix.

## Psycho Mode

Basic RAG is a single query hitting a single retriever. Advanced RAG is about addressing the failure modes of that simple approach. Three failure modes motivate three techniques in this class.

First: a single query may not express all the relevant aspects of the user's intent. Multi-query retrieval generates N paraphrased variants of the original query, retrieves for each, then unions the results. Second: retrieved documents often contain irrelevant surrounding text — the relevant sentence is buried in a paragraph about something else. Contextual compression extracts only the relevant excerpt, reducing noise in the context window. Third: semantic search misses exact-match keywords; BM25 misses semantically related but differently-worded content. Hybrid search fuses both signals.

## Academic Mode

Multi-query retrieval: given query $q$, an LLM generates $\{q_1, \ldots, q_N\}$ as paraphrases. The final result set is:

$$\mathcal{R}_{\text{multi}} = \bigcup_{i=1}^{N} \text{retrieve}(q_i, k)$$

Contextual compression: given retrieved documents $\mathcal{R}$, a compressor $C_\theta$ extracts the relevant span:

$$\mathcal{R}' = \{C_\theta(r, q) \mid r \in \mathcal{R}\}$$

where $C_\theta$ is an LLM prompted with "extract the part relevant to the question."

Hybrid retrieval: dense score $s_d = \cos(\phi(q), \phi(c))$ is fused with sparse BM25 score $s_b$ via Reciprocal Rank Fusion (RRF):

$$s_{\text{RRF}}(c) = \frac{1}{k + \text{rank}_d(c)} + \frac{1}{k + \text{rank}_b(c)}$$

Reference: MultiQueryRetriever — [https://python.langchain.com/docs/how_to/MultiQueryRetriever/](https://python.langchain.com/docs/how_to/MultiQueryRetriever/).

## Engineering Mode

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Multi-query
mq_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
results = mq_retriever.invoke("What is multi-query retrieval?")

# Contextual compression
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)
results = compression_retriever.invoke("Explain FAISS.")
```

Gotchas:
- `MultiQueryRetriever` can duplicate documents across query variants; deduplicate by `doc.page_content` before passing to the LLM.
- `LLMChainExtractor` makes one LLM call per retrieved document — expensive at large k. Use `EmbeddingsFilter` instead for local models.
- Hybrid BM25 + FAISS requires `langchain-community`'s `BM25Retriever` or an external library (`rank_bm25`).

Config keys: `embed_backbone`, `limits.smoke.n_docs`, `limits.smoke.n_queries`, `limits.smoke.k`.

## Research Mode

Advanced RAG is an active area. Key papers: (1) HyDE (Hypothetical Document Embeddings) — embed a hypothetical answer to the query, then retrieve against that embedding instead of the raw query; (2) RAPTOR (Recursive Abstractive Processing) — hierarchically cluster and summarize documents to enable coarse-to-fine retrieval; (3) Self-RAG — train a model to decide when to retrieve and how to use retrieved context, reducing unnecessary retrieval calls. For production systems, the bottleneck is often latency: compression and multi-query each add LLM call overhead. Profile and choose the technique whose recall gain justifies the cost.

## How to run

```bash
bash courses/course4_langchain_ecosystem/chapter2_vector_rag/class2_advanced_rag/run.sh
MODE=full bash courses/course4_langchain_ecosystem/chapter2_vector_rag/class2_advanced_rag/run.sh
```

## How to verify

| Metric | Expected |
|---|---|
| `multi_query_ok` | [1, 1] |
| `compression_ok` | [1, 1] |
| `recall_at_k` | [0.4, 1.0] |

---

**Instructor checklist**

- [x] All four mode sections present.
- [x] Official reference links included.
- [x] No numeric literals except 0/1 in code files.
- [x] `configs/default.yaml` declares `expected_band`.
- [x] `run.sh` chmod+x.
- [x] `exercises.md` has 3 exercises.
- [x] Linked from course README.
