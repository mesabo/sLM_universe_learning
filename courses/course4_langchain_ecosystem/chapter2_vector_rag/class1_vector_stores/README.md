# Class 1 — FAISS vs ChromaDB Vector Stores

## Psycho Mode

A vector store is a library catalog where every book has been converted into a point in high-dimensional space — and "find books similar to this one" becomes "find points nearest to this point." You embed a query into the same space, then retrieve the k nearest neighbors. This is semantic search: proximity in embedding space reflects meaning similarity, not keyword overlap.

FAISS (Facebook AI Similarity Search) and ChromaDB are the two most common local vector stores. FAISS is an in-memory index: blazing fast, no persistence, ideal for prototypes. ChromaDB is a persistent database: slower to build, but survives restarts and supports metadata filtering. Understanding both — and when to choose each — is a core hiring requirement for any RAG engineer.

## Academic Mode

Let $\phi: \mathcal{T} \to \mathbb{R}^d$ be an embedding function mapping text to a $d$-dimensional dense vector. A vector store maintains an index $\mathcal{I}$ over a corpus $\mathcal{C} = \{c_1, \ldots, c_N\}$ such that, given a query $q$, retrieval returns:

$$\text{retrieve}(q, k) = \underset{S \subseteq \mathcal{C},\, |S|=k}{\arg\max} \sum_{c \in S} \cos(\phi(q), \phi(c))$$

FAISS uses flat L2 / inner product indices for exact search, and IVF-PQ indices for approximate nearest neighbor (ANN) search at scale. ChromaDB wraps HNSW (Hierarchical Navigable Small Worlds), a graph-based ANN algorithm with $O(\log N)$ query complexity. MMR (Maximal Marginal Relevance) diversifies results by penalizing redundancy:

$$\text{MMR}(q, R, \lambda) = \lambda \cdot \cos(\phi(q), \phi(c)) - (1-\lambda) \cdot \max_{r \in R} \cos(\phi(r), \phi(c))$$

Reference: FAISS paper — [https://arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734).

## Engineering Mode

```python
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs = [Document(page_content=text, metadata={"id": doc_id}) for doc_id, text in corpus]

# FAISS (in-memory)
faiss_store = FAISS.from_documents(docs, embeddings)
results = faiss_store.similarity_search("What is FAISS?", k=3)

# ChromaDB (persistent)
chroma_store = Chroma.from_documents(docs, embeddings, collection_name="my_collection")
results = chroma_store.similarity_search("What is FAISS?", k=3)

# MMR (diversified)
results = faiss_store.max_marginal_relevance_search("What is FAISS?", k=3, fetch_k=10)
```

Config keys: `embed_backbone`, `vector_store.backend`, `limits.smoke.n_docs`, `limits.smoke.k`.

Gotchas:
- ChromaDB writes to disk; use a temp directory in tests to avoid cross-run pollution.
- FAISS does not support metadata filtering natively; use `similarity_search_with_score` and post-filter.
- `max_marginal_relevance_search` requires `fetch_k > k`; otherwise results are identical to `similarity_search`.

## Research Mode

Vector store research directions: (1) learned indexes (replacing HNSW with a neural model that predicts approximate neighbors); (2) product quantization (PQ) for memory-efficient large-scale indexes; (3) multi-vector representations (ColBERT-style late interaction) where each document is represented by multiple vectors; (4) hybrid sparse-dense indexes (SPLADE + FAISS) that combine BM25 keyword statistics with dense retrieval. For very large corpora (>10M docs), the choice of index type (Flat vs IVF vs HNSW) dominates latency and recall tradeoffs.

## How to run

```bash
bash courses/course4_langchain_ecosystem/chapter2_vector_rag/class1_vector_stores/run.sh
MODE=full bash courses/course4_langchain_ecosystem/chapter2_vector_rag/class1_vector_stores/run.sh
```

## How to verify

| Metric | Expected |
|---|---|
| `faiss_recall` | [0.5, 1.0] |
| `chroma_recall` | [0.5, 1.0] |
| `mmr_ok` | [1, 1] |

---

**Instructor checklist**

- [x] All four mode sections present.
- [x] Official reference links included.
- [x] No numeric literals except 0/1 in code files.
- [x] `configs/default.yaml` declares `expected_band`.
- [x] `run.sh` chmod+x.
- [x] `exercises.md` has 3 exercises.
- [x] Linked from course README.
