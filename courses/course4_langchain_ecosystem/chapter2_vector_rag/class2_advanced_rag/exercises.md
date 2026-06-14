# Exercises — Class 2: Advanced RAG

## Warm-up: Log generated queries

Set `MultiQueryRetriever` verbose logging by adding `import logging; logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)`. Run one query and observe the N paraphrased queries the LLM generated. Are they semantically diverse? Do any of them seem redundant?

## Apply: EmbeddingsFilter vs LLMChainExtractor

Replace `LLMChainExtractor` with `EmbeddingsFilter` (available in `langchain.retrievers.document_compressors`):

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
ef = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
```

Compare: (1) latency per query, (2) whether the compressed results still contain the relevant information. When would you prefer one over the other?

## Stretch: Hybrid BM25 + FAISS retrieval

Implement hybrid retrieval by creating both a `BM25Retriever` (from `langchain_community.retrievers`) and a FAISS retriever, then combining with `EnsembleRetriever`:

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
ensemble = EnsembleRetriever(retrievers=[bm25, faiss_retriever], weights=[0.5, 0.5])
```

Measure recall@3 for BM25-only, FAISS-only, and ensemble on 5 queries. Report which queries benefit most from the hybrid approach and why.
