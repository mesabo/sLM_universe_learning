"""Vector store factory for Course 4.

Builds a LangChain ``VectorStore`` from a list of ``Document`` objects using
either FAISS (in-memory, fast) or ChromaDB (persistent, queryable).

Usage::

    from langchain_core.documents import Document
    from shared.vector_store import build_store, query_store

    docs = [Document(page_content="hello", metadata={"id": "1"})]
    store = build_store(docs, cfg, embeddings)
    results = query_store(store, "hello", k=3)
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def build_store(
    docs: list[Document],
    cfg: dict[str, Any],
    embeddings: Embeddings,
) -> VectorStore:
    """Build a FAISS or ChromaDB vector store.

    Args:
        docs:       Documents to index.
        cfg:        Class config dict; reads ``cfg["vector_store"]["backend"]``.
        embeddings: LangChain Embeddings instance.

    Returns:
        A ``VectorStore`` instance.
    """
    backend = cfg.get("vector_store", {}).get("backend", "faiss")

    if backend == "chromadb":
        from langchain_community.vectorstores import Chroma

        collection = cfg.get("vector_store", {}).get("collection", "course4")
        return Chroma.from_documents(
            docs, embeddings, collection_name=collection
        )

    # Default: FAISS
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(docs, embeddings)


def query_store(
    store: VectorStore,
    query: str,
    k: int = 4,
    search_type: str = "similarity",
) -> list[Document]:
    """Retrieve top-k documents from ``store``.

    Args:
        store:       Built vector store.
        query:       Query string.
        k:           Number of results.
        search_type: ``"similarity"`` or ``"mmr"`` (max marginal relevance).

    Returns:
        List of ``Document`` objects.
    """
    if search_type == "mmr":
        return store.max_marginal_relevance_search(query, k=k)
    return store.similarity_search(query, k=k)
