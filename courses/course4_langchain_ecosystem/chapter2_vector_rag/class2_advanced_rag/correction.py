"""Correction for Course 4 / ch2 / class 2 exercises.

This file follows the same model/bootstrap style as ``train.py`` and focuses on
the three exercise extensions.

What is new relative to ``train.py``:
  1. Verbose logging for generated multi-query paraphrases.
  2. A comparison between ``EmbeddingsFilter`` and ``LLMChainExtractor``.
  3. Hybrid BM25 + FAISS retrieval with ``EnsembleRetriever``.

Sections marked ``NEW vs train.py`` highlight the corrections.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import apply_overrides, load_yaml
from shared.llm_client import get_embedding_model, get_llm
from shared.logging_utils import get_logger
from shared.repro import set_seed
from shared.vector_store import build_store

from train import CORPUS, EVAL_PAIRS, recall_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_docs(n_docs: int):
    from langchain_core.documents import Document

    return [
        Document(page_content=text, metadata={"id": doc_id})
        for doc_id, text in CORPUS[:n_docs]
    ]


def run_multiquery_logging(llm, base_retriever, query: str) -> list[str]:
    """Exercise 1 solution.

    NEW vs train.py:
    the retriever is still the same, but verbose logger output is enabled so
    the generated paraphrases become visible and inspectable.
    """
    from langchain.retrievers.multi_query import MultiQueryRetriever

    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    results = retriever.invoke(query)
    return [doc.metadata.get("id", "") for doc in results]


def compare_compressors(llm, embeddings, base_retriever, query: str) -> dict:
    """Exercise 2 solution comparing two compression strategies."""
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import (
        EmbeddingsFilter,
        LLMChainExtractor,
    )

    start = time.perf_counter()
    llm_compressor = LLMChainExtractor.from_llm(llm)
    llm_retriever = ContextualCompressionRetriever(
        base_compressor=llm_compressor,
        base_retriever=base_retriever,
    )
    llm_docs = llm_retriever.invoke(query)
    llm_time = time.perf_counter() - start

    start = time.perf_counter()
    emb_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
    emb_retriever = ContextualCompressionRetriever(
        base_compressor=emb_filter,
        base_retriever=base_retriever,
    )
    emb_docs = emb_retriever.invoke(query)
    emb_time = time.perf_counter() - start

    return {
        "llm_time_s": round(llm_time, 4),
        "embeddings_time_s": round(emb_time, 4),
        "llm_docs": [d.page_content[:80] for d in llm_docs],
        "embeddings_docs": [d.page_content[:80] for d in emb_docs],
    }


def run_hybrid_retrieval(cfg: dict, embeddings, docs, k: int) -> dict:
    """Exercise 3 solution for hybrid BM25 + FAISS retrieval."""
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    store = build_store(docs, cfg, embeddings)
    faiss_retriever = store.as_retriever(search_kwargs={"k": k})
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    ensemble = EnsembleRetriever(retrievers=[bm25, faiss_retriever], weights=[0.5, 0.5])

    report = {}
    for query, relevant in EVAL_PAIRS:
        faiss_ids = [d.metadata.get("id", "") for d in faiss_retriever.invoke(query)]
        bm25_ids = [d.metadata.get("id", "") for d in bm25.invoke(query)]
        ensemble_ids = [d.metadata.get("id", "") for d in ensemble.invoke(query)]
        report[query] = {
            "bm25_recall_at_3": round(recall_at_k(bm25_ids, relevant), 4),
            "faiss_recall_at_3": round(recall_at_k(faiss_ids, relevant), 4),
            "ensemble_recall_at_3": round(recall_at_k(ensemble_ids, relevant), 4),
            "ensemble_ids": ensemble_ids,
        }
    return report


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch2.class2.correction")
    mode = cfg.get("mode", "smoke")
    n_docs = cfg["limits"][mode]["n_docs"]
    k = cfg["limits"][mode]["k"]

    embeddings = get_embedding_model(cfg)
    llm = get_llm(cfg)
    docs = build_docs(n_docs)
    store = build_store(docs, cfg, embeddings)
    base_retriever = store.as_retriever(search_kwargs={"k": k})
    log.info("Models loaded", embed=cfg.get("embed_backbone"), llm=cfg.get("backbone"))

    print("\n=== Exercise 1: Log generated queries ===")
    print(
        "NEW vs train.py: MultiQueryRetriever is instrumented with verbose "
        "logging so the paraphrased queries can be inspected directly."
    )
    print("Expected output tip: logs should show several paraphrased queries, some useful and some possibly redundant.")
    mq_ids = run_multiquery_logging(llm, base_retriever, EVAL_PAIRS[0][0])
    print("Exercise 1 - retrieved ids:", mq_ids)

    print("\n=== Exercise 2: EmbeddingsFilter vs LLMChainExtractor ===")
    print(
        "NEW vs train.py: compression is no longer a single-path demo; two "
        "compressors are compared on latency and retained content."
    )
    print("Expected output tip: `EmbeddingsFilter` is often faster, while `LLMChainExtractor` may keep more targeted text.")
    compressor_report = compare_compressors(llm, embeddings, base_retriever, EVAL_PAIRS[0][0])
    print("Exercise 2 - LLMChainExtractor time:", compressor_report["llm_time_s"])
    print("Exercise 2 - EmbeddingsFilter time:", compressor_report["embeddings_time_s"])
    print("Exercise 2 - LLM docs:", compressor_report["llm_docs"])
    print("Exercise 2 - Embeddings docs:", compressor_report["embeddings_docs"])

    print("\n=== Exercise 3: Hybrid BM25 + FAISS retrieval ===")
    print(
        "NEW vs train.py: sparse and dense retrievers are fused with "
        "`EnsembleRetriever`, then compared against the single-retriever paths."
    )
    print("Expected output tip: ensemble recall should be at least competitive with the better single retriever on most queries.")
    hybrid_report = run_hybrid_retrieval(cfg, embeddings, docs, k=3)
    print("Exercise 3 - hybrid report:", hybrid_report)


if __name__ == "__main__":
    main()
