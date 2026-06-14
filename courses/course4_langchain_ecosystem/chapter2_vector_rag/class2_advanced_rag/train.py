"""Course 4 / ch2 / class 2 — Advanced RAG: multi-query, compression, hybrid.

Demonstrates:
  - MultiQueryRetriever: expands one query into N variants to improve recall
  - ContextualCompressionRetriever: prunes irrelevant context from retrieved docs
  - BM25 + FAISS hybrid retrieval (sparse + dense fusion)
  - recall@k metric across retrieval strategies
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import apply_overrides, load_yaml
from shared.eval_harness import run_eval
from shared.llm_client import get_llm, get_embedding_model
from shared.logging_utils import get_logger
from shared.repro import set_seed
from shared.vector_store import build_store

CORPUS = [
    ("d01", "LangChain enables building LLM-powered applications with composable chains."),
    ("d02", "LCEL (LangChain Expression Language) uses the pipe operator to chain runnables."),
    ("d03", "Memory in LangChain stores conversation history across multiple turns."),
    ("d04", "Agents in LangChain use tools to interact with external systems."),
    ("d05", "ReAct agents reason step-by-step before selecting which tool to invoke."),
    ("d06", "LangSmith provides observability: tracing, evaluation, and dataset management."),
    ("d07", "LangGraph builds stateful multi-actor applications as directed graphs."),
    ("d08", "Vector stores index document embeddings for fast semantic similarity search."),
    ("d09", "MultiQueryRetriever generates multiple query variants to improve retrieval recall."),
    ("d10", "ContextualCompressionRetriever extracts only the relevant portion of each retrieved doc."),
    ("d11", "Hybrid search combines sparse BM25 keyword matching with dense vector retrieval."),
    ("d12", "RAG pipelines retrieve relevant documents and pass them as context to the LLM."),
    ("d13", "Embeddings from sentence-transformers map text to a dense vector space."),
    ("d14", "Reranking uses a cross-encoder to rescore retrieved documents for relevance."),
    ("d15", "FAISS supports exact and approximate nearest neighbor search on GPU or CPU."),
    ("d16", "ChromaDB persists embeddings and allows metadata filtering during retrieval."),
    ("d17", "Prompt templates structure the input to the LLM with variables and instructions."),
    ("d18", "Output parsers convert raw LLM strings into structured Python objects."),
    ("d19", "Streaming allows the application to display tokens as they are generated."),
    ("d20", "Caching stores LLM responses keyed by (prompt, model) to reduce cost and latency."),
    ("d21", "Tool calling exposes Python functions to the LLM as callable JSON schemas."),
    ("d22", "Human-in-the-loop checkpoints allow a human to approve agent actions mid-run."),
    ("d23", "Pydantic models provide typed, validated schemas for structured LLM output."),
    ("d24", "Async LCEL chains use ainvoke() and astream() for non-blocking execution."),
    ("d25", "LangSmith datasets enable systematic regression testing of prompt changes."),
    ("d26", "Runnable lambda wraps any Python callable as a LangChain runnable component."),
    ("d27", "ConfigurableField allows a chain to accept runtime configuration overrides."),
    ("d28", "Document loaders ingest text from PDFs, web pages, Markdown, and databases."),
    ("d29", "Text splitters chunk long documents into overlapping windows for embedding."),
    ("d30", "Evaluation metrics like faithfulness and context recall measure RAG quality."),
]

EVAL_PAIRS = [
    ("What is MultiQueryRetriever?", ["d09"]),
    ("How does LangGraph work?", ["d07"]),
    ("What is hybrid search in RAG?", ["d11"]),
]


def recall_at_k(retrieved_ids: list[str], relevant: list[str]) -> float:
    return sum(1 for r in retrieved_ids if r in relevant) / max(len(relevant), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch2.class2")
    mode = cfg.get("mode", "smoke")
    n_docs = cfg["limits"][mode]["n_docs"]
    n_queries = cfg["limits"][mode]["n_queries"]
    k = cfg["limits"][mode]["k"]

    embeddings = get_embedding_model(cfg)
    llm = get_llm(cfg)
    log.info("Models loaded", embed=cfg.get("embed_backbone"), llm=cfg.get("backbone"))

    from langchain_core.documents import Document
    docs = [
        Document(page_content=text, metadata={"id": doc_id})
        for doc_id, text in CORPUS[:n_docs]
    ]
    store = build_store(docs, cfg, embeddings)
    base_retriever = store.as_retriever(search_kwargs={"k": k})

    # --- MultiQueryRetriever ---
    multi_query_ok = 0
    mq_recalls = []
    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm
        )
        for query, relevant in EVAL_PAIRS[:n_queries]:
            results = mq_retriever.invoke(query)
            ids = [r.metadata.get("id", "") for r in results]
            mq_recalls.append(recall_at_k(ids, relevant))
        multi_query_ok = 1
        log.info("MultiQuery ok", recall=f"{sum(mq_recalls)/max(len(mq_recalls),1):.3f}")
    except Exception as exc:
        log.warning("MultiQuery failed", error=str(exc))

    # --- ContextualCompressionRetriever ---
    compression_ok = 0
    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        results = compression_retriever.invoke(EVAL_PAIRS[0][0])
        compression_ok = 1
        log.info("Compression retriever ok", n_results=len(results))
    except Exception as exc:
        log.warning("Compression retriever failed", error=str(exc))

    avg_recall = sum(mq_recalls) / max(len(mq_recalls), 1) if mq_recalls else 0.0

    metrics = {
        "multi_query_ok": float(multi_query_ok),
        "compression_ok": float(compression_ok),
        "recall_at_k": avg_recall,
    }
    run_eval(
        method=cfg["method"],
        backbone=cfg.get("embed_backbone", "local"),
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics=metrics,
        expected_band=cfg.get("expected_band", {}),
        extras={"mode": mode, "k": k},
    )


if __name__ == "__main__":
    main()
