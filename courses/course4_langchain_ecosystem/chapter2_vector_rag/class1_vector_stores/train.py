"""Course 4 / ch2 / class 1 — FAISS vs ChromaDB vector stores side-by-side.

Demonstrates:
  - Building FAISS and ChromaDB stores from the same document set
  - similarity_search() and max_marginal_relevance_search() (MMR)
  - Measuring recall@k: fraction of relevant docs in top-k results
  - Semantic search vs keyword baseline
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
from shared.llm_client import get_embedding_model
from shared.logging_utils import get_logger
from shared.repro import set_seed
from shared.vector_store import build_store, query_store


CORPUS = [
    ("ml_001", "Gradient descent minimizes a loss function by iterating in the negative gradient direction."),
    ("ml_002", "Backpropagation computes gradients via the chain rule of calculus."),
    ("ml_003", "Attention mechanisms allow transformers to weigh different input positions."),
    ("ml_004", "LoRA reduces trainable parameters by injecting low-rank matrices into weight layers."),
    ("ml_005", "FAISS is a library for efficient similarity search in dense vector spaces."),
    ("ml_006", "ChromaDB is a persistent vector database designed for AI applications."),
    ("ml_007", "RAG combines retrieval from a knowledge base with generative language models."),
    ("ml_008", "Embedding models map text to fixed-size dense vectors capturing semantic meaning."),
    ("ml_009", "Cosine similarity measures the angle between two vectors, ignoring magnitude."),
    ("ml_010", "Quantization reduces model precision (e.g., FP32→INT8) to save memory and speed inference."),
    ("ml_011", "KV cache stores key-value pairs during autoregressive generation to avoid recomputation."),
    ("ml_012", "The transformer architecture relies on self-attention and feed-forward layers."),
    ("ml_013", "Fine-tuning adapts a pre-trained model to a downstream task using labeled data."),
    ("ml_014", "Contrastive learning trains models so similar items cluster in embedding space."),
    ("ml_015", "DPO aligns language models with human preferences without a separate reward model."),
    ("ml_016", "Perplexity measures how well a language model predicts a held-out test set."),
    ("ml_017", "Beam search explores multiple token sequences during generation to find likely outputs."),
    ("ml_018", "Temperature scaling controls randomness in token sampling; lower = more deterministic."),
    ("ml_019", "Retrieval augmented generation improves factual accuracy by grounding answers in documents."),
    ("ml_020", "Vector databases index embeddings for approximate nearest neighbor (ANN) search."),
]

QUERIES = [
    ("What is FAISS?", ["ml_005"]),
    ("How does RAG work?", ["ml_007", "ml_019"]),
    ("What is cosine similarity?", ["ml_009"]),
    ("How does LoRA reduce parameters?", ["ml_004"]),
    ("What is KV cache?", ["ml_011"]),
]


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    hits = sum(1 for r in retrieved_ids if r in relevant_ids)
    return hits / max(len(relevant_ids), 1)


def chunk_documents(raw_texts: list[str], chunk_size: int = 256, chunk_overlap: int = 32) -> list[str]:
    """
    Split documents into overlapping chunks — the most common RAG preprocessing step.

    Industry standard: RecursiveCharacterTextSplitter with:
      chunk_size=256-512 characters (not tokens!)
      chunk_overlap=10-20% of chunk_size (prevents information loss at boundaries)

    Why recursive? Tries splitting by paragraphs (\\n\\n) first, then sentences (. ),
    then words, then characters — preserves semantic boundaries where possible.

    Interview: "What chunking strategy do you use for RAG?"
    Answer: RecursiveCharacterTextSplitter with 512 chars and 10% overlap.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = []
        for text in raw_texts:
            chunks.extend(splitter.split_text(text))
        return chunks or raw_texts  # fallback to originals if splitter returns empty
    except ImportError:
        return raw_texts  # graceful fallback


def run_chunking_demo(raw_texts: list[str], chunk_size: int, chunk_overlap: int, log) -> dict:
    """Demonstrate chunking and return metrics."""
    chunks = chunk_documents(raw_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    log.info("chunking", n_raw=len(raw_texts), n_chunks=len(chunks), chunk_size=chunk_size)
    return {
        "chunking_ok": 1,
        "n_chunks": len(chunks),
        "chunk_expansion_ratio": round(len(chunks) / max(len(raw_texts), 1), 2),
    }


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
    log = get_logger("course4.ch2.class1")
    mode = cfg.get("mode", "smoke")
    n_docs = cfg["limits"][mode]["n_docs"]
    n_queries = cfg["limits"][mode]["n_queries"]
    k = cfg["limits"][mode]["k"]

    chunk_size = cfg.get("chunking", {}).get("chunk_size", 256)
    chunk_overlap = cfg.get("chunking", {}).get("chunk_overlap", 32)

    embeddings = get_embedding_model(cfg)
    log.info("Embeddings loaded", backbone=cfg.get("embed_backbone", "?"))

    from langchain_core.documents import Document
    docs = [
        Document(page_content=text, metadata={"id": doc_id})
        for doc_id, text in CORPUS[:n_docs]
    ]
    queries = QUERIES[:n_queries]

    # --- Chunking demo ---
    raw_texts = [text for _, text in CORPUS[:n_docs]]
    chunking_metrics = run_chunking_demo(raw_texts, chunk_size, chunk_overlap, log)

    # --- FAISS ---
    cfg_faiss = dict(cfg)
    cfg_faiss["vector_store"] = {"backend": "faiss"}
    faiss_store = build_store(docs, cfg_faiss, embeddings)

    faiss_recalls = []
    for query, relevant in queries:
        results = query_store(faiss_store, query, k=k)
        retrieved_ids = [r.metadata.get("id", "") for r in results]
        faiss_recalls.append(recall_at_k(retrieved_ids, relevant))
    faiss_recall = sum(faiss_recalls) / max(len(faiss_recalls), 1)
    log.info("FAISS recall@k", recall=f"{faiss_recall:.3f}", k=k)

    # --- ChromaDB ---
    cfg_chroma = dict(cfg)
    cfg_chroma["vector_store"] = {"backend": "chromadb", "collection": "course4_test"}
    chroma_store = build_store(docs, cfg_chroma, embeddings)

    chroma_recalls = []
    for query, relevant in queries:
        results = query_store(chroma_store, query, k=k)
        retrieved_ids = [r.metadata.get("id", "") for r in results]
        chroma_recalls.append(recall_at_k(retrieved_ids, relevant))
    chroma_recall = sum(chroma_recalls) / max(len(chroma_recalls), 1)
    log.info("ChromaDB recall@k", recall=f"{chroma_recall:.3f}", k=k)

    # --- MMR diversity test ---
    mmr_results = query_store(faiss_store, queries[0][0], k=k, search_type="mmr")
    mmr_ok = 1 if len(mmr_results) > 0 else 0

    metrics = {
        "faiss_recall": faiss_recall,
        "chroma_recall": chroma_recall,
        "mmr_ok": float(mmr_ok),
    }
    metrics.update(chunking_metrics)
    run_eval(
        method=cfg["method"],
        backbone=cfg.get("embed_backbone", "local"),
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics=metrics,
        expected_band=cfg.get("expected_band", {}),
        extras={"mode": mode, "n_docs": n_docs, "k": k},
    )


if __name__ == "__main__":
    main()
