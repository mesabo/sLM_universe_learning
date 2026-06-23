"""Correction for Course 4 / ch2 / class 1 exercises.

This file reuses the corpus and embedding bootstrap style from ``train.py`` and
adds the concrete exercise variants that are not present in the baseline.

What is new relative to ``train.py``:
  1. Persisting and reloading a FAISS index from disk.
  2. Metadata-filtered Chroma retrieval.
  3. A recall-vs-k sweep for FAISS.

Sections marked ``NEW vs train.py`` highlight the corrections.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import apply_overrides, load_yaml
from shared.llm_client import get_embedding_model
from shared.logging_utils import get_logger
from shared.repro import set_seed
from shared.vector_store import build_store, query_store

from train import CORPUS, QUERIES, recall_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def make_documents(n_docs: int, with_category: bool = False):
    from langchain_core.documents import Document

    docs = []
    for index, (doc_id, text) in enumerate(CORPUS[:n_docs]):
        metadata = {"id": doc_id}
        if with_category:
            metadata["category"] = "retrieval" if index % 2 == 0 else "training"
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def run_faiss_save_reload(cfg: dict, embeddings, log) -> dict:
    """Exercise 1 solution for FAISS persistence."""
    cfg_faiss = dict(cfg)
    cfg_faiss["vector_store"] = {"backend": "faiss"}
    docs = make_documents(cfg["limits"][cfg.get("mode", "smoke")]["n_docs"])
    store = build_store(docs, cfg_faiss, embeddings)
    index_dir = Path(__file__).with_name("faiss_correction_index")
    if index_dir.exists():
        shutil.rmtree(index_dir)

    start = time.perf_counter()
    store.save_local(str(index_dir))
    save_time = time.perf_counter() - start

    from langchain_community.vectorstores import FAISS

    start = time.perf_counter()
    reloaded = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    load_time = time.perf_counter() - start

    query = QUERIES[0][0]
    original_ids = [d.metadata.get("id") for d in query_store(store, query, k=3)]
    reloaded_ids = [d.metadata.get("id") for d in query_store(reloaded, query, k=3)]
    log.info("FAISS save/reload", save_s=save_time, load_s=load_time)
    return {
        "save_time_s": round(save_time, 4),
        "load_time_s": round(load_time, 4),
        "same_results": original_ids == reloaded_ids,
        "original_ids": original_ids,
        "reloaded_ids": reloaded_ids,
    }


def run_chroma_filter_demo(cfg: dict, embeddings, k: int) -> dict:
    """Exercise 2 solution for metadata filtering."""
    docs = make_documents(cfg["limits"][cfg.get("mode", "smoke")]["n_docs"], with_category=True)
    cfg_chroma = dict(cfg)
    cfg_chroma["vector_store"] = {"backend": "chromadb", "collection": "course4_correction"}
    store = build_store(docs, cfg_chroma, embeddings)
    query = "What is retrieval augmented generation?"
    full_ids = [d.metadata.get("id") for d in store.similarity_search(query, k=k)]
    filtered_ids = [
        d.metadata.get("id")
        for d in store.similarity_search(query, k=k, filter={"category": "retrieval"})
    ]
    relevant = ["ml_007", "ml_019", "ml_020"]
    return {
        "full_ids": full_ids,
        "filtered_ids": filtered_ids,
        "full_recall_at_3": round(recall_at_k(full_ids, relevant), 4),
        "filtered_recall_at_3": round(recall_at_k(filtered_ids, relevant), 4),
    }


def run_recall_curve(cfg: dict, embeddings) -> dict:
    """Exercise 3 solution for recall@k sweep."""
    cfg_faiss = dict(cfg)
    cfg_faiss["vector_store"] = {"backend": "faiss"}
    docs = make_documents(cfg["limits"][cfg.get("mode", "smoke")]["n_docs"])
    store = build_store(docs, cfg_faiss, embeddings)
    curve = {}
    for k in [1, 3, 5, 10]:
        recalls = []
        for query, relevant in QUERIES:
            ids = [d.metadata.get("id") for d in query_store(store, query, k=k)]
            recalls.append(recall_at_k(ids, relevant))
        curve[k] = round(sum(recalls) / max(len(recalls), 1), 4)
    return curve


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch2.class1.correction")
    mode = cfg.get("mode", "smoke")
    k = cfg["limits"][mode]["k"]

    embeddings = get_embedding_model(cfg)
    log.info("Embeddings loaded", backbone=cfg.get("embed_backbone", "?"))

    print("\n=== Exercise 1: Save and reload a FAISS index ===")
    print(
        "NEW vs train.py: the FAISS store is persisted to disk and reloaded, "
        "so retrieval behavior can be checked across process boundaries."
    )
    print("Expected output tip: `same_results` should be True and the original/reloaded ID lists should match.")
    faiss_stats = run_faiss_save_reload(cfg, embeddings, log)
    print("Exercise 1 - save_time_s:", faiss_stats["save_time_s"])
    print("Exercise 1 - load_time_s:", faiss_stats["load_time_s"])
    print("Exercise 1 - same_results:", faiss_stats["same_results"])
    print("Exercise 1 - original_ids:", faiss_stats["original_ids"])
    print("Exercise 1 - reloaded_ids:", faiss_stats["reloaded_ids"])

    print("\n=== Exercise 2: Metadata filtering in ChromaDB ===")
    print(
        "NEW vs train.py: documents now carry a `category` metadata field and "
        "Chroma retrieval is filtered on that field."
    )
    print("Expected output tip: filtered results should only come from `category='retrieval'`, even if recall changes.")
    chroma_stats = run_chroma_filter_demo(cfg, embeddings, k=3)
    print("Exercise 2 - full_ids:", chroma_stats["full_ids"])
    print("Exercise 2 - filtered_ids:", chroma_stats["filtered_ids"])
    print("Exercise 2 - full_recall_at_3:", chroma_stats["full_recall_at_3"])
    print("Exercise 2 - filtered_recall_at_3:", chroma_stats["filtered_recall_at_3"])

    print("\n=== Exercise 3: Recall vs k curve ===")
    print(
        "NEW vs train.py: instead of a single recall@k point, recall is swept "
        "across multiple k values to reveal where performance plateaus."
    )
    print("Expected output tip: recall should stay flat or improve as k increases; it should not decrease.")
    recall_curve = run_recall_curve(cfg, embeddings)
    print("Exercise 3 - recall curve:", recall_curve)


if __name__ == "__main__":
    main()
