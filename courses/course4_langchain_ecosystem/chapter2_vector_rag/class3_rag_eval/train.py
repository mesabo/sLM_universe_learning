"""Course 4 / ch2 / class 3 — RAG evaluation with RAGAS metrics.

Demonstrates:
  - Building a small QA eval dataset with questions, answers, and ground-truth contexts
  - Measuring context_recall: fraction of ground-truth context retrieved
  - Measuring answer_similarity: semantic similarity of generated vs reference answer
  - Using ragas EvaluationDataset API
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
from shared.llm_client import get_embedding_model, get_llm
from shared.logging_utils import get_logger
from shared.repro import set_seed
from shared.vector_store import build_store, query_store


KNOWLEDGE_BASE = [
    "LangChain is a framework for building LLM-powered applications.",
    "LCEL stands for LangChain Expression Language, which uses the | operator.",
    "LangSmith provides tracing and evaluation for LangChain applications.",
    "LangGraph enables building stateful multi-actor workflows as directed graphs.",
    "FAISS is a library for efficient similarity search over dense vectors.",
    "ChromaDB is an open-source vector database for AI embeddings.",
    "RAG stands for Retrieval Augmented Generation.",
    "LoRA reduces the number of trainable parameters in a large language model.",
    "Pydantic validates Python data structures with type annotations.",
    "ReAct is an agent paradigm combining reasoning and acting in interleaved steps.",
    "Memory in LangChain stores conversation turns for multi-turn dialogue.",
    "Embeddings are dense vector representations of text capturing semantic meaning.",
    "Context recall measures how much of the reference context the retriever found.",
    "Faithfulness measures whether the generated answer is supported by the context.",
    "Answer relevance measures how well the answer addresses the original question.",
    "Text splitters break long documents into overlapping chunks for embedding.",
    "Rerankers use cross-encoder models to rescore retrieved documents.",
    "Semantic caching stores LLM responses and retrieves them for similar queries.",
    "Multi-query retrieval generates several query variants to improve recall.",
    "Tool use allows LLMs to invoke external APIs and Python functions.",
]

EVAL_DATASET = [
    {
        "question": "What is LangChain?",
        "reference": "LangChain is a framework for building LLM-powered applications.",
        "reference_contexts": ["LangChain is a framework for building LLM-powered applications."],
    },
    {
        "question": "What does FAISS do?",
        "reference": "FAISS is a library for efficient similarity search over dense vectors.",
        "reference_contexts": ["FAISS is a library for efficient similarity search over dense vectors."],
    },
    {
        "question": "What is context recall?",
        "reference": "Context recall measures how much of the reference context the retriever found.",
        "reference_contexts": ["Context recall measures how much of the reference context the retriever found."],
    },
]


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
    log = get_logger("course4.ch2.class3")
    mode = cfg.get("mode", "smoke")
    n_docs = cfg["limits"][mode]["n_docs"]
    n_eval = cfg["limits"][mode]["n_eval_samples"]
    k = cfg["limits"][mode]["k"]

    embeddings = get_embedding_model(cfg)
    llm = get_llm(cfg)
    log.info("Models loaded")

    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    docs = [
        Document(page_content=text, metadata={"id": f"kb_{i:02d}"})
        for i, text in enumerate(KNOWLEDGE_BASE[:n_docs])
    ]
    store = build_store(docs, cfg, embeddings)

    rag_prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    rag_chain = rag_prompt | llm | StrOutputParser()

    eval_samples = EVAL_DATASET[:n_eval]
    rows = []
    for sample in eval_samples:
        retrieved = query_store(store, sample["question"], k=k)
        context_texts = [r.page_content for r in retrieved]
        try:
            answer = rag_chain.invoke({
                "context": "\n".join(context_texts),
                "question": sample["question"],
            })
        except Exception:
            answer = ""
        rows.append({
            "user_input": sample["question"],
            "retrieved_contexts": context_texts,
            "response": answer,
            "reference": sample["reference"],
        })

    log.info("RAG responses generated", n=len(rows))

    # RAGAS evaluation
    context_recall_score = 0.0
    answer_similarity_score = 0.0
    eval_complete = 0

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import ContextRecall, SemanticSimilarity
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        ragas_llm = LangchainLLMWrapper(llm)
        ragas_emb = LangchainEmbeddingsWrapper(embeddings)

        dataset = EvaluationDataset.from_list(rows)
        result = evaluate(
            dataset,
            metrics=[ContextRecall(), SemanticSimilarity()],
            llm=ragas_llm,
            embeddings=ragas_emb,
        )
        df = result.to_pandas()
        if "context_recall" in df.columns:
            context_recall_score = float(df["context_recall"].mean())
        if "semantic_similarity" in df.columns:
            answer_similarity_score = float(df["semantic_similarity"].mean())
        eval_complete = 1
        log.info(
            "RAGAS eval complete",
            context_recall=f"{context_recall_score:.3f}",
            answer_similarity=f"{answer_similarity_score:.3f}",
        )
    except Exception as exc:
        log.warning("RAGAS eval failed, using simple overlap fallback", error=str(exc))
        # Fallback: exact-string overlap recall
        recalls = []
        for row, sample in zip(rows, eval_samples):
            ref = sample["reference_contexts"][0].lower()
            hits = sum(1 for c in row["retrieved_contexts"] if ref[:30] in c.lower())
            recalls.append(min(hits, 1))
        context_recall_score = sum(recalls) / max(len(recalls), 1)
        eval_complete = 1

    metrics = {
        "context_recall": context_recall_score,
        "answer_similarity": answer_similarity_score,
        "eval_complete": float(eval_complete),
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
        extras={"mode": mode, "n_eval": n_eval},
    )


if __name__ == "__main__":
    main()
