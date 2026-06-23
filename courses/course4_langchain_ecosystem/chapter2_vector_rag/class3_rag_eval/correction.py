"""Correction for Course 4 / ch2 / class 3 exercises.

This file keeps the same small-RAG setup as ``train.py`` and extends it with
the three evaluation variants requested by the exercises.

What is new relative to ``train.py``:
  1. ``Faithfulness()`` is added beside the baseline metrics.
  2. An adversarial eval set is introduced.
  3. The evaluation is swept across multiple ``k`` values.

Sections marked ``NEW vs train.py`` highlight the corrections.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import apply_overrides, load_yaml
from shared.llm_client import get_embedding_model, get_llm
from shared.logging_utils import get_logger
from shared.repro import set_seed
from shared.vector_store import build_store, query_store

from train import EVAL_DATASET, KNOWLEDGE_BASE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_rag_rows(llm, store, eval_samples: list[dict], k: int) -> list[dict]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    chain = prompt | llm | StrOutputParser()
    rows = []
    for sample in eval_samples:
        retrieved = query_store(store, sample["question"], k=k)
        contexts = [doc.page_content for doc in retrieved]
        try:
            response = chain.invoke({"context": "\n".join(contexts), "question": sample["question"]})
        except Exception:
            response = ""
        rows.append(
            {
                "user_input": sample["question"],
                "retrieved_contexts": contexts,
                "response": response,
                "reference": sample["reference"],
            }
        )
    return rows


def evaluate_with_ragas(rows: list[dict], llm, embeddings) -> dict:
    """Exercise 1 solution with Faithfulness added."""
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import ContextRecall, Faithfulness, SemanticSimilarity

    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[ContextRecall(), SemanticSimilarity(), Faithfulness()],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )
    df = result.to_pandas()
    return {
        "context_recall": float(df["context_recall"].mean()) if "context_recall" in df else 0.0,
        "semantic_similarity": float(df["semantic_similarity"].mean()) if "semantic_similarity" in df else 0.0,
        "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0,
    }


def simple_context_recall(rows: list[dict], eval_samples: list[dict]) -> float:
    recalls = []
    for row, sample in zip(rows, eval_samples):
        references = [ctx.lower() for ctx in sample["reference_contexts"]]
        retrieved = [ctx.lower() for ctx in row["retrieved_contexts"]]
        hit = 0
        for ref in references:
            if any(ref[:30] in item for item in retrieved):
                hit = 1
                break
        recalls.append(hit)
    return sum(recalls) / max(len(recalls), 1)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch2.class3.correction")
    mode = cfg.get("mode", "smoke")
    n_docs = cfg["limits"][mode]["n_docs"]

    embeddings = get_embedding_model(cfg)
    llm = get_llm(cfg)
    log.info("Models loaded", embed=cfg.get("embed_backbone"), llm=cfg.get("backbone"))

    from langchain_core.documents import Document

    docs = [
        Document(page_content=text, metadata={"id": f"kb_{i:02d}"})
        for i, text in enumerate(KNOWLEDGE_BASE[:n_docs])
    ]
    store = build_store(docs, cfg, embeddings)

    print("\n=== Exercise 1: Add faithfulness metric ===")
    print(
        "NEW vs train.py: RAGAS evaluation now includes `Faithfulness()` in "
        "addition to the baseline context-recall and semantic-similarity metrics."
    )
    print("Expected output tip: the report should now contain `context_recall`, `semantic_similarity`, and `faithfulness`.")
    base_rows = build_rag_rows(llm, store, EVAL_DATASET[:3], k=3)
    try:
        metric_report = evaluate_with_ragas(base_rows, llm, embeddings)
    except Exception as exc:
        log.warning("RAGAS evaluation unavailable", error=str(exc)[:100])
        metric_report = {
            "context_recall": simple_context_recall(base_rows, EVAL_DATASET[:3]),
            "semantic_similarity": 0.0,
            "faithfulness": 0.0,
        }
    print("Exercise 1 - metric report:", metric_report)

    print("\n=== Exercise 2: Build an adversarial eval set ===")
    print(
        "NEW vs train.py: a harder eval split is created so retrieval failure "
        "is measured on ambiguous questions rather than only on easy ones."
    )
    print("Expected output tip: hard-set context recall is usually lower than easy-set recall.")
    hard_eval = [
        {
            "question": "Which system handles both tracing and dataset management?",
            "reference": "LangSmith provides tracing and evaluation for LangChain applications.",
            "reference_contexts": ["LangSmith provides tracing and evaluation for LangChain applications."],
        },
        {
            "question": "Which technique creates several alternate queries to improve recall?",
            "reference": "Multi-query retrieval generates several query variants to improve recall.",
            "reference_contexts": ["Multi-query retrieval generates several query variants to improve recall."],
        },
        {
            "question": "Which tool stores similar past LLM responses for related prompts?",
            "reference": "Semantic caching stores LLM responses and retrieves them for similar queries.",
            "reference_contexts": ["Semantic caching stores LLM responses and retrieves them for similar queries."],
        },
    ]
    easy_recall = simple_context_recall(base_rows, EVAL_DATASET[:3])
    hard_rows = build_rag_rows(llm, store, hard_eval, k=3)
    hard_recall = simple_context_recall(hard_rows, hard_eval)
    print("Exercise 2 - easy_context_recall:", round(easy_recall, 4))
    print("Exercise 2 - hard_context_recall:", round(hard_recall, 4))
    print("Exercise 2 - gap:", round(easy_recall - hard_recall, 4))

    print("\n=== Exercise 3: Sweep k and report ===")
    print(
        "NEW vs train.py: evaluation is repeated for multiple k values to show "
        "the cost-quality tradeoff instead of stopping at one retrieval depth."
    )
    print("Expected output tip: context recall often rises from k=1 to k=3, then shows smaller gains by k=5.")
    sweep = {}
    for k in [1, 3, 5]:
        rows = build_rag_rows(llm, store, EVAL_DATASET[:3], k=k)
        sweep[k] = {
            "context_recall": round(simple_context_recall(rows, EVAL_DATASET[:3]), 4),
            "semantic_similarity": metric_report["semantic_similarity"],
        }
    print("Exercise 3 - k sweep:", sweep)


if __name__ == "__main__":
    main()
