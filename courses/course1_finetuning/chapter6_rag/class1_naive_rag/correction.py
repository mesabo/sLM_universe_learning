"""Correction for Course 1 / ch6 / class 1 exercises."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import apply_overrides, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.overrides)

    print("\n=== Exercise 1: Sweep top_k ===")
    print("NEW vs train.py: only retrieval depth changes, while the generator and corpus stay fixed.")
    print("Expected output tip: `retrieval_recall_at_k` should rise or stay flat as k increases, while `answer_substring_match` may saturate earlier.")

    print("\n=== Exercise 2: Swap retriever to your fine-tuned encoder ===")
    print("NEW vs train.py: point `backbone` to the checkpoint produced in course 1 chapter 5 and rerun the same RAG loop.")
    print("Expected output tip: if the contrastive model really improved retrieval, recall@k should move first; answer quality may improve too, but less predictably.")

    print("\n=== Exercise 3: Add a reranker ===")
    print("NEW vs train.py: retrieve a larger candidate set with the encoder, then rescore it with a cross-encoder before building the final prompt.")
    print("Expected output tip: recall@5 often improves, but the reranker adds noticeable latency and quickly becomes expensive at scale.")
    print("\nConfigured top_k:", cfg.get("retrieval", {}).get("top_k"))


if __name__ == "__main__":
    main()
