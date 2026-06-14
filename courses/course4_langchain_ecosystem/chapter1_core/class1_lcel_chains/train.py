"""Course 4 / ch1 / class 1 — LCEL chains: PromptTemplate → LLM → OutputParser.

Demonstrates LangChain Expression Language (LCEL):
  - PromptTemplate construction
  - Pipe operator (|) for chain composition
  - StrOutputParser for plain text output
  - batch() for parallel inference
  - streaming tokens via stream()
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
from shared.llm_client import get_llm
from shared.logging_utils import get_logger
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_sentiment_chain(llm):
    """PromptTemplate | LLM | StrOutputParser — canonical LCEL pattern."""
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template(
        "Classify the sentiment of this review as POSITIVE or NEGATIVE.\n"
        "Review: {review}\nSentiment:"
    )
    return prompt | llm | StrOutputParser()


def build_summary_chain(llm):
    """Two-step chain: summarise → translate to bullet points."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    summarise = (
        ChatPromptTemplate.from_template("Summarise in one sentence: {text}")
        | llm
        | StrOutputParser()
    )
    bullets = (
        ChatPromptTemplate.from_template(
            "Convert this summary into 3 bullet points: {summary}"
        )
        | llm
        | StrOutputParser()
    )
    return {"summary": summarise} | bullets


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch1.class1")
    mode = cfg.get("mode", "smoke")
    n_prompts = cfg["limits"][mode]["n_prompts"]

    llm = get_llm(cfg)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"), provider=cfg.get("provider"))

    # --- Test 1: chain invoke ---
    chain = build_sentiment_chain(llm)
    reviews = [
        "This product is absolutely amazing, I love it!",
        "Terrible quality, broke after one day.",
        "It is okay, nothing special.",
    ][:n_prompts]

    results = []
    chain_ok = 0
    try:
        for review in reviews:
            out = chain.invoke({"review": review})
            results.append(out)
        chain_ok = 1
        log.info("Chain invoke ok", n=len(results))
    except Exception as exc:
        log.warning("Chain invoke failed", error=str(exc))

    # --- Test 2: batch ---
    batch_ok = 0
    try:
        batch_inputs = [{"review": r} for r in reviews]
        batch_out = chain.batch(batch_inputs)
        if len(batch_out) == len(reviews):
            batch_ok = 1
        log.info("Batch ok", n=len(batch_out))
    except Exception as exc:
        log.warning("Batch failed", error=str(exc))

    # --- Test 3: stream ---
    stream_ok = 0
    try:
        tokens = list(chain.stream({"review": reviews[0]}))
        if len(tokens) > 0:
            stream_ok = 1
        log.info("Stream ok", tokens=len(tokens))
    except Exception as exc:
        log.warning("Stream failed", error=str(exc))

    metrics = {
        "chain_ok": float(chain_ok),
        "batch_ok": float(batch_ok),
        "stream_ok": float(stream_ok),
    }
    run_eval(
        method=cfg["method"],
        backbone=cfg.get("backbone", "local"),
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics=metrics,
        expected_band=cfg.get("expected_band", {}),
        extras={"mode": mode, "n_prompts": n_prompts},
    )


if __name__ == "__main__":
    main()
