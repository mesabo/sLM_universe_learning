"""Correction for Course 4 / ch3 / class 3 exercises.

This file reuses the same LangGraph theme as ``train.py`` and turns the three
exercise requests into explicit runnable examples.

What is new relative to ``train.py``:
  1. Mermaid graph inspection.
  2. A retry counter field named exactly ``retry_count``.
  3. A human-approval checkpoint with ``interrupt_before`` and state resume.

Sections marked ``NEW vs train.py`` highlight the corrections.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import apply_overrides, load_yaml
from shared.llm_client import get_llm
from shared.logging_utils import get_logger
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


class ReviewState(TypedDict):
    text: str
    sentiment: str
    category: str
    retry_count: int
    human_approved: bool


def build_graph_with_retry_and_hitl(llm):
    """Exercise 2 + 3 graph.

    NEW vs train.py:
    the state now includes ``retry_count`` and ``human_approved``, and the
    graph has an explicit approval checkpoint before finalization.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    def classify_sentiment(state: ReviewState) -> ReviewState:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(
            "Classify this review as exactly one word: positive, negative, neutral, or invalid.\n"
            "Review: {text}\nSentiment:"
        )
        chain = prompt | llm | StrOutputParser()
        try:
            raw = chain.invoke({"text": state["text"]}).strip().lower()
        except Exception:
            raw = "invalid"
        if "pos" in raw:
            sentiment = "positive"
        elif "neg" in raw:
            sentiment = "negative"
        elif "neu" in raw:
            sentiment = "neutral"
        else:
            sentiment = "invalid"
        return {**state, "sentiment": sentiment}

    def retry(state: ReviewState) -> ReviewState:
        return {**state, "retry_count": state["retry_count"] + 1}

    def request_approval(state: ReviewState) -> ReviewState:
        print("Approval requested for sentiment:", state["sentiment"])
        return {**state, "human_approved": True}

    def assign_category(state: ReviewState) -> ReviewState:
        mapping = {"positive": "praise", "negative": "complaint", "neutral": "inquiry"}
        return {**state, "category": mapping.get(state["sentiment"], "unknown")}

    def route_after_sentiment(state: ReviewState) -> Literal["retry", "request_approval", "assign_category"]:
        if state["sentiment"] == "invalid" and state["retry_count"] < 3:
            return "retry"
        if state["sentiment"] == "invalid":
            return "assign_category"
        return "request_approval"

    def route_after_approval(state: ReviewState) -> Literal["assign_category", "__end__"]:
        return "assign_category" if state["human_approved"] else "__end__"

    graph = StateGraph(ReviewState)
    graph.add_node("classify_sentiment", classify_sentiment)
    graph.add_node("retry", retry)
    graph.add_node("request_approval", request_approval)
    graph.add_node("assign_category", assign_category)
    graph.set_entry_point("classify_sentiment")
    graph.add_conditional_edges("classify_sentiment", route_after_sentiment)
    graph.add_edge("retry", "classify_sentiment")
    graph.add_conditional_edges("request_approval", route_after_approval, {"assign_category": "assign_category", "__end__": END})
    graph.add_edge("assign_category", END)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory, interrupt_before=["request_approval"])
    return compiled


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch3.class3.correction")

    llm = get_llm(cfg)
    app = build_graph_with_retry_and_hitl(llm)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"))

    print("\n=== Exercise 1: Inspect the compiled graph ===")
    print(
        "NEW vs train.py: the compiled graph is rendered as Mermaid so the "
        "topology becomes visible instead of remaining implicit in code."
    )
    print("Expected output tip: the Mermaid diagram should show start, sentiment classification, retry/approval routing, and end nodes.")
    mermaid = app.get_graph().draw_mermaid()
    print("Exercise 1 - Mermaid diagram:\n", mermaid)

    print("\n=== Exercise 2: Add a retry counter node ===")
    print(
        "NEW vs train.py: the state now uses `retry_count` explicitly and the "
        "graph loops through a retry node before eventually stopping."
    )
    print("Expected output tip: the final state should include a numeric `retry_count`, even if it remains 0 for a valid classification.")
    ambiguous = {
        "text": "This purchase was... something.",
        "sentiment": "",
        "category": "",
        "retry_count": 0,
        "human_approved": False,
    }
    ambiguous_result = app.invoke(ambiguous, config={"configurable": {"thread_id": "retry-demo"}})
    print("Exercise 2 - result:", ambiguous_result)

    print("\n=== Exercise 3: Human-in-the-loop checkpoint ===")
    print(
        "NEW vs train.py: execution pauses before `request_approval`, the state "
        "is updated manually, and the graph then resumes from checkpoint."
    )
    print("Expected output tip: the paused state should appear before approval, and the resumed state should end with a filled `category`.")
    config = {"configurable": {"thread_id": "hitl-demo"}}
    first_pass = app.invoke(
        {
            "text": "This product is excellent.",
            "sentiment": "",
            "category": "",
            "retry_count": 0,
            "human_approved": False,
        },
        config=config,
    )
    app.update_state(config, {"human_approved": True})
    resumed = app.invoke(None, config=config)
    print("Exercise 3 - paused state:", first_pass)
    print("Exercise 3 - resumed state:", resumed)


if __name__ == "__main__":
    main()
