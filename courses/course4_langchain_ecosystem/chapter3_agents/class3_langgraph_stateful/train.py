"""Course 4 / ch3 / class 3 — LangGraph: stateful multi-actor graphs.

Demonstrates:
  - StateGraph: typed state shared across nodes
  - Node functions: pure state → state transformations
  - Conditional edges: routing based on state
  - Compiling and invoking a graph
  - Cycles: a simple retry/loop until a condition is met
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TypedDict, Literal

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


class ReviewState(TypedDict):
    text: str
    sentiment: str
    category: str
    n_retries: int
    done: bool


def build_review_graph(llm, max_graph_steps: int):
    """Build a sentiment → category routing graph with retry loop."""
    from langgraph.graph import StateGraph, END

    def classify_sentiment(state: ReviewState) -> ReviewState:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        prompt = ChatPromptTemplate.from_template(
            "Classify sentiment of this review as exactly one word: positive, negative, or neutral.\n"
            "Review: {text}\nSentiment:"
        )
        chain = prompt | llm | StrOutputParser()
        try:
            out = chain.invoke({"text": state["text"]}).strip().lower()
            sentiment = "positive" if "pos" in out else ("negative" if "neg" in out else "neutral")
        except Exception:
            sentiment = "neutral"
        return {**state, "sentiment": sentiment}

    def assign_category(state: ReviewState) -> ReviewState:
        categories = {"positive": "praise", "negative": "complaint", "neutral": "inquiry"}
        return {**state, "category": categories.get(state["sentiment"], "unknown"), "done": True}

    def retry_node(state: ReviewState) -> ReviewState:
        return {**state, "n_retries": state["n_retries"] + 1}

    def route_after_sentiment(state: ReviewState) -> Literal["assign_category", "retry"]:
        if state["sentiment"] in {"positive", "negative", "neutral"}:
            return "assign_category"
        if state["n_retries"] < max_graph_steps:
            return "retry"
        return "assign_category"

    graph = StateGraph(ReviewState)
    graph.add_node("classify_sentiment", classify_sentiment)
    graph.add_node("assign_category", assign_category)
    graph.add_node("retry", retry_node)

    graph.set_entry_point("classify_sentiment")
    graph.add_conditional_edges("classify_sentiment", route_after_sentiment)
    graph.add_edge("retry", "classify_sentiment")
    graph.add_edge("assign_category", END)

    return graph.compile()


def test_graph_routing(compiled_graph, runs: list[str], log) -> tuple[int, int]:
    """Run graph on sample inputs; return (routing_ok, state_persisted_ok)."""
    routing_ok = 0
    state_persisted_ok = 0
    for text in runs:
        init: ReviewState = {
            "text": text,
            "sentiment": "",
            "category": "",
            "n_retries": 0,
            "done": False,
        }
        try:
            result = compiled_graph.invoke(init)
            if result.get("category") in {"praise", "complaint", "inquiry"}:
                routing_ok = 1
            if result.get("sentiment") and result.get("category"):
                state_persisted_ok = 1
            log.info("Graph run", text=text[:30], sentiment=result.get("sentiment"), category=result.get("category"))
        except Exception as exc:
            log.warning("Graph run failed", error=str(exc))
    return routing_ok, state_persisted_ok


def run_hitl_graph(log) -> dict:
    """
    Human-in-the-loop (HITL) with LangGraph interrupt_before + MemorySaver.

    Production pattern (DeNA, AbemaTV, LINE — content moderation / approval flows):
    1. Graph runs until it hits an interrupt_before node — pauses, returns state
    2. Human reviews the draft state (via API response or UI)
    3. Graph resumes from checkpoint with same thread_id — continues to completion

    MemorySaver: in-memory checkpointer (use SqliteSaver or PostgresSaver in prod
    to persist state across service restarts and HTTP requests).

    Interview: "How do you implement a human-approval step in an LLM workflow?"
    Answer: LangGraph StateGraph with interrupt_before=["approval_node"] + MemorySaver.
    """
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from typing import TypedDict

        class ReviewState(TypedDict):
            task: str
            draft: str
            approved: bool
            final: str

        def generate_draft(state: ReviewState) -> ReviewState:
            return {**state, "draft": f"[Draft] Processed: {state['task']}"}

        def human_approval(state: ReviewState) -> ReviewState:
            """This node is interrupted — human reviews state['draft'] before it runs."""
            return {**state, "approved": True}

        def finalize(state: ReviewState) -> ReviewState:
            return {**state, "final": state["draft"] + " [APPROVED]"}

        builder = StateGraph(ReviewState)
        builder.add_node("generate_draft", generate_draft)
        builder.add_node("human_approval", human_approval)
        builder.add_node("finalize", finalize)
        builder.set_entry_point("generate_draft")
        builder.add_edge("generate_draft", "human_approval")
        builder.add_edge("human_approval", "finalize")
        builder.add_edge("finalize", END)

        memory = MemorySaver()
        # interrupt_before=["human_approval"] pauses BEFORE human_approval runs
        graph = builder.compile(checkpointer=memory, interrupt_before=["human_approval"])

        config = {"configurable": {"thread_id": "hitl-session-001"}}

        # Phase 1: run until interrupt
        partial_state = graph.invoke(
            {"task": "Summarize this article for publication", "draft": "", "approved": False, "final": ""},
            config
        )
        assert partial_state.get("draft"), "Draft should be generated before interrupt"

        # Phase 2: human approved (simulated) — resume from checkpoint
        # In production: HTTP endpoint receives human decision, calls graph.invoke(None, config)
        final_state = graph.invoke(None, config)  # None = resume from last checkpoint
        assert final_state.get("approved") is True
        assert "[APPROVED]" in final_state.get("final", "")

        log.info("HITL graph ok", draft=partial_state["draft"][:40], final=final_state["final"][:40])
        return {"hitl_ok": 1, "draft": partial_state["draft"], "approved": final_state["approved"]}

    except ImportError as exc:
        log.warning("LangGraph MemorySaver not available", error=str(exc)[:80])
        return {"hitl_ok": -1}
    except Exception as exc:
        log.warning("HITL graph failed", error=str(exc)[:100])
        return {"hitl_ok": 0}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch3.class3")
    mode = cfg.get("mode", "smoke")
    n_runs = cfg["limits"][mode]["n_runs"]
    max_graph_steps = cfg.get("max_graph_steps", 5)

    llm = get_llm(cfg)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"))

    sample_reviews = [
        "This product is absolutely fantastic! I love it.",
        "Terrible quality, broke after one day.",
        "Can you tell me the return policy?",
        "The delivery was on time and packaging was fine.",
        "I am very disappointed with this purchase.",
        "How long does shipping take?",
        "Outstanding service and superb product.",
        "This is the worst thing I ever bought.",
    ][:n_runs]

    graph_compiled_ok = 0
    try:
        compiled_graph = build_review_graph(llm, max_graph_steps)
        graph_compiled_ok = 1
        log.info("Graph compiled ok")
    except Exception as exc:
        log.warning("Graph compilation failed", error=str(exc))
        compiled_graph = None

    routing_ok = 0
    state_persisted_ok = 0
    if compiled_graph is not None:
        routing_ok, state_persisted_ok = test_graph_routing(compiled_graph, sample_reviews, log)

    hitl_metrics = run_hitl_graph(log)

    metrics = {
        "graph_compiled_ok": float(graph_compiled_ok),
        "routing_ok": float(routing_ok),
        "state_persisted_ok": float(state_persisted_ok),
    }
    metrics.update(hitl_metrics)
    run_eval(
        method=cfg["method"],
        backbone=cfg.get("backbone", "local"),
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics=metrics,
        expected_band=cfg.get("expected_band", {}),
        extras={"mode": mode, "n_runs": n_runs},
    )


if __name__ == "__main__":
    main()
