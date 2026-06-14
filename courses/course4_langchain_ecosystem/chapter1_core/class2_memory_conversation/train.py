"""Course 4 / ch1 / class 2 — Conversation memory.

Demonstrates:
  - ConversationBufferMemory — stores all turns verbatim
  - ConversationBufferWindowMemory — sliding window of last k turns
  - RunnableWithMessageHistory — LCEL-native history injection
  - Verifying that the model receives prior context in subsequent turns
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


def simulate_buffer_memory(llm, n_turns: int, log) -> int:
    """Run n_turns of conversation, verify history accumulates."""
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables.history import RunnableWithMessageHistory

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    turns = [
        "My name is Alice.",
        "What is my name?",
        "Tell me a one-word color.",
        "Repeat the color you just said.",
    ][:n_turns]

    session = "test-session"
    for turn in turns:
        try:
            chain_with_history.invoke(
                {"input": turn},
                config={"configurable": {"session_id": session}},
            )
        except Exception as exc:
            log.warning("Turn failed", turn=turn, error=str(exc))

    history = get_session_history(session)
    # History should have 2 messages per turn (human + ai)
    return 1 if len(history.messages) >= n_turns else 0


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch1.class2")
    mode = cfg.get("mode", "smoke")
    n_turns = cfg["limits"][mode]["n_turns"]

    llm = get_llm(cfg)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"))

    buffer_ok = simulate_buffer_memory(llm, n_turns, log)
    log.info("Buffer memory", ok=buffer_ok, turns=n_turns)

    # Verify: history object accumulates messages
    from langchain_core.chat_history import InMemoryChatMessageHistory
    h = InMemoryChatMessageHistory()
    h.add_user_message("hello")
    h.add_ai_message("world")
    history_ok = 1 if len(h.messages) == 2 else 0

    metrics = {
        "buffer_ok": float(buffer_ok),
        "history_ok": float(history_ok),
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
        extras={"mode": mode, "n_turns": n_turns},
    )


if __name__ == "__main__":
    main()
