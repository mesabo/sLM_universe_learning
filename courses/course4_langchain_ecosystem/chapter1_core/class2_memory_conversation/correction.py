"""Correction for Course 4 / ch1 / class 2 exercises.

This file keeps the same bootstrap shape as ``train.py`` so the student can
recognize the shared setup immediately.

What is new relative to ``train.py``:
  1. Explicit counting of stored history messages after 4 turns.
  2. A manual sliding-window trim that keeps only the last ``k`` turn pairs.
  3. A multi-session example proving that ``session_id`` isolates histories.

The comments marked ``NEW vs train.py`` identify the correction-specific parts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import apply_overrides, load_yaml
from shared.llm_client import get_llm
from shared.logging_utils import get_logger
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    """Mirror train.py so the correction uses the same config flow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_history_chain(llm, store: dict):
    """Shared conversation chain used by all three exercise solutions."""
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer briefly."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def run_turn(chain_with_history, session_id: str, text: str) -> str:
    """Invoke one turn on a specific session."""
    return chain_with_history.invoke(
        {"input": text},
        config={"configurable": {"session_id": session_id}},
    )


def trim_history_to_last_k_pairs(history, k: int) -> None:
    """Exercise 2 helper.

    NEW vs train.py:
    ``train.py`` accumulates history forever. Here we explicitly trim to the
    last ``2 * k`` messages so only the latest ``k`` human/AI pairs survive.
    """
    history.messages = history.messages[-(2 * k) :]


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch1.class2.correction")

    llm = get_llm(cfg)
    store: dict = {}
    chain_with_history = build_history_chain(llm, store)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"), provider=cfg.get("provider"))

    print("\n=== Exercise 1: Count history messages ===")
    print(
        "NEW vs train.py: after 4 turns we inspect the stored history directly "
        "instead of only checking that some messages accumulated."
    )
    print("Expected output tip: history length should be 8 because each turn adds 1 human message and 1 AI message.")
    warmup_session = "count-session"
    warmup_turns = [
        "My name is Alice.",
        "What is my name?",
        "My city is Tokyo.",
        "What is my city?",
    ]
    for turn in warmup_turns:
        run_turn(chain_with_history, warmup_session, turn)
    warmup_history = store[warmup_session]
    print("Exercise 1 - history length:", len(warmup_history.messages))
    print("Exercise 1 - expected:", 8)
    print("Exercise 1 - explanation: each turn stores one human message and one AI message.")

    print("\n=== Exercise 2: Sliding window memory ===")
    print(
        "NEW vs train.py: we manually trim history to the last k pairs before "
        "each turn, which simulates window memory instead of full-buffer memory."
    )
    print("Expected output tip: after the 6th turn, only 4 messages should remain in history for `k = 2`.")
    k = 2
    window_session = "window-session"
    six_turns = [
        "Turn 1: remember apple.",
        "Turn 2: remember banana.",
        "Turn 3: remember carrot.",
        "Turn 4: remember dragonfruit.",
        "Turn 5: remember emerald.",
        "What did I say on turn 5?",
    ]
    for turn in six_turns:
        if window_session in store:
            trim_history_to_last_k_pairs(store[window_session], k)
        response = run_turn(chain_with_history, window_session, turn)
    window_history = store[window_session]
    print("Exercise 2 - history length after 6th turn:", len(window_history.messages))
    print("Exercise 2 - expected length:", 4)
    print("Exercise 2 - final response:", response)

    print("\n=== Exercise 3: Multi-session isolation ===")
    print(
        "NEW vs train.py: two session IDs are used in parallel so each user "
        "sees only their own memory state."
    )
    print("Expected output tip: Alice's answer should mention blue and Bob's should mention red.")
    run_turn(chain_with_history, "alice", "My favorite color is blue.")
    run_turn(chain_with_history, "bob", "My favorite color is red.")
    alice_answer = run_turn(chain_with_history, "alice", "What is my favorite color?")
    bob_answer = run_turn(chain_with_history, "bob", "What is my favorite color?")
    print("Exercise 3 - Alice answer:", alice_answer)
    print("Exercise 3 - Bob answer:", bob_answer)
    print("Exercise 3 - Alice history length:", len(store["alice"].messages))
    print("Exercise 3 - Bob history length:", len(store["bob"].messages))


if __name__ == "__main__":
    main()
