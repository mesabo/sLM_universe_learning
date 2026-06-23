"""Correction for Course 4 / ch3 / class 2 exercises.

This file stays aligned with ``train.py`` by reusing the same tool set and
LangChain agent primitives, then adds the exercise-only variants.

What is new relative to ``train.py``:
  1. Verbose intermediate-step inspection on ``AgentExecutor``.
  2. A task that should require exactly two tool calls.
  3. A token-budget guardrail wrapper around the executor result.

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
from shared.llm_client import get_llm
from shared.logging_utils import get_logger
from shared.repro import set_seed

from train import make_math_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_executor(llm, tools, verbose: bool, max_iterations: int):
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import PromptTemplate

    react_template = (
        "You have access to the following tools:\n\n"
        "{tools}\n\n"
        "Use this format:\n"
        "Thought: think about what to do\n"
        "Action: the tool name, one of [{tool_names}]\n"
        "Action Input: the input to the tool\n"
        "Observation: the tool result\n"
        "... (repeat as needed)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: the final answer\n\n"
        "Question: {input}\n"
        "{agent_scratchpad}"
    )
    prompt = PromptTemplate.from_template(react_template)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def count_trace_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


class BudgetAwareExecutor:
    """Exercise 3 wrapper for a token budget guardrail.

    NEW vs train.py:
    the baseline executor only has ``max_iterations``. This wrapper inspects
    the textual size of ``intermediate_steps`` and short-circuits the final
    output if the trace grows beyond a threshold.
    """

    def __init__(self, executor, token_budget: int):
        self.executor = executor
        self.token_budget = token_budget

    def invoke(self, payload: dict) -> dict:
        result = self.executor.invoke(payload)
        serialized_steps = "\n".join(str(step) for step in result.get("intermediate_steps", []))
        token_count = count_trace_tokens(serialized_steps)
        if token_count > self.token_budget:
            result["output"] = "BUDGET_EXCEEDED"
        result["trace_tokens"] = token_count
        return result


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch3.class2.correction")
    mode = cfg.get("mode", "smoke")
    max_iterations = cfg["limits"][mode]["max_iterations"]

    llm = get_llm(cfg)
    tools = make_math_tools()
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"))

    print("\n=== Exercise 1: Log intermediate steps ===")
    print(
        "NEW vs train.py: `verbose=True` and returned intermediate steps make "
        "the Thought-Action-Observation trace inspectable."
    )
    print("Expected output tip: expect at least one intermediate step plus a final answer, unless the local model fails to follow ReAct format.")
    verbose_executor = build_executor(llm, tools, verbose=True, max_iterations=max_iterations)
    warmup_result = verbose_executor.invoke({"input": "What is 3 plus 4?"})
    print("Exercise 1 - output:", warmup_result.get("output"))
    print("Exercise 1 - intermediate step count:", len(warmup_result.get("intermediate_steps", [])))
    print("Exercise 1 - intermediate steps:", warmup_result.get("intermediate_steps", []))

    print("\n=== Exercise 2: Multi-step chain task ===")
    print(
        "NEW vs train.py: the task is chosen so the agent should need two tool "
        "calls in sequence: multiply first, then sqrt."
    )
    print("Expected output tip: the best-case trace has exactly 2 intermediate steps and ends near the answer 12.")
    two_step_result = verbose_executor.invoke(
        {"input": "What is the square root of (3 multiplied by 48)?"}
    )
    print("Exercise 2 - output:", two_step_result.get("output"))
    print("Exercise 2 - step count:", len(two_step_result.get("intermediate_steps", [])))
    print("Exercise 2 - intermediate steps:", two_step_result.get("intermediate_steps", []))

    print("\n=== Exercise 3: Token budget guardrail ===")
    print(
        "NEW vs train.py: a token budget is checked after the run and the final "
        "answer is replaced with `BUDGET_EXCEEDED` if the trace is too large."
    )
    print("Expected output tip: with a small budget, the guardrail should often replace the normal answer with `BUDGET_EXCEEDED`.")
    guarded_executor = BudgetAwareExecutor(verbose_executor, token_budget=20)
    budget_result = guarded_executor.invoke({"input": "Multiply 6 by 7, then subtract 12."})
    print("Exercise 3 - output:", budget_result.get("output"))
    print("Exercise 3 - trace_tokens:", budget_result.get("trace_tokens"))


if __name__ == "__main__":
    main()
