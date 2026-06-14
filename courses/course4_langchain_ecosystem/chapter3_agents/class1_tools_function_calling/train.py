"""Course 4 / ch3 / class 1 — Tools and function calling.

Demonstrates:
  - @tool decorator: wrapping Python functions as LangChain tools
  - Tool schema: name, description, args_schema (Pydantic)
  - Invoking tools directly and through a chain
  - Verifying the LLM can select and call the right tool
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Annotated

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


def make_tools():
    from langchain_core.tools import tool

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers and return the result."""
        return a + b

    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers and return the result."""
        return a * b

    @tool
    def sqrt(x: float) -> float:
        """Return the square root of x."""
        return math.sqrt(x)

    @tool
    def word_count(text: str) -> int:
        """Count the number of words in the given text."""
        return len(text.split())

    return [add, multiply, sqrt, word_count]


def test_tool_direct_invocation(tools: list) -> int:
    """Verify tools can be invoked directly with correct output."""
    add, multiply, sqrt_tool, word_count = tools
    ok = 1
    if abs(add.invoke({"a": 3.0, "b": 4.0}) - 7.0) > 1e-9:
        ok = 0
    if abs(multiply.invoke({"a": 3.0, "b": 4.0}) - 12.0) > 1e-9:
        ok = 0
    if abs(sqrt_tool.invoke({"x": 16.0}) - 4.0) > 1e-9:
        ok = 0
    if word_count.invoke({"text": "hello world"}) != 2:
        ok = 0
    return ok


def test_tool_schema(tools: list) -> int:
    """Verify each tool has name, description, and args_schema."""
    for t in tools:
        if not t.name or not t.description:
            return 0
        if t.args_schema is None:
            return 0
    return 1


def test_llm_tool_bind(llm, tools: list, n_prompts: int, log) -> int:
    """Bind tools to LLM and check it returns some ToolCall or text response."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    try:
        llm_with_tools = llm.bind_tools(tools)
        prompts = [
            "What is 3 + 4?",
            "What is 5 * 6?",
            "How many words are in 'hello world foo'?",
        ][:n_prompts]
        called = 0
        for p in prompts:
            result = llm_with_tools.invoke(p)
            # Only count if tool_calls were actually dispatched
            if hasattr(result, "tool_calls") and result.tool_calls:
                called += 1
            elif hasattr(result, "additional_kwargs") and result.additional_kwargs.get("tool_calls"):
                called += 1
            # Note: SmolLM2 may return 0 tool_calls — that's acceptable.
            # This test verifies bind_tools doesn't crash the LLM, not that
            # SmolLM2 (which lacks function-calling training) actually uses them.
        return 1 if called > 0 else 0
    except Exception as exc:
        log.warning("Tool bind test failed", error=str(exc))
        return 0


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch3.class1")
    mode = cfg.get("mode", "smoke")
    n_prompts = cfg["limits"][mode]["n_prompts"]

    llm = get_llm(cfg)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"))

    tools = make_tools()
    direct_ok = test_tool_direct_invocation(tools)
    schema_ok = test_tool_schema(tools)
    bind_ok = test_llm_tool_bind(llm, tools, n_prompts, log)

    log.info("Tool tests", direct=direct_ok, schema=schema_ok, bind=bind_ok)

    metrics = {
        "tool_direct_ok": float(direct_ok),
        "tool_schema_ok": float(schema_ok),
        "tool_bind_ok": float(bind_ok),
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
        extras={"mode": mode, "n_tools": len(tools)},
    )


if __name__ == "__main__":
    main()
