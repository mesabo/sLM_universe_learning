"""Correction for Course 4 / ch3 / class 1 exercises.

This file preserves the same tool-construction pattern as ``train.py`` and
adds the three exercise extensions as explicit teaching examples.

What is new relative to ``train.py``:
  1. Direct inspection of tool schemas.
  2. A stateful tool with side effects.
  3. A runtime ``ToolRegistry`` abstraction.

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
from shared.logging_utils import get_logger
from shared.repro import set_seed

from train import make_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_increment_tool():
    """Exercise 2 tool with side effects.

    NEW vs train.py:
    the baseline tools are pure functions. This one mutates shared state, which
    makes the side effect visible and therefore easier to discuss.
    """
    from langchain_core.tools import tool

    counter = {"value": 0}

    @tool
    def increment(amount: int) -> int:
        """Increment the internal counter by amount and return the new value."""
        counter["value"] += amount
        return counter["value"]

    return increment, counter


class ToolRegistry:
    """Exercise 3 registry abstraction.

    NEW vs train.py:
    the original file passes a plain list of tools around. A registry gives the
    agent layer a dynamic lookup point, which is closer to plugin-style systems.
    """

    def __init__(self) -> None:
        self._tools = {}

    def register(self, tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str):
        return self._tools.get(name)

    def all(self) -> list:
        return list(self._tools.values())

    def dispatch(self, name: str, args: dict):
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        return tool.invoke(args)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch3.class1.correction")

    tools = make_tools()
    add, multiply, sqrt_tool, _word_count = tools
    log.info("Tools built", n_tools=len(tools))

    print("\n=== Exercise 1: Inspect tool schemas ===")
    print(
        "NEW vs train.py: instead of only checking that a schema exists, the "
        "schema payload itself is printed and inspected."
    )
    print("Expected output tip: `add.args` should expose fields for `a` and `b`, and the manual invoke should return 300.0.")
    print("Exercise 1 - add.args:", add.args)
    print("Exercise 1 - add input schema:", add.get_input_schema().schema())
    print("Exercise 1 - add.invoke({'a': 100.0, 'b': 200.0}):", add.invoke({"a": 100.0, "b": 200.0}))

    print("\n=== Exercise 2: Tool with side effects ===")
    print(
        "NEW vs train.py: the tool mutates shared state, which makes repeated "
        "invocations observable across calls."
    )
    print("Expected output tip: three calls with amount=5 should produce outputs ending at 15.")
    increment, counter = build_increment_tool()
    outputs = [increment.invoke({"amount": 5}) for _ in range(3)]
    print("Exercise 2 - outputs:", outputs)
    print("Exercise 2 - final counter:", counter["value"])

    print("\n=== Exercise 3: Dynamic tool registry ===")
    print(
        "NEW vs train.py: a registry replaces the hardcoded tool list with "
        "runtime registration and name-based dispatch."
    )
    print("Expected output tip: registered tool names should include add, multiply, and sqrt, and dispatch should return numeric results.")
    registry = ToolRegistry()
    registry.register(add)
    registry.register(multiply)
    registry.register(sqrt_tool)
    print("Exercise 3 - registered tools:", [tool.name for tool in registry.all()])
    print("Exercise 3 - dispatch add:", registry.dispatch("add", {"a": 2.0, "b": 5.0}))
    print("Exercise 3 - dispatch multiply:", registry.dispatch("multiply", {"a": 3.0, "b": 7.0}))
    print("Exercise 3 - dispatch sqrt:", registry.dispatch("sqrt", {"x": 81.0}))


if __name__ == "__main__":
    main()
