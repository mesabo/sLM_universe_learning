"""Course 4 / ch3 / class 2 — ReAct agent with AgentExecutor.

Demonstrates:
  - Building a ReAct (Reason + Act) agent using create_react_agent
  - AgentExecutor: wraps agent loop with tools and stop condition
  - Multi-step tool chaining: calculator → lookup → final answer
  - Parsing intermediate steps and final output
  - Iteration limit as safety guard
"""

from __future__ import annotations

import argparse
import math
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


def make_math_tools():
    from langchain_core.tools import tool

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @tool
    def subtract(a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b

    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    @tool
    def sqrt(x: float) -> float:
        """Return the square root of x."""
        return math.sqrt(x)

    @tool
    def lookup_fact(topic: str) -> str:
        """Look up a quick fact about a topic. Topics: 'pi', 'e', 'golden_ratio'."""
        facts = {
            "pi": "Pi is approximately 3.14159, the ratio of circumference to diameter.",
            "e": "Euler's number e is approximately 2.71828.",
            "golden_ratio": "The golden ratio phi is approximately 1.61803.",
        }
        return facts.get(topic.lower(), f"No fact found for '{topic}'.")

    return [add, subtract, multiply, sqrt, lookup_fact]


def _tool_map(tools: list) -> dict:
    return {t.name: t for t in tools}


def _describe_tools(tools: list) -> str:
    lines = []
    for t in tools:
        lines.append(f"  {t.name}: {t.description}")
    return "\n".join(lines)


def _build_react_prompt(tools: list, task: str) -> str:
    tool_names = ", ".join(t.name for t in tools)
    return (
        f"You have tools: [{tool_names}]\n\n"
        f"Tool descriptions:\n{_describe_tools(tools)}\n\n"
        "Respond ONLY using this format:\n"
        "Thought: <reasoning>\n"
        "Action: <tool_name>\n"
        "Action Input: <single number or string>\n"
        "...repeat Thought/Action/Action Input as needed...\n"
        "Final Answer: <answer>\n\n"
        f"Question: {task}\n"
        "Thought:"
    )


def run_react_agent(llm, tools: list, task: str, max_iterations: int, log) -> dict:
    """Manual ReAct loop — parses Action/Action Input tokens, dispatches tools."""
    import re
    from langchain_core.output_parsers import StrOutputParser

    tmap = _tool_map(tools)
    history = _build_react_prompt(tools, task)
    from langchain_core.prompts import PromptTemplate
    steps_taken = []

    for _ in range(max_iterations):
        try:
            prompt_tmpl = PromptTemplate.from_template("{text}")
            chain = prompt_tmpl | llm | StrOutputParser()
            raw = chain.invoke({"text": history})
        except Exception as exc:
            log.warning("LLM call failed in ReAct loop", error=str(exc))
            break

        # Check for Final Answer
        fa_match = re.search(r"Final Answer[:\s]+(.+)", raw, re.IGNORECASE)
        if fa_match:
            return {"output": fa_match.group(1).strip(), "steps": len(steps_taken), "agent_ok": 1}

        # Parse Action / Action Input
        action_match = re.search(r"Action[:\s]+(\w+)", raw, re.IGNORECASE)
        ainput_match = re.search(r"Action Input[:\s]+(.+?)(?:\n|$)", raw, re.IGNORECASE)
        if not action_match:
            break

        tool_name = action_match.group(1).strip()
        raw_input = ainput_match.group(1).strip() if ainput_match else ""

        tool = tmap.get(tool_name)
        if tool is None:
            observation = f"Tool '{tool_name}' not found."
        else:
            try:
                # Try numeric first, then string
                try:
                    import ast
                    arg_val = ast.literal_eval(raw_input)
                except Exception:
                    arg_val = raw_input
                # single-arg tools: pass positional; multi-arg: wrap as dict
                if isinstance(arg_val, dict):
                    observation = str(tool.invoke(arg_val))
                else:
                    # Infer first param name from tool schema
                    schema = tool.args
                    first_param = next(iter(schema)) if schema else "x"
                    observation = str(tool.invoke({first_param: arg_val}))
                steps_taken.append({"tool": tool_name, "input": raw_input, "result": observation})
            except Exception as exc:
                observation = f"ERROR: {exc}"

        history += f" {raw}\nObservation: {observation}\nThought:"

    # Fallback: return last raw output even without Final Answer marker
    return {"output": raw[:200] if "raw" in dir() else "", "steps": len(steps_taken), "agent_ok": 1 if steps_taken else 0}


def run_create_react_agent(llm, tools: list, task: str, max_iterations: int, log) -> dict:
    """
    Modern ReAct: create_react_agent + AgentExecutor — LangChain 0.3+ standard.

    create_react_agent generates the ReAct prompt template automatically.
    AgentExecutor handles the Thought/Action/Observation loop internally.

    Production note: AgentExecutor is being superseded by LangGraph for
    complex workflows, but remains the standard for simple tool-calling agents.

    In production, pull the prompt from LangChain Hub:
      from langchain import hub
      prompt = hub.pull("hwchase17/react")  # requires LANGCHAIN_API_KEY
    """
    try:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_core.prompts import PromptTemplate

        react_template = (
            "You have access to the following tools:\n\n"
            "{tools}\n\n"
            "Use this format:\n"
            "Thought: think about what to do\n"
            "Action: the tool name, one of [{tool_names}]\n"
            "Action Input: the input to the tool\n"
            "Observation: the tool result\n"
            "... (repeat Thought/Action/Action Input/Observation as needed)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer\n\n"
            "Question: {input}\n"
            "{agent_scratchpad}"
        )
        prompt = PromptTemplate.from_template(react_template)
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        result = executor.invoke({"input": task})
        return {
            "output": result.get("output", ""),
            "steps": len(result.get("intermediate_steps", [])),
            "agent_ok": 1,
            "path": "create_react_agent",
        }
    except Exception as exc:
        log.warning("create_react_agent failed", error=str(exc)[:100])
        return {"output": "", "steps": 0, "agent_ok": 0, "path": "create_react_agent"}


def run_tool_calling_agent(llm, tools: list, task: str, max_iterations: int, log) -> dict:
    """Tool-calling agent via AgentExecutor (requires chat model with bind_tools)."""
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful math assistant. Use tools to solve problems step by step."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        result = executor.invoke({"input": task})
        return {
            "output": result.get("output", ""),
            "steps": len(result.get("intermediate_steps", [])),
            "agent_ok": 1,
        }
    except Exception as exc:
        log.warning("Tool-calling agent failed (needs chat model)", error=str(exc)[:80])
        return {"output": "", "steps": 0, "agent_ok": 0}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch3.class2")
    mode = cfg.get("mode", "smoke")
    max_iterations = cfg["limits"][mode]["max_iterations"]
    n_tasks = cfg["limits"][mode]["n_tasks"]

    llm = get_llm(cfg)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"))

    tools = make_math_tools()
    tasks = [
        "What is 15 plus 27?",
        "What is the square root of 144?",
        "What is the value of pi?",
        "Multiply 6 by 7, then subtract 12.",
    ][:n_tasks]

    results = []
    paths_used = []
    for task in tasks:
        # Primary: create_react_agent (LangChain 0.3+ standard)
        r = run_create_react_agent(llm, tools, task, max_iterations, log)
        if not r["agent_ok"]:
            # Secondary: tool-calling agent (requires chat model with bind_tools)
            r = run_tool_calling_agent(llm, tools, task, max_iterations, log)
            r["path"] = "tool_calling"
        if not r["agent_ok"]:
            # Fallback: manual string-parser ReAct loop
            r = run_react_agent(llm, tools, task, max_iterations, log)
            r["path"] = "string_parser"
        results.append(r)
        paths_used.append(r.get("path", "unknown"))
        log.info("Task done", task=task[:40], steps=r["steps"], ok=r["agent_ok"], path=r.get("path"))

    agent_ok = 1 if any(r["agent_ok"] for r in results) else 0
    avg_steps = sum(r["steps"] for r in results) / max(len(results), 1)

    metrics = {
        "agent_ok": float(agent_ok),
        "avg_steps": avg_steps,
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
        extras={"mode": mode, "n_tasks": n_tasks},
    )


if __name__ == "__main__":
    main()
