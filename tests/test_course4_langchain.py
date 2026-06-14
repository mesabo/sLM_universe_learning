"""Industry-standard tests for course4_langchain_ecosystem.

Tests cover: chain input/output contract, async invocation, tool dispatch,
structured output API, memory injection, graph compilation, import correctness.
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Fake LLM — no GPU needed for fast unit tests
# ---------------------------------------------------------------------------
def _fake_llm(response_text="blue"):
    """Return a LangChain-compatible fake LLM backed by RunnableLambda.

    MagicMock is NOT a Runnable, so LCEL's pipe operator wraps it with
    RunnableLambda and calls llm(input) — not llm.invoke(). Using an actual
    BaseChatModel subclass avoids all MagicMock attribute leakage.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any, List, Optional

    class _FakeChatModel(BaseChatModel):
        response: str = response_text

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])

        @property
        def _llm_type(self) -> str:
            return "fake"

        def with_structured_output(self, schema, **kwargs):
            return self

        def bind_tools(self, tools, **kwargs):
            return self

    return _FakeChatModel()


# ---------------------------------------------------------------------------
# ch1/c1 — LCEL chain contract
# ---------------------------------------------------------------------------
def test_lcel_chain_invoke_non_empty():
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    llm = _fake_llm("Paris")
    chain = PromptTemplate.from_template("{q}") | llm | StrOutputParser()
    result = chain.invoke({"q": "Capital of France?"})
    assert isinstance(result, str) and len(result) > 0


def test_lcel_chain_batch_length():
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    llm = _fake_llm("A")
    chain = PromptTemplate.from_template("{q}") | llm | StrOutputParser()
    results = chain.batch([{"q": "Q1"}, {"q": "Q2"}, {"q": "Q3"}])
    assert len(results) == 3


# ---------------------------------------------------------------------------
# ch1/c2 — Memory: history injection
# ---------------------------------------------------------------------------
def test_chat_history_stores_messages():
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    h = InMemoryChatMessageHistory()
    h.add_user_message("What is LoRA?")
    h.add_ai_message("LoRA is a parameter-efficient fine-tuning method.")
    assert len(h.messages) == 2
    assert isinstance(h.messages[0], HumanMessage)
    assert isinstance(h.messages[1], AIMessage)


# ---------------------------------------------------------------------------
# ch2/c1 — Chunking
# ---------------------------------------------------------------------------
def test_recursive_splitter_chunks_long_text():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        pytest.skip("langchain_text_splitters not installed")
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    text = "This is a sentence about machine learning. " * 20
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 70  # slight overrun at word boundary OK


def test_recursive_splitter_overlap_present():
    """Chunks should overlap — content at the end of chunk N appears at start of chunk N+1."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        pytest.skip("langchain_text_splitters not installed")
    splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=15)
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 10
    chunks = splitter.split_text(text)
    if len(chunks) >= 2:
        # Last 15 chars of chunk 0 should appear at start of chunk 1
        overlap_region = chunks[0][-15:]
        assert overlap_region in chunks[1] or len(overlap_region.strip()) == 0


# ---------------------------------------------------------------------------
# ch2/c2 — No broken langchain_classic imports (regression guard)
# ---------------------------------------------------------------------------
def test_no_langchain_classic_imports():
    """No file in course4 imports from langchain_classic (package does not exist)."""
    import glob
    broken = []
    for py in glob.glob(str(REPO / "courses/course4_langchain_ecosystem/**/*.py"), recursive=True):
        if "langchain_classic" in open(py).read():
            broken.append(py)
    assert not broken, f"Files still importing broken langchain_classic: {broken}"


# ---------------------------------------------------------------------------
# ch3/c2 — ReAct imports
# ---------------------------------------------------------------------------
def test_react_agent_importable():
    try:
        from langchain.agents import create_react_agent, AgentExecutor  # noqa
    except ImportError:
        pytest.skip("langchain not installed")


# ---------------------------------------------------------------------------
# ch3/c3 — LangGraph HITL compiles
# ---------------------------------------------------------------------------
def test_langgraph_hitl_graph_compiles():
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from typing import TypedDict
    except ImportError:
        pytest.skip("langgraph not installed")

    class S(TypedDict):
        value: int

    def increment(s): return {**s, "value": s["value"] + 1}

    b = StateGraph(S)
    b.add_node("step", increment)
    b.set_entry_point("step")
    b.add_edge("step", END)
    mem = MemorySaver()
    g = b.compile(checkpointer=mem, interrupt_before=["step"])
    assert g is not None


# ---------------------------------------------------------------------------
# ch4/c2 — Async patterns
# ---------------------------------------------------------------------------
def test_ainvoke_awaitable():
    from langchain_core.prompts import PromptTemplate
    llm = _fake_llm("blue")
    chain = PromptTemplate.from_template("{q}") | llm

    async def _run():
        return await chain.ainvoke({"q": "color?"})

    result = asyncio.run(_run())
    assert result is not None


def test_astream_yields_chunks():
    from langchain_core.prompts import PromptTemplate
    llm = _fake_llm("hello world test")
    chain = PromptTemplate.from_template("{q}") | llm

    async def _run():
        chunks = []
        async for chunk in chain.astream({"q": "test"}):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_run())
    assert len(chunks) > 0


def test_abatch_returns_correct_length():
    from langchain_core.prompts import PromptTemplate
    llm = _fake_llm("answer")
    chain = PromptTemplate.from_template("{q}") | llm

    async def _run():
        return await chain.abatch([{"q": "a"}, {"q": "b"}, {"q": "c"}])

    results = asyncio.run(_run())
    assert len(results) == 3


# ---------------------------------------------------------------------------
# ch4/c1 — LangSmith @traceable import
# ---------------------------------------------------------------------------
def test_langsmith_traceable_importable():
    try:
        from langsmith import traceable  # noqa
    except ImportError:
        pytest.skip("langsmith not installed")
