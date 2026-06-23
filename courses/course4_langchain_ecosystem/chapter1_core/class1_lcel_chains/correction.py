"""Correction for Course 4 / ch1 / class 1 exercises.

This file is intentionally close to ``train.py`` in its project bootstrap:
argument parsing, YAML config loading, seeding, logger creation, and LLM
construction all follow the same pattern so the student can recognize the
baseline quickly.

What is new relative to ``train.py``:
  1. A ``CommaSeparatedListOutputParser`` chain that returns ``list[str]``.
  2. A ``RunnableParallel`` fan-out that runs two chains on the same input.
  3. A ``with_fallbacks()`` example that recovers from a simulated failure.

The comments marked ``NEW vs train.py`` identify the correction-specific
concepts that were not present in the original practice file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import apply_overrides, load_yaml
from shared.llm_client import get_llm
from shared.logging_utils import get_logger
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    """Mirror train.py so the correction runs with the same config flow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_synonym_chain(llm):
    """Exercise 1 solution.

    NEW vs train.py:
    ``train.py`` only uses ``StrOutputParser``, which always returns a string.
    Here we swap in ``CommaSeparatedListOutputParser`` so the final LCEL output
    is a Python list instead of raw text.
    """
    from langchain_core.output_parsers import CommaSeparatedListOutputParser
    from langchain_core.prompts import PromptTemplate

    parser = CommaSeparatedListOutputParser()
    prompt = PromptTemplate(
        template=(
            "List exactly 3 synonyms for the word below.\n"
            "{format_instructions}\n"
            "Word: {word}"
        ),
        input_variables=["word"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser


def build_summary_chain(llm):
    """Small reusable chain for Exercise 2.

    This is structurally similar to the prompt -> llm -> parser pattern already
    shown in ``train.py``. The purpose here is to create one branch of work that
    can later be combined in parallel with another branch.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(
        "Summarize the following text in one sentence:\n{text}"
    )
    return prompt | llm | StrOutputParser()


def build_keywords_chain(llm):
    """Second reusable chain for Exercise 2.

    NEW vs train.py:
    ``train.py`` demonstrates one linear chain. This helper exists so we can
    fan out into two independent LCEL branches that consume the same input.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(
        "Extract 5 short keywords from the following text.\n"
        "Return them as a comma-separated list.\n"
        "Text: {text}"
    )
    return prompt | llm | StrOutputParser()


def build_failing_primary_chain(llm):
    """Primary chain for Exercise 3 that fails once on purpose.

    NEW vs train.py:
    the original file only exercises normal success paths. For a fallback demo,
    we need a deterministic failure. ``RunnableLambda`` gives us a tiny
    pre-processing step that raises ``RuntimeError`` exactly once, before the
    request reaches the underlying LLM chain.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda

    prompt = PromptTemplate.from_template("Answer briefly: {question}")
    base_chain = prompt | llm | StrOutputParser()
    state = {"failed_once": False}

    def fail_first_call(payload):
        if not state["failed_once"]:
            state["failed_once"] = True
            raise RuntimeError("Simulated primary failure")
        return payload

    return RunnableLambda(fail_first_call) | base_chain


def build_backup_chain():
    """Backup runnable returned by ``with_fallbacks()``.

    The fallback does not call the LLM at all. That makes the control flow easy
    to verify: if this string appears in the output, the fallback path really
    executed.
    """
    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(
        lambda payload: f"Fallback response for question: {payload['question']}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch1.class1.correction")

    llm = get_llm(cfg)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"), provider=cfg.get("provider"))

    print("\n=== Exercise 1: CommaSeparatedListOutputParser ===")
    print(
        "NEW vs train.py: the parser at the end of the LCEL chain is no longer "
        "StrOutputParser. It converts the model text into a Python list."
    )
    print("Expected output tip: a Python list of 3 synonym strings, and `isinstance(..., list)` should be True.")
    synonym_chain = build_synonym_chain(llm)
    synonym_result = synonym_chain.invoke({"word": "happy"})
    print("Exercise 1 - synonym result:", synonym_result)
    print("Exercise 1 - isinstance(result, list):", isinstance(synonym_result, list))

    print("\n=== Exercise 2: RunnableParallel fan-out ===")
    print(
        "NEW vs train.py: instead of one linear chain, two independent chains "
        "run on the same input and return a dictionary of outputs."
    )
    print("Expected output tip: both sequential and parallel results should have `summary` and `keywords` keys.")
    from langchain_core.runnables import RunnableParallel

    sample_text = (
        "LangChain Expression Language lets you compose prompts, models, and parsers "
        "into one runnable pipeline with invoke, batch, and stream interfaces."
    )
    summary_chain = build_summary_chain(llm)
    keywords_chain = build_keywords_chain(llm)
    parallel_chain = RunnableParallel(summary=summary_chain, keywords=keywords_chain)

    sequential_start = time.perf_counter()
    sequential_result = {
        "summary": summary_chain.invoke({"text": sample_text}),
        "keywords": keywords_chain.invoke({"text": sample_text}),
    }
    sequential_time = time.perf_counter() - sequential_start

    parallel_start = time.perf_counter()
    parallel_result = parallel_chain.invoke({"text": sample_text})
    parallel_time = time.perf_counter() - parallel_start

    speedup = sequential_time / parallel_time if parallel_time > 0 else 0.0
    print("Exercise 2 - sequential result:", sequential_result)
    print("Exercise 2 - parallel result:", parallel_result)
    print("Exercise 2 - sequential_time:", round(sequential_time, 4))
    print("Exercise 2 - parallel_time:", round(parallel_time, 4))
    print("Exercise 2 - speedup:", round(speedup, 4))

    print("\n=== Exercise 3: with_fallbacks() ===")
    print(
        "NEW vs train.py: the primary chain is wrapped with a fallback chain, "
        "so an exception does not end the pipeline."
    )
    print("Expected output tip: the returned text should contain the fallback marker, proving the backup path ran.")
    primary_chain = build_failing_primary_chain(llm)
    backup_chain = build_backup_chain()
    resilient_chain = primary_chain.with_fallbacks([backup_chain])
    fallback_result = resilient_chain.invoke({"question": "What is LCEL?"})
    print("Exercise 3 - fallback result:", fallback_result)
    print("Exercise 3 - fallback used:", "Fallback response" in fallback_result)


if __name__ == "__main__":
    main()
