"""Correction for Course 4 / ch1 / class 3 exercises.

This file stays close to ``train.py`` for bootstrap and model loading, then
adds the missing exercise-specific structured-output patterns.

What is new relative to ``train.py``:
  1. A schema extended with ``is_sequel: bool``.
  2. A nested Pydantic extraction schema for film cast data.
  3. A retry + fix-up path for malformed JSON.

Sections marked ``NEW vs train.py`` highlight the corrections.
"""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_movie_chain(llm):
    """Exercise 1 solution with an extended schema.

    NEW vs train.py:
    the baseline schema only had ``title``, ``year``, and ``genre``. This
    correction adds ``is_sequel`` and updates the prompt accordingly.
    """
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from pydantic import BaseModel, Field

    class Movie(BaseModel):
        title: str = Field(description="Movie title")
        year: int = Field(description="Release year")
        genre: str = Field(description="Primary genre")
        is_sequel: bool = Field(description="Whether the movie is a sequel")

    parser = JsonOutputParser(pydantic_object=Movie)
    prompt = PromptTemplate(
        template=(
            "Extract movie info as JSON.\n"
            "{format_instructions}\n\n"
            "Also indicate whether it is a sequel.\n"
            "Text: {text}\nJSON:"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser


def build_cast_chain(llm):
    """Exercise 2 solution using a nested Pydantic schema."""
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from pydantic import BaseModel, Field

    class Actor(BaseModel):
        name: str = Field(description="Actor name")
        role: str = Field(description="Role played in the film")

    class FilmCast(BaseModel):
        film_title: str = Field(description="Film title")
        actors: list[Actor] = Field(description="List of actors and roles")

    parser = JsonOutputParser(pydantic_object=FilmCast)
    prompt = PromptTemplate(
        template=(
            "Extract cast information as JSON matching the schema below.\n"
            "{format_instructions}\n\n"
            "Text: {text}\nJSON:"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser


def run_fixup_retry_demo(llm, text: str, log) -> dict:
    """Exercise 3 solution.

    NEW vs train.py:
    the baseline file has a generic fallback after a parse failure. Here the
    malformed JSON is fed into a dedicated fix-up prompt and retried up to
    three times, which is the concrete pattern the exercise asks for.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda

    raw_prompt = PromptTemplate.from_template(
        "Return JSON with keys title, year, genre.\nText: {text}\nJSON:"
    )
    fix_prompt = PromptTemplate.from_template(
        "Fix this malformed JSON to match the schema "
        '{"title": str, "year": int, "genre": str}.\n'
        "Return valid JSON only.\nMalformed JSON:\n{raw}"
    )
    raw_chain = raw_prompt | llm | StrOutputParser()
    fix_chain = fix_prompt | llm | StrOutputParser()

    stats = {"original_success": 0, "fixup_success": 0}

    def parse_or_fix(payload: dict) -> dict:
        raw = raw_chain.invoke({"text": payload["text"]})
        try:
            parsed = json.loads(raw)
            stats["original_success"] += 1
            return parsed
        except json.JSONDecodeError:
            fixed = fix_chain.invoke({"raw": raw})
            parsed = json.loads(fixed)
            stats["fixup_success"] += 1
            return parsed

    resilient_chain = RunnableLambda(parse_or_fix).with_retry(stop_after_attempt=3)
    try:
        result = resilient_chain.invoke({"text": text})
        log.info("Fix-up retry demo ok", result=str(result)[:80])
        stats["result"] = result
    except Exception as exc:
        log.warning("Fix-up retry demo failed", error=str(exc)[:100])
        stats["result"] = {}
    return stats


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch1.class3.correction")

    llm = get_llm(cfg)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"), provider=cfg.get("provider"))

    print("\n=== Exercise 1: Add a field ===")
    print(
        "NEW vs train.py: the schema now includes `is_sequel`, so the parsed "
        "output carries one more validated field than the baseline example."
    )
    print("Expected output tip: the parsed dict should include `is_sequel: true` for The Dark Knight example.")
    movie_chain = build_movie_chain(llm)
    sequel_result = movie_chain.invoke(
        {
            "text": (
                "The Dark Knight (2008), a superhero film, is the sequel to "
                "Batman Begins."
            )
        }
    )
    print("Exercise 1 - parsed result:", sequel_result)
    print("Exercise 1 - is_sequel:", sequel_result.get("is_sequel"))

    print("\n=== Exercise 2: Nested schema ===")
    print(
        "NEW vs train.py: instead of a flat object, the output now contains a "
        "nested `actors` list with one typed object per cast member."
    )
    print("Expected output tip: expect a dict with `film_title` plus at least one item inside `actors`.")
    cast_chain = build_cast_chain(llm)
    cast_result = cast_chain.invoke(
        {
            "text": (
                "In The Matrix, Keanu Reeves plays Neo and Carrie-Anne Moss "
                "plays Trinity."
            )
        }
    )
    print("Exercise 2 - parsed result:", cast_result)
    print("Exercise 2 - has film_title:", "film_title" in cast_result)
    print("Exercise 2 - actor count:", len(cast_result.get("actors", [])))

    print("\n=== Exercise 3: Retry with correction ===")
    print(
        "NEW vs train.py: malformed JSON is not merely discarded; it is routed "
        "through a fix-up prompt and retried up to three times."
    )
    print("Expected output tip: either the original parse succeeds, or `fixup_success` becomes positive and returns valid JSON.")
    retry_stats = run_fixup_retry_demo(
        llm,
        "Interstellar is a 2014 science-fiction film.",
        log,
    )
    print("Exercise 3 - original_success:", retry_stats["original_success"])
    print("Exercise 3 - fixup_success:", retry_stats["fixup_success"])
    print("Exercise 3 - final result:", retry_stats["result"])


if __name__ == "__main__":
    main()
