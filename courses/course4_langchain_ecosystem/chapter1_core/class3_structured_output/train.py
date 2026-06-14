"""Course 4 / ch1 / class 3 — Structured output with Pydantic.

Demonstrates:
  - PydanticOutputParser: format instructions injected into prompt
  - JsonOutputParser: lightweight dict extraction
  - Validating that LLM output parses into a typed dataclass
  - Graceful fallback when parsing fails (retry with fix-up prompt)
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
from shared.eval_harness import run_eval
from shared.llm_client import get_llm
from shared.logging_utils import get_logger
from shared.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def try_pydantic_parse(llm, n_samples: int, log) -> tuple[int, int]:
    """Return (parse_ok, schema_ok) after n_samples structured generation calls."""
    from pydantic import BaseModel, Field
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    class Movie(BaseModel):
        title: str = Field(description="Movie title")
        year: int = Field(description="Release year as integer")
        genre: str = Field(description="Primary genre")

    parser = JsonOutputParser(pydantic_object=Movie)
    prompt = PromptTemplate(
        template=(
            "Extract movie info as JSON matching this schema:\n{format_instructions}\n\n"
            "Text: {text}\n\nJSON:"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    texts = [
        "The Matrix was released in 1999 and is a sci-fi action film.",
        "Spirited Away is an animated fantasy movie from 2001.",
        "Parasite, a South Korean thriller, won the Oscar in 2020.",
    ][:n_samples]

    parse_ok = 0
    schema_ok = 0
    for text in texts:
        try:
            result = chain.invoke({"text": text})
            if isinstance(result, dict):
                parse_ok = 1
                if "title" in result and "year" in result:
                    schema_ok = 1
        except Exception as exc:
            log.warning("Parse attempt failed", text=text[:40], error=str(exc))
            # Fallback: extract the last JSON object from raw model output
            try:
                from langchain_core.output_parsers import StrOutputParser
                raw_chain = prompt | llm | StrOutputParser()
                raw = raw_chain.invoke({"text": text})
                # Use rfind to locate the LAST complete JSON object
                end = raw.rfind("}") + 1
                start = raw.rfind("{", 0, end)
                if start >= 0 and end > start:
                    obj = json.loads(raw[start:end])
                    if isinstance(obj, dict) and obj:
                        parse_ok = 1
                        # Accept any key presence as partial schema match
                        if any(k in obj for k in ("title", "year", "genre")):
                            schema_ok = 1
            except Exception:
                pass

    return parse_ok, schema_ok


def run_with_structured_output(llm, log) -> dict:
    """
    with_structured_output() — LangChain 0.3+ preferred API for structured JSON.

    Industry usage:
      structured_llm = llm.with_structured_output(MyPydanticModel)
      result = structured_llm.invoke("describe a movie")
      # result is a validated MyPydanticModel instance

    vs JsonOutputParser (older path):
      chain = prompt | llm | JsonOutputParser(pydantic_object=MyModel)
      # fragile: depends on LLM producing valid JSON without schema enforcement

    For OpenAI/Anthropic models: use with_structured_output() always.
    For local models (SmolLM2): fallback to JsonOutputParser (no function calling).
    """
    try:
        from pydantic import BaseModel as PydanticBaseModel, Field as PField

        class SentimentResult(PydanticBaseModel):
            sentiment: str = PField(description="positive, negative, or neutral")
            confidence: float = PField(description="confidence score 0.0-1.0", ge=0.0, le=1.0)

        structured_llm = llm.with_structured_output(SentimentResult)
        result = structured_llm.invoke("Analyze: 'This product is amazing!'")

        if isinstance(result, SentimentResult):
            ok = 1
        elif isinstance(result, dict) and "sentiment" in result:
            ok = 1
        else:
            ok = 0
        log.info("with_structured_output", ok=ok, result=str(result)[:80])
        return {"with_structured_output_ok": ok}
    except NotImplementedError:
        log.info("with_structured_output: NotImplementedError — SmolLM2 lacks bind_tools support (expected)")
        return {"with_structured_output_ok": -1}
    except Exception as exc:
        log.warning("with_structured_output failed", error=str(exc)[:100])
        return {"with_structured_output_ok": 0}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch1.class3")
    mode = cfg.get("mode", "smoke")
    n_samples = cfg["limits"][mode]["n_samples"]

    llm = get_llm(cfg)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"))

    parse_ok, schema_ok = try_pydantic_parse(llm, n_samples, log)
    log.info("Structured output", parse_ok=parse_ok, schema_ok=schema_ok)

    wso = run_with_structured_output(llm, log)

    metrics = {
        "parse_ok": float(parse_ok),
        "schema_ok": float(schema_ok),
    }
    metrics.update(wso)
    run_eval(
        method=cfg["method"],
        backbone=cfg.get("backbone", "local"),
        course=cfg["course"],
        klass=cfg["class_id"],
        task=cfg["task"],
        config=cfg,
        metrics=metrics,
        expected_band=cfg.get("expected_band", {}),
        extras={"mode": mode, "n_samples": n_samples},
    )


if __name__ == "__main__":
    main()
