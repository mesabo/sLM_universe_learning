"""Correction for Course 4 / ch4 / class 1 exercises.

This file keeps the offline-trace-first style from ``train.py`` and adds the
exercise-specific trace analysis and enrichment steps.

What is new relative to ``train.py``:
  1. A summary table over the generated JSONL traces.
  2. Extra metadata fields: backbone, n_tokens_in, n_tokens_out.
  3. A mocked LangSmith dataset-creation path when no API key is present.

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

from train import PROMPTS, run_with_offline_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def summarize_traces(trace_path: Path) -> list[dict]:
    """Exercise 1 summary table."""
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    rows = []
    for record in records:
        rows.append(
            {
                "run_id": record["run_id"][:8],
                "latency_ms": record["latency_ms"],
                "output_preview": record["output"][:40],
            }
        )
    return rows


def enrich_trace_records(trace_path: Path, backbone: str) -> dict:
    """Exercise 2 metadata enrichment.

    NEW vs train.py:
    the offline trace schema is extended after generation with token-count
    estimates and the active model backbone.
    """
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    enriched = []
    throughputs = []
    for record in records:
        record["backbone"] = backbone
        record["n_tokens_in"] = count_tokens(record["input"])
        record["n_tokens_out"] = count_tokens(record["output"])
        enriched.append(record)
        if record["latency_ms"] > 0:
            throughputs.append(record["n_tokens_out"] / record["latency_ms"])
    trace_path.write_text("".join(json.dumps(record) + "\n" for record in enriched))
    return {
        "avg_tokens_per_ms": round(sum(throughputs) / max(len(throughputs), 1), 6),
        "records": enriched,
    }


def mock_langsmith_dataset_creation(records: list[dict]) -> list[dict]:
    """Exercise 3 offline fallback.

    Without a live API key, we document the expected payload shape by building
    the arguments that would be sent to ``client.create_example(...)``.
    """
    payloads = []
    for record in records[:5]:
        payloads.append(
            {
                "inputs": {"question": record["input"]},
                "outputs": {"answer": record["output"]},
                "dataset_name": "course4-langsmith-demo",
            }
        )
    return payloads


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 42))
    log = get_logger("course4.ch4.class1.correction")
    mode = cfg.get("mode", "smoke")
    n_traces = cfg["limits"][mode]["n_traces"]

    llm = get_llm(cfg)
    trace_path = Path(cfg.get("offline_trace_path", "results/course4_langchain_ecosystem/traces.jsonl"))
    if trace_path.exists():
        trace_path.unlink()
    run_with_offline_trace(llm, PROMPTS[:n_traces], trace_path, log)
    log.info("LLM loaded", backbone=cfg.get("backbone", "?"))

    print("\n=== Exercise 1: Summarize the trace file ===")
    print(
        "NEW vs train.py: the trace JSONL is not only written; it is reloaded "
        "and summarized into an inspection-friendly table."
    )
    print("Expected output tip: expect one summary row per trace, with one row having the highest latency.")
    summary_rows = summarize_traces(trace_path)
    highest_latency = max(summary_rows, key=lambda row: row["latency_ms"])
    print("Exercise 1 - summary rows:", summary_rows)
    print("Exercise 1 - highest latency row:", highest_latency)

    print("\n=== Exercise 2: Add metadata fields ===")
    print(
        "NEW vs train.py: each trace record is enriched with backbone and rough "
        "token counts, which makes throughput estimation possible."
    )
    print("Expected output tip: enriched records should include `backbone`, `n_tokens_in`, and `n_tokens_out` fields.")
    enriched = enrich_trace_records(trace_path, backbone=cfg.get("backbone", "unknown"))
    print("Exercise 2 - avg tokens/ms:", enriched["avg_tokens_per_ms"])
    print("Exercise 2 - sample enriched record:", enriched["records"][0] if enriched["records"] else {})

    print("\n=== Exercise 3: LangSmith dataset creation (offline mock) ===")
    print(
        "NEW vs train.py: when no API key is present, the expected LangSmith "
        "example payloads are built explicitly so the live step is documented."
    )
    print("Expected output tip: mocked payloads should look like `inputs`/`outputs` objects ready for `create_example(...)`.")
    mocked_payloads = mock_langsmith_dataset_creation(enriched["records"])
    print("Exercise 3 - mocked payloads:", mocked_payloads)


if __name__ == "__main__":
    main()
