# Exercises — Class 1: LangSmith Tracing

## Warm-up: Parse the trace file

After running in smoke mode, load `results/course4_langchain_ecosystem/traces.jsonl` and print a summary table: `run_id` (first 8 chars), `latency_ms`, and the first 40 characters of `output`. Which invocation had the highest latency? What does the latency variance tell you about the local model's generation speed?

## Apply: Add metadata fields

Extend each trace record with `backbone`, `n_tokens_in`, and `n_tokens_out` fields. Use `tiktoken.get_encoding("gpt2")` to estimate token counts from the input prompt and output string. Reload the JSONL file and compute average `n_tokens_out / latency_ms` — this is a rough tokens-per-millisecond throughput metric. Report it.

## Stretch: LangSmith dataset creation (live API)

If you have a `LANGSMITH_API_KEY`:
1. Run 5 invocations with `LANGCHAIN_TRACING_V2=true`.
2. Use `langsmith.Client().list_runs(project_name="course4-demo", limit=5)` to retrieve the runs.
3. For each run, call `client.create_example(inputs=..., outputs=..., dataset_id=...)` to add it to a new dataset named `"course4-langsmith-demo"`.
4. Verify the dataset appears in the LangSmith UI.

If you do not have a key, document what each step would do and write a unit test that mocks the `Client` calls and verifies the expected arguments.
