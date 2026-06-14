# Exercises — Class 1: LCEL Chains

## Warm-up: Add a second parser

Modify `train.py` to add a chain that uses `CommaSeparatedListOutputParser` instead of `StrOutputParser`. Ask the LLM to list three synonyms for a word and parse the result into a Python list. Verify `isinstance(result, list)`.

## Apply: Parallel fan-out with RunnableParallel

Build a chain that runs two prompts in parallel on the same input using `RunnableParallel`:

```python
from langchain_core.runnables import RunnableParallel
chain = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
)
result = chain.invoke({"text": "..."})
# result == {"summary": "...", "keywords": "..."}
```

Measure the wall-clock time vs running each chain sequentially. Report the speedup (or lack thereof) for a local HuggingFace model.

## Stretch: Runnable with fallback

Use `.with_fallbacks([backup_chain])` to create a chain that tries the primary LLM first and falls back to a backup (e.g., a simpler prompt or a cached response) on `Exception`. Simulate a failure in the primary chain by patching `llm.invoke` to raise `RuntimeError` on the first call. Verify the fallback response is returned.
