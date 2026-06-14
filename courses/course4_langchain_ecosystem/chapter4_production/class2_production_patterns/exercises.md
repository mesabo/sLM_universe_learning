# Exercises — Class 2: Production Patterns

## Warm-up: Measure TTFT

Add a timer around `next(iter(chain.stream({"text": "..."})))` — the time to receive the first chunk is TTFT (Time to First Token). Repeat for 5 prompts and report mean TTFT. Compare TTFT to total generation latency (time for all chunks). For a local HuggingFace model, is streaming actually useful, or does the model batch-generate and flush at once?

## Apply: Semantic similarity cache

Implement a simple in-memory semantic cache: store (embedding, response) pairs. For a new query, compute its embedding and check if any stored embedding has cosine similarity > 0.95. If so, return the cached response; otherwise, call the LLM, store the pair, and return the result. Use `sentence-transformers/all-MiniLM-L6-v2` for embeddings. Measure the cache hit rate on 10 queries where 5 are paraphrases of the first 5.

## Stretch: Async streaming endpoint

Implement an async streaming endpoint using FastAPI:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
async def stream_response(q: str):
    async def generate():
        async for chunk in chain.astream({"text": q}):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")
```

Run the server with `uvicorn` and test it with `curl -N "http://localhost:8000/stream?q=What+is+LCEL"`. Verify that tokens appear progressively in the terminal rather than all at once.
