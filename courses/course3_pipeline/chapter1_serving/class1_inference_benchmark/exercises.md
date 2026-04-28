# Exercises — Course 3 · ch 1 · class 1

## 1. Warm-up — sweep batch size

```bash
for b in 1 4 16 64; do
  bash run.sh --config configs/default.yaml batches.batch_size=$b
done
```

Plot `tokens_per_second` vs `batch_size`. At what batch does the curve flatten? That's your "max useful batch" for this hardware. Above it, you're paying memory without gaining throughput.

## 2. Apply — long-context regime

Edit the prompt in the YAML to a 2000-token paragraph (e.g. paste a Wikipedia article excerpt). Drop `max_new_tokens` to 32. Re-run. Does `prefill_fraction` jump above 0.9?

When prefill dominates, throughput is bounded by your prefill optimization (chunked prefill, KV-cache reuse), not your decode optimization (continuous batching). vLLM and TGI optimize different things depending on this regime — knowing your `prefill_fraction` tells you which one to reach for.

## 3. Stretch — vLLM comparison

Install `vllm` (`pip install vllm`) and serve SmolLM2-135M-Instruct via its OpenAI-compatible server. Hit it with the same `batch_size=64` setup using `httpx` from a separate process. How does `tokens_per_second` compare to plain HF batched generate? At what batch size does the gap appear?
