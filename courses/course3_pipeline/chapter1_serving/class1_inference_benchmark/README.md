# Course 3 · Chapter 1 · Class 1 — Inference fundamentals: prefill vs decode, single vs batched

> Goal: profile SmolLM2 generation under three lenses — single-request latency, batch throughput, and the prefill/decode split — using nothing but `transformers` + `torch.cuda.synchronize`. After this class you'll know what numbers to look at *before* deciding whether vLLM, TGI, or batched inference is worth the operational cost.

---

## Psycho — the mental model

A decoder generation has two distinct phases:

1. **Prefill** — the model runs the *whole* input prompt through every layer once, building the KV cache. Cost is `O(prompt_len² · layers)` for attention; this is where long prompts hurt.
2. **Decode** — for each new token, the model runs *one* token through every layer, attending against the cached KV. Cost is `O(prompt_len · layers)` per token; this is the slow drip of generation.

A request's total latency is `prefill_time + n_new_tokens × per_token_decode_time`. For a 64-token completion off a 64-token prompt, decode usually dominates. For a 16-token completion off a 4096-token prompt (RAG context!), prefill dominates.

Batching helps differently for each phase:

- **Prefill batching** is great — many small prompts share the GPU well.
- **Decode batching** requires sequences to be aligned, which is what vLLM's PagedAttention solves. Plain HF `model.generate` with a list of inputs uses padding — it works but throughput plateaus quickly.

This class lets you measure all of that on your hardware.

## Academic — what's measured

Per request type:

- `single_request_latency_ms` — end-to-end mean latency for a single (prompt, completion) pair, after warmup.
- `batched_latency_ms` — same end-to-end latency when N requests are processed together.
- `tokens_per_second` — `(N_requests × n_new_tokens) / batched_total_time`.
- `prefill_time_ms` — time to run just the prompt through the model (no new tokens generated).
- `mean_decode_per_token_ms` — `(generation_total - prefill_total) / n_new_tokens`.
- `prefill_fraction` — `prefill_time / generation_total`. High means prompt dominates; low means decode dominates.

References:
- [HF — `model.generate` / KV cache docs](https://huggingface.co/docs/transformers/llm_optims)
- [vLLM paper, *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., SOSP 2023)](https://arxiv.org/abs/2309.06180) — the production serving option, listed in research extensions
- [NVIDIA blog — *Mastering LLM Techniques: Inference Optimization*](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

### Production Latency Vocabulary

| Term | Definition | SLO target (typical) |
|---|---|---|
| **TTFT** (Time to First Token) | Prefill latency — time from request to first token | < 500 ms at p95 |
| **TPOT** (Time Per Output Token) | Decode latency per step — determines streaming fluency | < 50 ms at p95 |
| **p50 / p95 / p99** | Percentile latency — SLOs are ALWAYS set on percentiles, never mean | Service-specific |
| **Throughput @ concurrency N** | Tokens/second while N requests are in-flight simultaneously | Benchmark at N=8,16,32 |

> **Interview tip:** Never say "average latency" in a production context. Say "p95 latency" or "p99 tail latency."

## Engineering — what the code does

[`train.py`](./train.py) (no training; named for class-folder uniformity):

1. Loads SmolLM2-135M-Instruct via `shared.backbones.load_backbone` (decoder kind).
2. **Single-request benchmark** — runs `model.generate` `n_passes` times after `warmup` untimed passes. Records per-pass latency (mean, p50, p95).
3. **Batched benchmark** — same prompt repeated `batch_size` times, single `model.generate` call (with padding). Records per-batch latency and derives tokens/sec.
4. **Prefill/decode split** — runs `model(...)` on the prompt only (forward without generate) to time prefill, then runs `model.generate` to get total time, derives mean per-token decode cost.
5. Persists a result JSON via `shared.eval_harness.run_eval` with all the metrics.

The bands assert sane orderings (decode is slower than prefill per token; tokens/sec rises with batch up to a point), not absolute numbers — those vary too much across hardware.

### Gotchas
- **Always `torch.cuda.synchronize()` before timing.** GPU kernels are async; without sync, your `time.perf_counter()` measures the launch, not the work.
- **Warmup matters.** First-call overhead (cuDNN init, allocator warmup) can be 100× steady-state latency. We default `warmup=2`; bump if you see your p95 dominated by the first sample.
- **Padding wastes work.** Batched inference pads short prompts to the longest in the batch. If your batch has wildly varying lengths, throughput suffers — that's exactly the problem PagedAttention (vLLM) solves.
- **`max_new_tokens` is a ceiling, not an exact length.** EOS can stop generation early; record actual `n_new_tokens` per pass for accurate tokens/sec.

## Research — open questions / extensions

- **Sweep batch size** `1 / 4 / 16 / 64` and plot `tokens_per_second` vs batch. Where does the curve flatten? That's your "max useful batch" for this hardware.
- **Long-context regime**: feed a 2k-token prompt and 32-token completion. Does `prefill_fraction` jump above 0.9? That's when you stop optimizing decode and start optimizing prefill (or KV-cache reuse).
- **vLLM upgrade**: install `vllm` and rerun the same benchmark via its OpenAI-compatible server. Tokens/sec should jump 2–10× at higher batch sizes thanks to PagedAttention. Cost: a separate server process and a different client API.
- **Quantization stack-up**: rerun this benchmark on a QLoRA-loaded SmolLM2-360M (Course 1 ch3). 4-bit base means more model fits in cache; how does that change `prefill_time` and `decode_per_token`?

---

## How to run

```bash
bash courses/course3_pipeline/chapter1_serving/class1_inference_benchmark/run.sh
```

Smoke mode by default — ~30 s on a single GPU.

## How to verify

`results/full/<backbone>/course3_pipeline/chapter1_serving_class1_inference_benchmark/inference/bench-b<BATCH>.json`. Expected band:

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `single_request_latency_ms` | 0 | 600000 | Sanity that timing happened (machines vary) |
| `batched_latency_ms` | 0 | 600000 | Same, batched |
| `tokens_per_second` | 0 | 100000 | Whatever your hardware delivers |
| `prefill_time_ms` | 0 | 600000 | Prefill cost for the configured prompt |
| `mean_decode_per_token_ms` | 0 | 100000 | Decode cost per output token |
| `prefill_fraction` | 0.0 | 1.0 | What fraction of generation time was the initial prompt pass |

The interesting numbers are in `extras` — per-pass latency arrays, batch-vs-single ratios, etc.

## Instructor checklist

Before marking this class "done":

- [ ] All four mode sections present.
- [ ] `train.py`/`eval.py` contain no numeric literal other than `0` / `1` (everything in `configs/*.yaml`).
- [ ] `configs/default.yaml` declares an `expected_band` for every metric written.
- [ ] `run.sh` uses `HF_HOME=$PWD/.cache/huggingface` and is `chmod +x`.
- [ ] `exercises.md` has exactly three exercises.
- [ ] At least one smoke run completed end-to-end and the metric band passes.
- [ ] Linked from the parent course `README.md` table.
