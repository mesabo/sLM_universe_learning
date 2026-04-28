# Course 0 · Chapter 2 · Class 1 — Encoder vs decoder, side by side

> Goal: feel the difference. Same prompt, two backbones, two completely different outputs. After this class you'll instinctively know which kind of sLM to reach for in any subsequent course.

---

## Psycho — the mental model

- **Encoder sLMs** (MiniLM, BGE-small, GTE-small) are *understanders*. They map text → a fixed-size vector. Use them for retrieval, classification, clustering. They cannot generate text.
- **Decoder sLMs** (SmolLM2-135M, SmolLM2-360M) are *speakers*. They map a prompt → next-token probabilities, which you sample to produce text. They can also be used as encoders by pooling hidden states, but it's wasteful.

Mnemonic: **encoders condense, decoders extend.**

## Academic — the structural difference

The Transformer block is the same; the **attention mask** is different:

- Encoder uses *bidirectional* (full) attention — every token sees every other token. Output: a vector per token.
- Decoder uses *causal* (lower-triangular) attention — every token sees only previous tokens. Output: next-token logits.

That single mask change cascades into:
- Training objective (MLM / contrastive vs next-token prediction).
- Position-encoding choice (absolute vs RoPE / ALiBi).
- Output head (pooling head vs LM head over the vocabulary).

References:
- [BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805) — encoder
- [Llama 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) — decoder, the lineage SmolLM2 inherits
- [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

## Engineering — what the code does

`train.py` loads two backbones:

1. The **default sentence-encoder** (configurable; defaults to MiniLM).
2. The **default decoder** (configurable; defaults to SmolLM2-135M-Instruct).

For each:
- Encoder → encode the prompt to a vector → report dim and L2 norm.
- Decoder → generate `max_new_tokens` tokens → report length and a preview.

Both run `iterations.n_passes` timed passes (after `iterations.warmup` untimed ones). The result JSON's `extras` block carries `mean_latency_ms`, `p50_latency_ms`, `p95_latency_ms`, `throughput_passes_per_s`, and (for the decoder) `tokens_per_second`. Default is one pass — bump for benchmarking:

```bash
bash run.sh   # one-pass smoke
# or:
python train.py --config configs/default.yaml \
    iterations.n_passes=20 iterations.warmup=2
```

We assert that the encoder's output vector dimension matches `hidden_size`, and that the decoder produced at least one token.

## Research — open questions

- Can you use an encoder for *classification without fine-tuning*? (Hint: nearest-neighbor over class prototypes — Course 1 ch5 covers this.)
- Decoder-as-encoder via mean-pooling hidden states is a thing. Try it. How does the resulting embedding compare to MiniLM on a single nearest-neighbor query?
- SmolLM2 is *instruction-tuned* (`-Instruct`). How does the base (non-Instruct) variant respond to the same prompt? (Hint: probably much worse on instruction-style prompts.)

---

## How to run

```bash
bash courses/course0_bridge/chapter2_encoder_vs_decoder/class1_side_by_side/run.sh
```

First-time downloads: ~22 MB (MiniLM) + ~270 MB (SmolLM2-135M).

## How to verify

`results/full/<backbone>/course0_bridge/chapter2_encoder_vs_decoder_class1_side_by_side/sanity/compare.json` per backbone. Expected band:

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `output_ok` | 1 | 1 | Forward / generation produced output |
| `dim_matches_hidden` | 1 | 1 | Encoder vector dim equals reported hidden_size (encoder only, ignored for decoders) |
| `tokens_generated` | 1 | 256 | Decoder produced ≥ 1 new token (decoder only) |
| `n_passes_ran` | 1 | 100000 | Sanity: the iteration loop actually ran |
| `mean_latency_ms` | 0 | 600000 | Permissive — machines vary. Real numbers are in `extras.p50_latency_ms` / `throughput_passes_per_s`. |
