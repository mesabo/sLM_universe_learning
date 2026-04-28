# Course 0 · Chapter 3 · Class 1 — Tokenization & chat templates

> Goal: stop being mysterious about tokenizers. After this class you'll know exactly what `tokenizer(...)` and `apply_chat_template(...)` produce, why subword segmentation matters, and what the BOS / EOS / `<|im_start|>` tokens are doing.

---

## Psycho — the mental model

A tokenizer is **a deterministic codec**: text ⇄ list of integer IDs. It has *no learning* at inference time. Its only state is:

- A **vocabulary** (a fixed map of subwords → integer IDs).
- **Merge rules** (BPE / WordPiece / Unigram — how it splits text).
- **Special tokens** (`<s>`, `</s>`, `<pad>`, `<|im_start|>`, …).
- A **chat template** (only for instruction-tuned decoders) — a Jinja string that renders a message list into the model's expected wire format.

Mental model: think of it as the protocol layer. Get it wrong and the model sees garbage; get it right and the model "speaks" your data.

## Academic — what's actually happening

Three subword algorithms cover ~all modern tokenizers:

| Algorithm | Examples | How it splits |
|---|---|---|
| **BPE** (byte-pair encoding) | GPT-2, Llama, SmolLM2 | Greedy merges of most-frequent pairs |
| **WordPiece** | BERT, MiniLM, BGE | Greedy longest-match, prefers in-vocab pieces; uses `##` for continuation |
| **Unigram** | T5, ALBERT | Probabilistic; keeps a vocab maximizing data likelihood |

Why subword? Trade-off:
- Char-level: tiny vocab but long sequences and slow training.
- Word-level: short sequences but huge vocab and OOVs are catastrophic.
- **Subword**: vocab ~30k–100k, no OOVs, decent sequence length. Standard since 2016.

A **chat template** is only relevant for instruction-tuned decoders. SmolLM2-Instruct uses the ChatML format:

```
<|im_start|>user
What is sLM?<|im_end|>
<|im_start|>assistant
A small language model...<|im_end|>
```

`apply_chat_template(messages)` returns either a string or `input_ids` ready for `model.generate(...)`. Setting `add_generation_prompt=True` appends `<|im_start|>assistant\n` so the model continues from there.

References:
- [HF — Tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [HF — Chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [SmolLM2-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

## Engineering — what the code does

[`train.py`](./train.py) loads the **same prompt** through:

1. An encoder tokenizer (MiniLM or BGE) — shows: `input_ids`, `attention_mask`, `special_tokens_mask`.
2. A decoder tokenizer (SmolLM2-Instruct) — shows: raw `input_ids`, then `apply_chat_template` rendering with and without `add_generation_prompt`.

It writes a JSON with measured token counts. The expected band asserts that:
- The decoder produces *fewer* tokens than the encoder for an English prompt (BPE > WordPiece efficiency on natural text).
- Chat-template rendering with the generation prompt is *strictly longer* than without it (it appends `<|im_start|>assistant\n`).

## Research — open questions

- Try the same prompt in Japanese / Arabic / code. Which tokenizer wins?
- The default chat template lives in `tokenizer.chat_template`. Read it (it's Jinja). What does it do for the *system* role?
- BGE-small is *case-sensitive*; MiniLM is *uncased*. How does that affect classification fairness on user-written text?

---

## How to run

```bash
bash courses/course0_bridge/chapter3_tokenization/class1_chat_templates/run.sh
```

## How to verify

`results/full/<encoder|decoder>/course0_bridge/chapter3_tokenization_class1_chat_templates/sanity/tokens.json`. Expected band:

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `encoder_tokens` | 1 | 256 | Encoder produced ≥1 token |
| `decoder_tokens` | 1 | 256 | Decoder produced ≥1 raw token |
| `chat_no_genprompt_tokens` | 1 | 512 | Chat-rendered length |
| `chat_with_genprompt_tokens` | 1 | 512 | Chat-rendered length with assistant header |
| `genprompt_adds_tokens` | 1 | 1 | Generation prompt strictly adds tokens |
