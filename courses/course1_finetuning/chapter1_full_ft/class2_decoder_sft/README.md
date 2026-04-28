# Course 1 · Chapter 1 · Class 2 — Full SFT of a decoder sLM with TRL

> Goal: take SmolLM2-135M-Instruct and continue instruction-tuning it on a small open chat dataset using `trl.SFTTrainer`. After this class you'll know what an "SFT step" actually is, what a chat template does at training time, and why the loss only counts assistant tokens.

---

## Psycho — the mental model

Supervised fine-tuning of a decoder is **next-token prediction with selective loss masking**. You take dialogues like:

```
[system]
You are a concise teacher.
[user]
Explain LoRA in two sentences.
[assistant]
LoRA inserts low-rank adapters into the attention/FFN projections of a frozen base model, so you only train ~0.1% of params...
```

You train the model to predict each *assistant* token given everything before it. **You do NOT train it to predict user/system tokens** — those are inputs, not targets. TRL's `SFTTrainer` handles this automatically when you pass a chat template.

Mental model: SFT is "show, don't tell". You're not changing the loss function; you're just showing the model more of the kind of completion you want.

## Academic — what's happening

The objective is the standard causal LM loss restricted to assistant spans:

$$\mathcal{L}_\text{SFT} = -\frac{1}{|A|} \sum_{(p, t) \in A} \log p_\theta(t \mid p)$$

where $A$ is the set of (prefix, target-token) pairs falling inside an assistant turn.

In practice:
1. `apply_chat_template(messages, add_generation_prompt=False)` renders messages to a single string.
2. The data collator builds `labels` such that every non-assistant token has `label = -100` (ignored by the cross-entropy loss).
3. Backprop teaches the model the assistant continuation conditioned on system+user.

Why Smol talk? Because the user's hardware (CUDA 4–7) can fully load SmolLM2-135M and a small SFT run in minutes. We use a tiny subset of [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) — the official dataset that produced SmolLM2-Instruct.

References:
- [TRL — `SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer)
- [SmolLM2 paper](https://huggingface.co/papers/2502.02737) (and the model card)
- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — the SFT pattern

## Engineering — what the code does

[`train.py`](./train.py):

1. Loads SmolLM2-135M-Instruct via `shared.backbones.load_backbone`.
2. Loads a tiny smoltalk subset via `datasets.load_dataset(..., split=...)` with a config-driven cap.
3. Wraps with `trl.SFTTrainer`, passing `formatting_func` that calls `tokenizer.apply_chat_template(messages, tokenize=False)`.
4. Trains for the configured steps; evaluates by holding out the last N rows.
5. Computes a coarse "post-SFT generation looks reasonable" metric: mean log-probability the model assigns to the held-out assistant tokens.

We deliberately keep the metric simple — chapter 6 (eval discipline) is where we get rigorous about chat-style evaluation.

### Gotchas
- **CPU is impractically slow** for SmolLM2 SFT even at 135M. Smoke mode runs ~50 steps and is still uncomfortable. Use the GPU env.
- `tokenizer.pad_token` defaults to `None` for many decoders; we set it to `eos_token` (TRL warns about this — that's fine for our setting).
- `bf16` is the default dtype on CUDA; on CPU we silently fall back to `float32`.

## Research — open questions

- Replace smoltalk with `tatsu-lab/alpaca` (older, lower quality). Does eval log-prob improve on the smoltalk held-out anyway? What does that tell you about transfer between SFT corpora?
- Pure SFT degrades safety-tuned behaviors. Try a "harmful" prompt before and after SFT — what changes?
- The `-Instruct` SmolLM2 was *already* SFT'd by HuggingFaceTB. Why do *more* SFT? (Hint: domain shift, style drift, instruction-following on YOUR distribution.)

---

## How to run

```bash
bash courses/course1_finetuning/chapter1_full_ft/class2_decoder_sft/run.sh
```

Smoke mode by default (~50 steps on a 1024-row subset). Use `MODE=full` for the full subset.

## How to verify

`results/full/<backbone>/course1_finetuning/chapter1_full_ft_class2_decoder_sft/smoltalk/sft.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `train_loss_final` | 0 | 6.0 | Final training loss (raw NLL) |
| `eval_loss` | 0 | 6.0 | Held-out NLL on assistant tokens |
| `loss_decreased` | 1 | 1 | Final train loss is below initial — sanity check that learning happened |
