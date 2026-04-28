# Course 1 · Chapter 2 · Class 1 — LoRA fine-tuning of a decoder sLM

> Goal: take SmolLM2-135M-Instruct and fine-tune it on the same smoltalk subset as ch1, but with LoRA — training only ~0.3% of the parameters and producing a tiny adapter file you can stack on top of the frozen base.

---

## Psycho — the mental model

Full fine-tuning **edits the original engine**. LoRA **slips a small bolt-on between the engine and the gear box** that learns to nudge the output. You can:

- Save many adapters per base (one per task / domain / customer).
- Hot-swap adapters at inference time (`peft` does this in one call).
- Avoid catastrophic forgetting (Course 2 ch4 uses this).
- Train large models on small GPUs (chapter 3's QLoRA combines LoRA with 4-bit base).

Cost: a tiny representation-capacity hit vs full FT — usually negligible at our scale.

## Academic — what's happening

For a target weight matrix $W \in \mathbb{R}^{d \times k}$ (e.g. `q_proj`), LoRA adds:

$$W' = W + \frac{\alpha}{r} B A, \quad A \in \mathbb{R}^{r \times k}, \, B \in \mathbb{R}^{d \times r}$$

with rank $r \ll \min(d, k)$ (we use $r=16$ here; common values 4–64). Only $A$ and $B$ are trained; $W$ stays frozen. $A$ is initialized Kaiming-random, $B$ is initialized to zero so the model starts identical to the base.

Trainable parameter count is $r(d+k)$ instead of $dk$. For SmolLM2's `q_proj` ($576 \times 576$): full = 331 776, LoRA-r16 = 18 432 — an 18× reduction *per matrix*.

References:
- LoRA paper: [Hu et al., 2021 (ICLR'22)](https://arxiv.org/abs/2106.09685)
- [PEFT — LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [PEFT — `LoraConfig`](https://huggingface.co/docs/peft/package_reference/lora)

## Engineering — what the code does

[`train.py`](./train.py) reuses ch1 class 2's data path but:

1. After loading the base model, calls `peft.get_peft_model(model, LoraConfig(...))`.
2. `LoraConfig.target_modules` defaults to `shared.training.lora_target_modules(model)` — for SmolLM2/Llama family that's `["q_proj","k_proj","v_proj","o_proj"]`.
3. Trains via `trl.SFTTrainer` exactly as ch1 class 2; the only difference is the model.
4. Reports trainable / total parameter counts in the result JSON.
5. Saves the adapter only (`trainer.save_model(output_dir)` writes `adapter_config.json` + `adapter_model.safetensors` — usually a few MB).

### Gotchas
- `target_modules` is architecture-specific. Wrong list → silent no-op. Always print `model.print_trainable_parameters()` (PEFT prints it for you on construction) and confirm it's > 0.
- `lora_alpha / r` is the effective scaling. Don't change `r` and `alpha` independently without thinking; the common rule of thumb is `alpha = 2*r` to keep the scale stable.
- Adapter merging (`model.merge_and_unload()`) is destructive — it bakes the LoRA into the base. Don't do it on the only copy you have.

## Research — open questions

- Sweep `r ∈ {4, 8, 16, 32, 64}` and `alpha = 2r`. Plot `eval_loss vs r`. At what rank does the curve flatten?
- IA³ (`peft.IA3Config`) is even smaller (per-element gating). Does it match LoRA on this task?
- LoRA on `o_proj` only vs all four projections — how big is the gap? (Hint: surprisingly small for some tasks.)

---

## How to run

```bash
bash courses/course1_finetuning/chapter2_lora/class1_decoder_lora/run.sh
```

Smoke by default; `MODE=full` for the longer subset.

## How to verify

`results/full/<backbone>/course1_finetuning/chapter2_lora_class1_decoder_lora/smoltalk/lora-r16.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `train_loss_final` | 0 | 6.0 | Final training loss |
| `eval_loss` | 0 | 6.0 | Held-out NLL on assistant tokens |
| `loss_decreased` | 1 | 1 | Final < initial (sanity) |
| `trainable_ratio_pct` | 0 | 5 | LoRA should train < 5% of params |
