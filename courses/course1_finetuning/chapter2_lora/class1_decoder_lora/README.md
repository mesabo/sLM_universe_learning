# Course 1 · Chapter 2 · Class 1 — LoRA fine-tuning of a decoder sLM

> Goal: take SmolLM2-135M-Instruct and fine-tune it on the same smoltalk subset as ch1, but with LoRA — training only ~0.3% of the parameters and producing a tiny adapter file you can stack on top of the frozen base.

---

## Psycho — the mental model

> **One-line takeaway:** LoRA freezes the original engine and bolts on a *tiny tunable steering wheel*. The base never moves; only the bolt-on adapter learns.

Full fine-tuning rewrites the original parameters of the model. That's powerful but expensive (you're moving ~135M numbers around) and dangerous (you've now lost the original; if you didn't checkpoint, the pretrained model is gone).

LoRA's bet: **the change you need to apply is low-rank**. The full weight matrix has, say, 576×576 entries, but the *delta* you'd add for your task lives in a tiny rank-16 subspace. So instead of training the matrix, you train two small matrices `A` (16×576) and `B` (576×16); the effective update is `B @ A`. That's ~18× fewer parameters touched.

What this buys you:

- **Many adapters per base.** One per task, domain, or customer. Each adapter is a few MB.
- **Hot-swap at inference.** `peft.set_adapter("name")` flips the active personality in microseconds.
- **No catastrophic forgetting on the base.** The base's pretrained knowledge is preserved bit-for-bit because its parameters never updated. (Course 2 ch4 uses this property to isolate tasks.)
- **Memory headroom for bigger models.** With LoRA you can train SmolLM2-360M on a 8 GB card — full FT couldn't fit. QLoRA (chapter 3) takes this further.

The cost: a tiny capacity hit. At rank 16, you're saying "the optimal task delta lives in a 16-dimensional subspace". For most adaptation tasks that's plenty; for very different domains you might need rank 64+.

**Common confusion to head off:** "If the base is frozen, how does the model learn anything new?" Because at every linear layer, the forward pass is now `(W + BA) @ x` instead of `W @ x`. The `BA` term is your trainable correction — small but in exactly the dimensions that matter, because gradient descent picks them.

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
