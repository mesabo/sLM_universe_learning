# Course 1 · Chapter 3 · Class 1 — QLoRA: 4-bit base + LoRA adapters

> Goal: train SmolLM2-360M-Instruct on the same smoltalk subset as ch2, but with the **base frozen and quantized to 4-bit** while LoRA adapters stay in bf16. After this class you'll know exactly when 4-bit is worth the complexity (almost always for ≥360M models on smaller GPUs) and when it isn't (when memory was never the bottleneck).

---

## Psycho — the mental model

Three layers stacked:

1. **The base model lives in 4-bit memory** (~90 MB for SmolLM2-360M instead of ~720 MB in bf16). Forward passes dequantize to bf16 on the fly.
2. **LoRA adapters stay in bf16** and receive gradients normally.
3. **The optimizer state lives in bf16** for the LoRA params only — paged optimizers (`paged_adamw_8bit`) keep optimizer state spillable to CPU.

Net effect: you can fit & train a much larger base on the same GPU, at the cost of slower steps (dequantization isn't free) and a small quality hit from quantization noise.

Mental shortcut: **LoRA changed which params you train; QLoRA changed which params you store.** They're orthogonal and stack cleanly.

## Academic — what's happening

QLoRA = LoRA over a base quantized to **NF4** (NormalFloat-4), an information-theoretically optimal 4-bit data type for normally-distributed weights, plus **double quantization** (the quantization constants are themselves quantized) and **paged optimizers** (CPU↔GPU paging of optimizer state via NVIDIA unified memory).

For each base weight $W \approx \mathrm{dequant}_\text{NF4}(\tilde W)$, the effective forward pass is:

$$y = (\mathrm{dequant}(\tilde W) + \tfrac{\alpha}{r} B A) x$$

Backprop only flows into $A, B$. The quantized $\tilde W$ has no gradient — it's a constant.

Key empirical claim from the QLoRA paper: **QLoRA matches full 16-bit fine-tuning quality** across many benchmarks, with ≈4× memory savings for the base.

References:
- QLoRA paper: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
- [`bitsandbytes` BitsAndBytesConfig](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)
- [PEFT — `prepare_model_for_kbit_training`](https://huggingface.co/docs/peft/package_reference/peft_model#peft.prepare_model_for_kbit_training)

## Engineering — what the code does

[`train.py`](./train.py):

1. Builds a `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bfloat16, bnb_4bit_use_double_quant=True)`.
2. Loads SmolLM2-360M-Instruct with that config — base ends up in 4-bit on GPU.
3. Calls `peft.prepare_model_for_kbit_training` (casts `LayerNorm` / output head to fp32 for stability).
4. Wraps with the same `LoraConfig` as ch2 (auto-detected target modules).
5. Trains via `trl.SFTTrainer` with `optim="paged_adamw_8bit"` and bf16.
6. Saves the adapter only — to use it later, load the base in 4-bit again and attach the adapter.

### Gotchas
- **CPU has no 4-bit path.** This class will error out on a CPU-only env. The smoke test below skips the QLoRA call and just verifies the bitsandbytes import works (skips with a clear reason if not installed).
- `prepare_model_for_kbit_training` is idempotent but `gradient_checkpointing` interacts with it — enable through that helper, not directly.
- Saving / loading: do NOT call `model.merge_and_unload()` on a 4-bit base. Merging requires fp16/bf16 base weights, which you no longer have. Save adapters separately.
- Learning rate often needs to be ~2× ch2's rate because the effective gradient scale is smaller.

## Research — open questions

- Compare ch2 (LoRA over bf16 base) vs this class (LoRA over 4-bit base) on the same backbone (135M for fairness): how much of an `eval_loss` gap, if any?
- NF4 vs FP4: re-run with `bnb_4bit_quant_type="fp4"`. Same final loss?
- Double quantization adds ~0.4 bits per weight back. With it disabled, do you measure a quality regression or only a memory difference?

---

## How to run

```bash
bash courses/course1_finetuning/chapter3_qlora/class1_decoder_qlora/run.sh
```

GPU env required (`env/slm-gpu.yml`); the script will fail loud and clear on a CPU-only env.

## How to verify

`results/full/<backbone>/course1_finetuning/chapter3_qlora_class1_decoder_qlora/smoltalk/qlora-r16.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `train_loss_final` | 0 | 6.0 | Final training loss |
| `eval_loss` | 0 | 6.0 | Held-out NLL on assistant tokens |
| `loss_decreased` | 1 | 1 | Final < initial |
| `trainable_ratio_pct` | 0 | 5 | LoRA stays under 5% of params |
| `base_in_4bit` | 1 | 1 | Sanity check: base really is loaded in 4-bit |
