# Course 1 · Chapter 3 · Class 1 — QLoRA: 4-bit base + LoRA adapters

> **Goal:** Train SmolLM2-360M-Instruct using **QLoRA (Quantized LoRA)**. You will load the base model in 4-bit precision to save massive amounts of memory, while still training high-quality adapters.

---

## 🧭 The 5 W's & 1 H (Foundations)

### WHAT are we doing?
We are performing **Quantized Parameter-Efficient Fine-Tuning**.
*   **The Component:** We take the concept of **LoRA** (from Chapter 2) and combine it with **Quantization**.
*   **The Quantization:** We "compress" the 135M or 360M weights of the original model from 16-bit numbers (bf16) down to 4-bit numbers (NF4).
*   **The Hybrid:** The base model is compressed (frozen), while the LoRA adapters remain in high precision (trainable).

### WHY are we doing this?
*   **Maximum Memory Savings:** 4-bit quantization reduces the memory footprint of the base model by about **4x**. For a 360M model, this means going from ~720MB of VRAM down to just ~90MB.
*   **Democratization of AI:** QLoRA allows you to train much larger models (like Llama-3 8B) on consumer GPUs (like an RTX 3060 or 4070) that simply couldn't hold the model otherwise.
*   **Near-Zero Quality Loss:** Thanks to advanced techniques like **NormalFloat (NF4)**, the model performs almost as well in 4-bit as it does in 16-bit once fine-tuned.

### WHEN should you use this?
*   Use **QLoRA** whenever memory is your primary bottleneck.
*   Use it when you want to fine-tune the largest model that can possibly fit on your GPU.
*   *Note:* If you have plenty of VRAM (e.g., training a tiny 135M model on an A100), standard LoRA might be slightly faster because it skips the dequantization step.

### WHERE does the compression happen?
The compression happens in the **GPU VRAM**.
1.  The model is loaded from disk in 4-bit format.
2.  During a "forward pass," a small chunk of these 4-bit weights is briefly turned back into 16-bit (dequantized) to do the math.
3.  As soon as the math for that layer is done, the weights are "discarded," and we move to the next 4-bit chunk. This keeps the *peak* memory usage very low.

### HOW does it work (The Pipeline)?
1.  **Configure:** Use `BitsAndBytesConfig` to tell the model to load in 4-bit using `nf4`.
2.  **Load:** Load the model with `quantization_config`.
3.  **Stable-Kbit:** Call `prepare_model_for_kbit_training`. This ensures that sensitive parts of the model (like LayerNorm) stay in high precision so the model doesn't "break."
4.  **Attach LoRA:** Add the standard LoRA adapters on top.
5.  **Train:** Use a "paged optimizer" (like `paged_adamw_8bit`) which can swap memory to your system RAM if the GPU fills up.

---

## 🧠 Psycho — the mental model

> **One-line takeaway:** *LoRA changed which params you **train**; QLoRA changed which params you **store**.* The two ideas are orthogonal and compose perfectly.

QLoRA is what lets students with one consumer GPU train models that "should" need a server. The trick is to shrink the part you're not editing anyway. Three layers stacked:

1. **The base model lives in 4-bit memory** (~90 MB for SmolLM2-360M instead of ~720 MB in bf16). Forward passes dequantize to bf16 on the fly.
2. **LoRA adapters stay in bf16** and receive gradients normally — these are the only params learning.
3. **The optimizer state lives in bf16 for the LoRA params only** — paged optimizers (`paged_adamw_8bit`) keep optimizer state spillable to CPU memory when GPU is tight.

Net effect: you can fit & train a much larger base on the same GPU, at the cost of slightly slower steps (dequantization isn't free) and a small quality hit from quantization noise.

The intuition that helps: **the base is a *constant*, not a variable**. You're not editing those weights, you're just using them in forward passes. Storing a constant in 4-bit is fine; storing the *thing being learned* in 4-bit would lose precision in the gradients.

---

## 🎓 Academic — what's happening

QLoRA = LoRA over a base quantized to **NF4** (NormalFloat-4), an information-theoretically optimal 4-bit data type for normally-distributed weights, plus **double quantization** (the quantization constants are themselves quantized) and **paged optimizers** (CPU↔GPU paging of optimizer state via NVIDIA unified memory).

For each base weight $W \approx \mathrm{dequant}_\text{NF4}(\tilde W)$, the effective forward pass is:

$$y = (\mathrm{dequant}(\tilde W) + \tfrac{\alpha}{r} B A) x$$

**Key Terms for Students:**
*   **NF4 (NormalFloat 4):** A specialized 4-bit format that is much better at representing neural network weights than standard integers.
*   **Double Quantization:** Quantizing the quantization constants themselves. It sounds crazy, but it saves an extra 0.3 bits per parameter on average!
*   **Paged Optimizers:** Uses Nvidia's Unified Memory to prevent "Out of Memory" (OOM) errors by using your computer's regular RAM as a safety net for the GPU.

---

## 🛠️ Engineering — what the code does

[`train.py`](./train.py):

1.  **Quantization Config:** Sets up `BitsAndBytesConfig` with `load_in_4bit=True`.
2.  **K-Bit Prep:** Uses `prepare_model_for_kbit_training` to stabilize the gradients.
3.  **Adapter Attachment:** Adds LoRA adapters to the 4-bit base.
4.  **Training Loop:** Uses `SFTTrainer` with the `paged_adamw_8bit` optimizer.
5.  **Memory Monitoring:** In a real-world scenario, you would see your VRAM usage stay flat and low compared to Chapter 1 or 2.

### Gotchas
- **No CPU Support:** 4-bit quantization requires specialized CUDA kernels. This will not work on a CPU.
- **No Merging:** You cannot call `model.merge_and_unload()` on a 4-bit model. If you need a merged model, you must train in 16-bit LoRA (Chapter 2) or re-load the base in 16-bit and merge the 4-bit adapter later.
- **Performance Hit:** Training is roughly 20-30% slower than 16-bit LoRA because of the overhead of turning 4-bit weights into 16-bit weights during every step.

---

## 🧪 Research — open questions

- Compare ch2 (LoRA over bf16 base) vs this class (LoRA over 4-bit base) on the same backbone (135M for fairness): how much of an `eval_loss` gap, if any?
- NF4 vs FP4: re-run with `bnb_4bit_quant_type="fp4"`. Same final loss?
- Double quantization adds ~0.4 bits per weight back. With it disabled, do you measure a quality regression or only a memory difference?

---

## 🚀 How to run

```bash
bash courses/course1_finetuning/chapter3_qlora/class1_decoder_qlora/run.sh
```

GPU env required (`env/slm-gpu.yml`); the script will fail loud and clear on a CPU-only env.

## ✔ How to verify

`results/full/<backbone>/course1_finetuning/chapter3_qlora_class1_decoder_qlora/smoltalk/qlora-r16.json`. Expected band (smoke):

| Metric | Passing Range | Meaning |
|---|---|---|
| `train_loss_final` | 0 - 6.0 | Final training loss |
| `eval_loss` | 0 - 6.0 | Held-out NLL on assistant tokens |
| `loss_decreased` | 1 (True) | Sanity check |
| `trainable_ratio_pct` | 0 - 5.0 | LoRA stays under 5% of params |
| `base_in_4bit` | 1 (True) | Sanity check: is the base really 4-bit? |
