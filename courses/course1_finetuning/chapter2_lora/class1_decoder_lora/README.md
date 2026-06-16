# Course 1 · Chapter 2 · Class 1 — LoRA fine-tuning of a decoder sLM

> **Goal:** Take SmolLM2-135M-Instruct and fine-tune it using **LoRA (Low-Rank Adaptation)**. You will train only ~0.3% of the parameters, creating a tiny "adapter" that sits on top of a frozen base model.

---

## 🧭 The 5 W's & 1 H (Foundations)

### WHAT are we doing?
We are performing **Parameter-Efficient Fine-Tuning (PEFT)** using a method called **LoRA**.
*   **The Problem:** Full fine-tuning (from Chapter 1) requires updating millions of parameters, which is slow and memory-intensive.
*   **The Solution:** Instead of changing the big weight matrices of the model, we add two much smaller matrices (A and B) next to them.
*   **The Result:** We only train these tiny new matrices. The original model stays exactly as it was.

### WHY are we doing this?
*   **Memory Efficiency:** Because we only update a fraction of the parameters, we need far less GPU memory. This allows you to train larger models on consumer-grade hardware.
*   **Storage Efficiency:** A full model might be several gigabytes. A LoRA "adapter" is typically only a few megabytes. You can store hundreds of specialized adapters for the same base model.
*   **Preventing Forgetting:** Since the base model weights are **frozen**, the model's original "general knowledge" is perfectly preserved. It doesn't "drift" as easily as it does in full fine-tuning.

### WHEN should you use this?
*   Use **LoRA** for almost every fine-tuning task today. It has become the industry standard.
*   Use it when you have limited hardware (e.g., a single 8GB or 12GB GPU).
*   Use it when you need to deploy many specialized versions of a model (e.g., one for each customer) without storing multiple full-size copies.

### WHERE do the "Adapters" go?
The adapters are injected into the **Linear Layers** of the model's Attention mechanism.
1.  Specifically, we target layers like `q_proj` (queries), `k_proj` (keys), and `v_proj` (values).
2.  During a "forward pass," the data flows through the frozen original layer AND the new trainable LoRA matrices simultaneously.
3.  The outputs are summed together. The LoRA matrices act as a "learned correction" to the original model's behavior.

### HOW does it work (The Pipeline)?
1.  **Freeze:** Set `requires_grad = False` for all original model parameters.
2.  **Inject:** Use the `PEFT` library to add the low-rank matrices `A` and `B` into the target layers.
3.  **Train:** Run SFT (Supervised Fine-Tuning) exactly like before, but only the new `A` and `B` weights will receive updates.
4.  **Save:** Only save the tiny `adapter_model.bin`. At inference time, you load the big base model and "apply" this tiny file on top.

---

## 🧠 Psycho — the mental model

> **One-line takeaway:** LoRA freezes the original engine and bolts on a *tiny tunable steering wheel*. The base never moves; only the bolt-on adapter learns.

Full fine-tuning rewrites the original parameters of the model. That's powerful but expensive (you're moving ~135M numbers around) and dangerous (you've now lost the original; if you didn't checkpoint, the pretrained model is gone).

LoRA's bet: **the change you need to apply is low-rank**. The full weight matrix has, say, 576×576 entries, but the *delta* you'd add for your task lives in a tiny rank-16 subspace. So instead of training the matrix, you train two small matrices `A` (16×576) and `B` (576×16); the effective update is `B @ A`. That's ~18× fewer parameters touched.

### The Benefits:
- **Many adapters per base.** One per task, domain, or customer. Each adapter is a few MB.
- **Hot-swap at inference.** `peft.set_adapter("name")` flips the active personality in microseconds.
- **No catastrophic forgetting on the base.** The base's pretrained knowledge is preserved bit-for-bit.
- **Memory headroom.** Train 360M models where full FT would crash your GPU.

**Common confusion to head off:** "If the base is frozen, how does the model learn anything new?" Because at every linear layer, the forward pass is now `(W + BA) @ x` instead of `W @ x`. The `BA` term is your trainable correction — small but in exactly the dimensions that matter.

---

## 🎓 Academic — what's happening

For a target weight matrix $W \in \mathbb{R}^{d \times k}$ (e.g. `q_proj`), LoRA adds:

$$W' = W + \frac{\alpha}{r} B A, \quad A \in \mathbb{R}^{r \times k}, \, B \in \mathbb{R}^{d \times r}$$

with rank $r \ll \min(d, k)$ (we use $r=16$ here; common values 4–64). Only $A$ and $B$ are trained; $W$ stays frozen. $A$ is initialized Kaiming-random, $B$ is initialized to zero so the model starts identical to the base.

**Key Terms for Students:**
*   **Rank (r):** The "width" of the tiny matrices. A higher rank means the adapter has more "brain power" but uses more memory.
*   **Alpha (α):** A scaling factor. We usually set `alpha = 2 * rank` to ensure the adapter's influence is stable during training.
*   **Target Modules:** The specific parts of the model (like `q_proj`) where we bolt on the adapters.

---

## 🛠️ Engineering — what the code does

[`train.py`](./train.py):

1.  **PEFT Setup:** Calls `peft.get_peft_model` with a `LoraConfig`. This transforms the model into a "PeftModel".
2.  **Targeting:** The script automatically finds the right modules to target (usually the Attention projections).
3.  **SFT Training:** Uses `trl.SFTTrainer` just like in Chapter 1. The trainer is smart enough to only update the adapter weights.
4.  **Parameter Reporting:** The logs will show you a "Trainable vs Total" parameter count. You should see a huge reduction (e.g., from 135,000,000 down to ~400,000).
5.  **Lightweight Saving:** We save only the adapters, not the whole 135M model.

### Gotchas
- **Silent No-Ops:** If you target modules that don't exist in your model, LoRA will "succeed" but train 0 parameters. Always check the "Trainable Parameter Count" in the logs.
- **Alpha Scaling:** Don't just increase `r` without thinking about `alpha`. If you change `r` from 16 to 32, usually increase `alpha` to 64.
- **Merging:** You can "merge" the LoRA into the base model to speed up inference, but once you do, you lose the ability to swap it out.

---

## 🧪 Research — open questions

- Sweep `r ∈ {4, 8, 16, 32, 64}` and `alpha = 2r`. Plot `eval_loss vs r`. At what rank does the curve flatten?
- IA³ (`peft.IA3Config`) is even smaller (per-element gating). Does it match LoRA on this task?
- LoRA on `o_proj` only vs all four projections — how big is the gap? (Hint: surprisingly small for some tasks.)

---

## 🚀 How to run

```bash
bash courses/course1_finetuning/chapter2_lora/class1_decoder_lora/run.sh
```

Smoke by default; `MODE=full` for the longer subset.

## ✔ How to verify

`results/full/<backbone>/course1_finetuning/chapter2_lora_class1_decoder_lora/smoltalk/lora-r16.json`. Expected band (smoke):

| Metric | Passing Range | Meaning |
|---|---|---|
| `train_loss_final` | 0 - 6.0 | Final training loss |
| `eval_loss` | 0 - 6.0 | Held-out NLL on assistant tokens |
| `loss_decreased` | 1 (True) | Sanity check |
| `trainable_ratio_pct` | 0 - 5.0 | Confirm LoRA trained < 5% of params |
