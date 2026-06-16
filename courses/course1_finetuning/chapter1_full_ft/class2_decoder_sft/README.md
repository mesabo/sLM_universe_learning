# Course 1 · Chapter 1 · Class 2 — Full SFT of a decoder sLM with TRL

> **Goal:** Take SmolLM2-135M-Instruct and continue instruction-tuning it on a small open chat dataset using `trl.SFTTrainer`. This is the foundational recipe for teaching a model "how to talk."

---

## 🧭 The 5 W's & 1 H (Foundations)

### WHAT are we doing?
We are performing **Supervised Fine-Tuning (SFT)**.
*   **The Data:** We use a subset of **SmolTalk**, a high-quality dataset of conversations. Each example is a list of messages like `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`.
*   **The Model:** We use a **Decoder** (SmolLM2). Unlike encoders, decoders are "generative" machines designed to predict the next word in a sequence.
*   **The Task:** We train the model to predict the assistant's response word-by-word, conditioned on the user's prompt.

### WHY are we doing this?
*   **Behavioral Alignment:** Pretrained models are "document completers." SFT teaches them to behave like assistants. It aligns the model's output style with human expectations.
*   **Domain Specialization:** If you want a model to talk like a doctor, a pirate, or a Python expert, SFT is how you inject that "persona" or "voice."
*   **Format Adherence:** SFT teaches the model to follow instructions (e.g., "Answer in JSON format") and respect chat boundaries.

### WHEN should you use this?
*   Use **SFT** when your base model is "smart" but "wild" (doesn't follow instructions well).
*   Use it when you need to change the **style, tone, or format** of the model's responses.
*   *Note:* Do not use SFT as the primary way to teach the model new facts; use RAG for that!

### WHERE does the "Learning" happen?
Learning happens specifically on the **Assistant's tokens**.
1.  The model sees the System prompt and User prompt.
2.  It uses these as "context" (it doesn't try to learn them).
3.  When it reaches the Assistant's part of the conversation, we calculate the loss for every word it predicts.
4.  **The Mask:** We "mask out" the user and system tokens so the model isn't penalized for how it thinks a user *should* have asked the question—it only cares about how it *answers*.

### HOW does it work (The Pipeline)?
1.  **Chat Templating:** We wrap the conversation in special markers (like `<|im_start|>user\n...<|im_end|>`) so the model knows who is talking.
2.  **Forward Pass:** The model predicts the next token for the entire sequence.
3.  **Label Masking:** We set the "Labels" for the user/system tokens to `-100`. PyTorch ignores any token with a `-100` label during loss calculation.
4.  **Loss Calculation:** We use **Negative Log-Likelihood (NLL)** on the assistant's tokens.
5.  **Optimization:** We update all the weights of the decoder to make the assistant's response more likely.

---

## 🧠 Psycho — the mental model

> **One-line takeaway:** SFT is *"show, don't tell"*. Same loss function as pretraining; you just curate the examples.

Students often expect supervised fine-tuning to be some new kind of training. It isn't — it's exactly the same next-token-prediction loss the model was pretrained with. What changes is **what you let it see** and **which tokens count toward the loss**.

```
[system]
You are a concise teacher.
[user]
Explain LoRA in two sentences.
[assistant]
LoRA inserts low-rank adapters into the attention/FFN projections of a frozen base model, so you only train ~0.1% of params...
```

You train the model to predict each *assistant* token given everything before it. **You do NOT train it to predict user/system tokens** — those are inputs, not targets. TRL's `SFTTrainer` handles this masking automatically when you pass a chat template.

The instinct that helps most: imagine you're teaching by example, not by rule. You don't tell the model *"be concise and helpful"* — you show it 10000 examples of "user → concise helpful answer" and trust gradient descent to extract the pattern. SFT is curation more than coding.

**Common confusion to head off:** "Why does the model still hallucinate after SFT?" Because SFT only teaches *style* and *format*; it doesn't insert new facts. New facts come from pretraining (frozen) or RAG (Course 1 ch6). If your SFT data implies "always give a confident answer", the model learns to do that even on questions it can't answer.

---

## 🎓 Academic — what's happening

The objective is the standard causal LM loss restricted to assistant spans:

$$\mathcal{L}_\text{SFT} = -\frac{1}{|A|} \sum_{(p, t) \in A} \log p_\theta(t \mid p)$$

where $A$ is the set of (prefix, target-token) pairs falling inside an assistant turn.

In practice:
1. `apply_chat_template(messages, add_generation_prompt=False)` renders messages to a single string.
2. The data collator builds `labels` such that every non-assistant token has `label = -100` (ignored by the cross-entropy loss).
3. Backprop teaches the model the assistant continuation conditioned on system+user.

Why Smol talk? Because the user's hardware can fully load SmolLM2-135M and a small SFT run in minutes. We use a tiny subset of [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) — the official dataset that produced SmolLM2-Instruct.

References:
- [TRL — `SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer)
- [SmolLM2 paper](https://huggingface.co/papers/2502.02737) (and the model card)
- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — the SFT pattern

---

## 🛠️ Engineering — what the code does

[`train.py`](./train.py):

1.  **Model Loading:** Loads SmolLM2-135M-Instruct via `shared.backbones.load_backbone`.
2.  **Dataset Preparation:** Loads a tiny smoltalk subset.
3.  **SFTTrainer:** Wraps the model and data. We pass a `formatting_func` that converts our message lists into strings that the model can understand using the chat template.
4.  **Training:** The model iterates through the data, learning the "style" of the assistant responses.
5.  **Evaluation:** We measure the "Loss" on a held-out set. A lower loss means the model is getting better at predicting what the *real* assistant would have said.

### Gotchas
- **CPU Slowness:** SFT on a decoder is heavy. Even at 135M parameters, CPU training is very slow. Always prefer a GPU for this class.
- **Padding Token:** Decoders often don't have a padding token by default. We reuse the `eos_token` (End Of Sentence) as a pad token to keep the math working.
- **Precision:** We use `bf16` or `fp16` on GPUs to speed up training and save memory.

---

## 🧪 Research — open questions

- Replace smoltalk with `tatsu-lab/alpaca` (older, lower quality). Does eval log-prob improve on the smoltalk held-out anyway? What does that tell you about transfer between SFT corpora?
- Pure SFT degrades safety-tuned behaviors. Try a "harmful" prompt before and after SFT — what changes?
- The `-Instruct` SmolLM2 was *already* SFT'd by HuggingFaceTB. Why do *more* SFT? (Hint: domain shift, style drift, instruction-following on YOUR distribution.)

---

## 🚀 How to run

```bash
bash courses/course1_finetuning/chapter1_full_ft/class2_decoder_sft/run.sh
```

Smoke mode by default (~50 steps on a 1024-row subset). Use `MODE=full` for the full subset.

## ✔ How to verify

`results/full/<backbone>/course1_finetuning/chapter1_full_ft_class2_decoder_sft/smoltalk/sft.json`. Expected band (smoke):

| Metric | Passing Range | Meaning |
|---|---|---|
| `train_loss_final` | 0 - 6.0 | Final training loss (raw NLL) |
| `eval_loss` | 0 - 6.0 | Held-out NLL on assistant tokens |
| `loss_decreased` | 1 (True) | Sanity check: Did the model actually learn something? |
