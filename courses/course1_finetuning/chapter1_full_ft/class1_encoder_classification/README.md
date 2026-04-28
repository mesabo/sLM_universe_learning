# Course 1 · Chapter 1 · Class 1 — Full fine-tuning an encoder for classification

> Goal: take MiniLM (or any encoder backbone), bolt a linear classification head on top, and train every parameter on AG News. This is the most "pre-2023" recipe and the cleanest baseline against which we'll compare every PEFT method in chapters 2–3.

---

## Psycho — the mental model

> **One-line takeaway:** the pretrained encoder is a *gift of already-meaningful vectors* — full fine-tuning teaches it new tricks at the cost of forgetting old ones.

When students see "fine-tune the encoder + classification head", they often miss that **the encoder is doing 99% of the work and the head is doing 1%**. A pretrained encoder gives you sentence vectors that already cluster news-style text apart from review-style text, even before you've shown it a single label. The head just learns to draw the decision boundary in the existing vector space. *Most* of the accuracy comes from the encoder; the head is the cheap part.

The trade-off has two clean ends:

- **Head-only training (linear probe)**: cheap, fast, the original encoder is preserved bit-for-bit. Usually a few accuracy points behind full fine-tuning. The right choice when you'll deploy the encoder for several tasks.
- **Full fine-tuning** (this class): best per-task accuracy, but you've now created a single-task model — the encoder has drifted. If you didn't save the original weights, they're gone.

Course 2 is the whole story of how to keep both worlds — replay, EWC, LoRA-per-task. For now, just notice the trade-off and pick the regime your application actually needs.

**Common confusion to head off:** "Why bother with full FT if linear probe is so close?" Two reasons: (1) on out-of-distribution test data, the FT'd encoder's vectors are usually better-separated, and (2) at the larger end of "small" models, FT can still buy 5–10 accuracy points. Try both at exercise 2 and form your own intuition.

## Academic — what's happening

We minimize cross-entropy on `(text, label) ∈ AG News`:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log p_\theta(y_i \mid x_i)$$

with $p_\theta = \mathrm{softmax}(W h_\theta(x))$, where $h_\theta$ is the pooled encoder output and $W$ is a new $D \times K$ classification head ($K=4$ for AG News).

`AutoModelForSequenceClassification` does exactly this: it loads the encoder, adds a `nn.Linear(hidden_size, num_labels)` head, and applies the right pooling for the architecture (CLS for BERT-family, mean for sentence-transformers when wrapped — but here we use BERT-style CLS pooling on MiniLM through HF's API).

References:
- AG News: [Zhang et al., 2015](https://arxiv.org/abs/1509.01626)
- [HF — `AutoModelForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification)
- [HF — `Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer)

## Engineering — what the code does

[`train.py`](./train.py):

1. Loads the AG News dataset via `shared.datasets.to_classification`.
2. Loads the encoder backbone *with a 4-class head* via `AutoModelForSequenceClassification.from_pretrained(name, num_labels=4)`.
3. Tokenizes with truncation/padding to `max_len`.
4. Trains with `transformers.Trainer` for the configured epochs.
5. Evaluates on the test split, computes accuracy + macro-F1 via `shared.training.classification_metrics`.
6. Persists a result JSON via `shared.eval_harness.run_eval` and asserts the metric band.

### Why `Trainer` and not a custom loop
For a textbook supervised-classification task, `Trainer` is the right choice — it handles the data collator, mixed precision, gradient accumulation, evaluation, and checkpointing for free. We outgrow it in chapter 4 (DPO) where TRL takes over.

### Gotchas
- The HF cache is project-local (`HF_HOME=.cache/huggingface`). Don't override it.
- `num_labels=4` MUST match the dataset; mismatch silently corrupts training.
- Encoder backbones may not have a `pad_token`; `Trainer` handles padding via the collator, but if you write a custom loop you'll need `tokenizer.pad_token = tokenizer.eos_token` or a real `[PAD]`.

## Research — open questions

- AG News is "easy" — accuracy >0.92 is reachable in 1 epoch. What's the *gap* between MiniLM and BGE-small / GTE-small here? Run all three and compare.
- Linear-probe baseline: freeze the encoder, train only the head. How much accuracy do you lose? (You'll need ~30s on CPU.)
- The default `pooler_output` for BERT-style encoders is the `[CLS]` representation. For MiniLM, `sentence-transformers` uses mean pooling. Why doesn't HF's `AutoModelForSequenceClassification` use mean pooling here? (Hint: it follows the architecture's default.)

---

## How to run

```bash
bash courses/course1_finetuning/chapter1_full_ft/class1_encoder_classification/run.sh
```

The script runs a smoke (1 epoch, 1024 train + 512 eval samples) first; pass `MODE=full` to run the full split.

## How to verify

`results/full/<backbone>/course1_finetuning/chapter1_full_ft_class1_encoder_classification/ag_news/full-ft.json`. Expected band (smoke):

| Metric | Lo | Hi |
|---|---|---|
| `accuracy` | 0.70 | 1.0 |
| `f1_macro` | 0.65 | 1.0 |

For the full split, lift to `[0.88, 1.0]` — adjust the YAML if your hardware can do better.
