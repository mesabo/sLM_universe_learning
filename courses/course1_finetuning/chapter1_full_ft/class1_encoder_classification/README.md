# Course 1 · Chapter 1 · Class 1 — Full fine-tuning an encoder for classification

> Goal: take MiniLM (or any encoder backbone), bolt a linear classification head on top, and train every parameter on AG News. This is the most "pre-2023" recipe and the cleanest baseline against which we'll compare every PEFT method in chapters 2–3.

---

## Psycho — the mental model

A pre-trained encoder gives you **already-meaningful sentence vectors**. To use it for a task, you only need to teach a small head to map "vector → label". When you do *full* fine-tuning, you also let the encoder itself drift to make those vectors more discriminative for *your* task.

Trade-off:
- **Head only** (a.k.a. linear probe): cheap, never forgets the base, often worse on the task.
- **Full fine-tuning**: best task accuracy, but you've now created a single-task model — the original encoder is gone unless you saved it.

Course 2 will show how to keep both worlds (param isolation, LoRA, replay).

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
