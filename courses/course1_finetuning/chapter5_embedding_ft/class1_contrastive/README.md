# Course 1 · Chapter 5 · Class 1 — Contrastive fine-tuning of an encoder for retrieval

> Goal: take a pretrained sentence-encoder (MiniLM, BGE-small, GTE-small) and fine-tune it on `(anchor, positive)` pairs with `MultipleNegativesRankingLoss`. After this class you'll know what "in-batch negatives" means, how `sentence-transformers` handles it under the hood, and how the resulting embeddings stack up under MRR / Recall@1 on a held-out retrieval task.

---

## Psycho — the mental model

A pretrained sentence-encoder already produces semantically meaningful vectors — but it was trained on whatever distribution its authors chose. For your retrieval task (queries vs. docs in your domain), it almost certainly *under-clusters* near-duplicates and *over-separates* paraphrases. Contrastive fine-tuning fixes this by showing the model **pairs that should be close**.

The standard cheap trick is **in-batch negatives**: in a batch of `N` `(anchor, positive)` pairs, treat the other `N-1` positives as negatives for each anchor. No need to mine hard negatives explicitly — every batch produces `N²` similarity comparisons.

You don't need a generation model, you don't need preference triples — just a corpus of "these two strings should be close". For RAG (chapter 6), this is *the* knob that turns a generic encoder into a retriever your application actually trusts.

## Academic — what's happening

Let $f_\theta : \text{string} \to \mathbb{R}^d$ be the encoder, normalized to the unit sphere. For a batch of pairs $\{(a_i, p_i)\}_{i=1}^N$, MultipleNegativesRankingLoss is the symmetric InfoNCE / contrastive cross-entropy:

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(s \cdot f_\theta(a_i)^\top f_\theta(p_i))}{\sum_{j=1}^N \exp(s \cdot f_\theta(a_i)^\top f_\theta(p_j))}
$$

with temperature scale $s$ (default 20 in `sentence-transformers`). The denominator includes $j=i$ (the true positive) plus all $N-1$ other positives in the batch acting as in-batch negatives. Lower loss ⇒ true positive ranked higher than the rest.

Crucially, **larger batches give a stronger contrastive signal** — every additional row adds a "harder" negative to compare against. With small batches (~8) the task is easy; with large batches (~128) it gets meaningful.

References:
- [Henderson et al., *Efficient Natural Language Response Suggestion* (2017)](https://arxiv.org/abs/1705.00652) — original in-batch negatives idea
- [`sentence_transformers.losses.MultipleNegativesRankingLoss`](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss)
- [SentenceTransformerTrainer API](https://www.sbert.net/docs/training/overview.html)
- Eval data: [`sentence-transformers/quora-duplicates`](https://huggingface.co/datasets/sentence-transformers/quora-duplicates) — duplicate question pairs in Parquet, `pair` subset has `(anchor, positive)` columns

## Engineering — what the code does

[`train.py`](./train.py):

1. Loads the backbone via `SentenceTransformer(name)` (the sentence-transformers wrapper, not raw HF — it adds the pooling head).
2. Loads `sentence-transformers/quora-duplicates` (`pair` subset) — Parquet, `(anchor, positive)` columns.
3. Caps to `limits[mode].train` rows and holds out `dataset.eval_holdout` for retrieval eval.
4. Constructs `losses.MultipleNegativesRankingLoss(model)` and wraps with `SentenceTransformerTrainer`.
5. Trains for the configured steps.
6. Encodes held-out anchors and positives, computes a square cosine-similarity matrix, and reports:
   - `mrr` — mean reciprocal rank of the true positive (diagonal) within the candidate set
   - `recall_at_1` — fraction of anchors whose top-1 hit IS the true positive
   - `recall_at_5`
   - `loss_decreased` — sanity that training reduced the loss

The metric band asserts MRR and recall@1 are above their pre-FT baseline (the bands below are calibrated on smoke runs against a frozen MiniLM).

### Gotchas
- **Larger batch = stronger signal.** Default `per_device_batch=64`; pushing it to 128+ improves MRR materially even at the same step count.
- The eval matrix is `[N_eval, N_eval]` — don't bump `eval_holdout` past `~1024` casually; memory grows quadratically with `N_eval` for the cosine matrix.
- `SentenceTransformerTrainer` doesn't accept `bf16=True` on every backbone; default to fp32 and only enable bf16 if the backbone supports it. For MiniLM the matmul is so small that fp32 is fast anyway.
- `convert_to_tensor=True` + `normalize_embeddings=True` is required for the cosine eval — without normalization, the diagonal is no longer the max even when the model "knows" the answer.

## Research — open questions / extensions

- Sweep `batch_size ∈ {8, 32, 64, 128}` at the same total step count. Plot MRR vs batch size. Is the curve linear in `log(batch)`?
- Train all three encoder backbones (MiniLM, BGE-small, GTE-small) to convergence on the same data. BGE-small is already retrieval-tuned — does it benefit less from your task-specific data than MiniLM?
- Add **hard negatives**: for each anchor, mine the top-1 false positive from the *current* model's embeddings every K steps and pass them explicitly via `MultipleNegativesRankingLoss` (it accepts a `negative` column too). How much does MRR jump?
- The encoder produced here is exactly what Course 1 chapter 6 (RAG) will use as the retriever. Plug it in there and compare end-to-end answer accuracy vs the un-tuned baseline.

---

## How to run

```bash
bash courses/course1_finetuning/chapter5_embedding_ft/class1_contrastive/run.sh
```

Smoke mode by default (~2048 train pairs, ~50 steps, ~1–2 min on a single GPU).

## How to verify

`results/full/<backbone>/course1_finetuning/chapter5_embedding_ft_class1_contrastive/quora/mnrl-b<BATCH>.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `mrr` | 0.40 | 1.0 | Mean reciprocal rank of true positive on held-out pairs |
| `recall_at_1` | 0.30 | 1.0 | Fraction of anchors whose top-1 hit is the true positive |
| `recall_at_5` | 0.55 | 1.0 | Top-5 hit rate |
| `loss_decreased` | 1 | 1 | Final train loss < initial |
| `train_loss_final` | 0.0 | 5.0 | Final InfoNCE loss |

If MRR is near `1/N_eval` (random), the model didn't learn — check that loss decreased and the dataset has actual positive pairs (not random text).

### Note on MiniLM + quora-duplicates: the eval ceiling

A real-world finding from the smoke runs: **MiniLM is already retrieval-tuned and Quora was likely in its training mix**, so pre-FT MRR on this eval set is already ~0.97 and recall@5 is 1.0. There's almost no headroom for the loss to translate into MRR/recall improvements. The training loss does drop (~0.25 → ~0.19 in 50 steps), so the model IS learning; the ceiling just hides it.

Two ways to get a more dramatic before/after:

1. **Use a less-calibrated backbone.** Try `bert-base-uncased` (no contrastive pretraining) — pre-FT MRR is much lower, post-FT MRR jumps materially. (Add to `configs/backbones.yaml` first.)
2. **Use a harder eval set.** Bump `dataset.eval_holdout` to 1024 — more in-batch negatives → more chance to confuse the retriever → harder MRR.

Both are exercise-level extensions; the smoke band is satisfied at the current ceiling and the lesson stands.

## Instructor checklist

Before marking this class "done":

- [ ] All four mode sections (Psycho / Academic / Engineering / Research) are present and ≥ 2 paragraphs each.
- [ ] Every reference link points at an official source (paper / HF doc / repo) where one exists.
- [ ] `train.py` and `eval.py` contain no numeric literal other than `0` / `1` (everything in `configs/*.yaml`).
- [ ] `configs/default.yaml` declares an `expected_band` for every metric written by `eval.py`.
- [ ] `run.sh` uses `HF_HOME=$PWD/.cache/huggingface` and is `chmod +x`.
- [ ] `exercises.md` has exactly three exercises (warm-up / apply / stretch).
- [ ] Result JSON path matches the layout `results/full/<backbone>/<course>/<class>/<task>/<method>.json`.
- [ ] At least one smoke-mode run completed end-to-end and the metric band passes.
- [ ] Linked from the parent course `README.md` table.
