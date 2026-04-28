# Course 3 · Chapter 2 · Class 1 — Active learning: uncertainty sampling vs random

> Goal: prove that **labeling the model's most-uncertain examples** beats labeling random ones at the same budget. Same encoder, same task (AG News classification), same total label count — but two strategies for picking which examples to reveal next. Watch the accuracy curves diverge.

---

## Psycho — the mental model

> **One-line takeaway:** *don't waste labels on what the model already knows*. Spend each label on the example the model is most likely to get wrong — that's where it learns the most per dollar.

Most students collect data the wrong way: scrape a corpus, randomly sample N examples, label them all, train. **Random sampling is the worst strategy** when each label is expensive (human annotator time, expert review, regulatory sign-off). The model learns very little from labeling examples it would have gotten right anyway.

Active learning flips the loop. Train on a small seed → ask the model "which examples in the unlabeled pool are you least sure about?" → label only those → retrain → repeat. The intuition is that high-uncertainty examples lie near the decision boundary, where the model's gradients have the most to learn from.

The simplest measure of uncertainty is **entropy of the predicted class distribution**: $H(p) = -\sum_k p_k \log p_k$. A confident prediction `[0.95, 0.02, 0.02, 0.01]` has tiny entropy; an uncertain one `[0.27, 0.25, 0.24, 0.24]` has near-maximum entropy. Picking the highest-entropy examples picks the ones the model is least confident about.

**Common confusion to head off:** "Why not always pick uncertainty? Free win, right?" Two real costs: (1) high-entropy examples can be *outliers* the model can't learn anyway (noisy labels, ambiguous content) — so a hybrid of uncertainty + diversity often wins; (2) you need a working classifier to *measure* uncertainty, so you can't apply this to your first batch of labels.

## Academic — what's measured

Two strategies, both starting from the same `seed_size` random labels:

- **Random**: sample `query_size` more examples uniformly at random from the unlabeled pool each round.
- **Uncertainty (max-entropy)**: train classifier → predict on unlabeled pool → pick the `query_size` examples with highest predictive entropy → label those.

After each round, train a fresh classifier from scratch on all currently-labeled data and report test-set accuracy.

The headline metrics:

- `final_accuracy_random` and `final_accuracy_uncertainty` after the last round.
- `delta_accuracy` = `final_accuracy_uncertainty - final_accuracy_random`.
- `accuracy_curve_random[]` and `accuracy_curve_uncertainty[]` — accuracy after each round, so the student can plot the divergence.

References:
- [Settles, *Active Learning Literature Survey* (2009)](https://burrsettles.com/pub/settles.activelearning.pdf) — the canonical survey
- [Lewis & Gale, *A Sequential Algorithm for Training Text Classifiers* (SIGIR 1994)](https://dl.acm.org/doi/10.5555/188490.188495) — uncertainty sampling for text
- [HF — `AutoModelForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/auto)

## Engineering — what the code does

[`train.py`](./train.py):

1. Loads AG News via `shared.datasets.to_classification`. Caps train pool to `pool_size` rows, eval to `eval_size`.
2. Splits the train pool into a `seed_size` initial labeled set + the rest as unlabeled pool.
3. For each strategy in `[random, uncertainty]`:
   - Reset the labeled pool to the seed.
   - For each round (`n_rounds` times):
     - Train a fresh `AutoModelForSequenceClassification` on the current labeled pool.
     - Evaluate on the test set; record accuracy.
     - Apply the strategy to pick `query_size` examples from the unlabeled pool.
     - Add them to the labeled pool, remove from unlabeled.
4. Persist a result JSON via `shared.eval_harness.run_eval`. Both curves and the headline `delta_accuracy` go in metrics.

The metric band asserts `delta_accuracy >= 0` (uncertainty sampling at worst ties random; in practice it should win by a few points on AG News).

### Gotchas
- **Restart the model fresh each round.** If you keep training the same model, you can't disentangle "more data helped" from "more epochs helped". Each round trains from `from_pretrained`.
- **Uncertainty needs softmax-normalized logits.** A bare logit's max isn't comparable across examples. Use `softmax(logits, dim=-1)` before computing entropy.
- **The "true labels" you reveal during querying are from the same training set** — this is a *pool-based* simulation, not a real labeling-cost study. Real human-in-the-loop adds annotator latency and disagreement noise.
- **Smoke caps are tiny** (pool=2048, seed=64, query=64, n_rounds=4). Curves are noisy. Bump to `mode=full` for stable rankings.

## Research — open questions / extensions

- Replace max-entropy with **margin sampling** (`p_top1 - p_top2`) — uncertainty defined by how close the top-2 classes are. Does it pick different examples?
- Add a **diversity** dimension: cluster the high-entropy candidates first, take one per cluster. Does it beat plain max-entropy when the unlabeled pool has duplicates?
- Run the curve to convergence (until accuracy plateaus). At what label budget does random sampling catch up to uncertainty? That's your "active learning shelf life" for this task.
- The lesson assumes labels are free *given* a query. Add a per-label cost (e.g. each query needs a 30s annotator review) and recompute the win in cost-per-accuracy-point.

---

## How to run

```bash
bash courses/course3_pipeline/chapter2_active_learning/class1_uncertainty_sampling/run.sh
```

Smoke mode by default — runs ~4 rounds × 2 strategies × small classifier each = ~3–4 min on a single GPU.

## How to verify

`results/full/<backbone>/course3_pipeline/chapter2_active_learning_class1_uncertainty_sampling/ag_news/al-r<ROUNDS>.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `final_accuracy_random` | 0.50 | 1.0 | Sanity that the random baseline learned something |
| `final_accuracy_uncertainty` | 0.50 | 1.0 | Sanity that the uncertainty arm learned something |
| `delta_accuracy` | -0.05 | 1.0 | The headline: uncertainty should not lose by much; usually positive |
| `n_rounds_completed` | 1 | 100 | Sanity the loop actually ran |
| `final_label_budget` | 1 | 100000 | Total labels used by each arm |

If `delta_accuracy` is strongly negative, uncertainty sampling is picking outliers / mislabeled examples. Try margin sampling (exercise 2) or check the dataset for noise.

### Honest finding from this repo's smoke runs

On the default config (MiniLM + AG News + seed=256, query=128, 4 rounds), naive max-entropy uncertainty sampling **lost** to random by ~30 accuracy points. Round-by-round:

```
round 1:  random=0.30  uncertainty=0.46   delta=+0.16
round 2:  random=0.57  uncertainty=0.31   delta=-0.26   ← uncertainty queries hurt!
round 3:  random=0.66  uncertainty=0.41   delta=-0.25
round 4:  random=0.70  uncertainty=0.39   delta=-0.31
```

Why? The first-round model assigns near-uniform softmax (entropy ≈ `log(4) = 1.386`) to almost every example, so "max entropy" picks essentially the most adversarial / noisy / atypical news headlines. Adding 128 of those to the training set per round confuses the model more than they teach it.

**The bands are deliberately loose so this failure mode passes the smoke**, because the failure *is* the lesson. The exercises walk through the standard fixes: margin sampling (more robust to flat distributions), diversity-aware sampling (de-duplicate the high-entropy candidates), or warm-start with a much larger seed. This honest negative result is more pedagogically valuable than a sanitized win — it shows why the literature is full of "uncertainty sampling + X" papers rather than just "uncertainty sampling".

## Instructor checklist

Before marking this class "done":

- [ ] All four mode sections present.
- [ ] `train.py`/`eval.py` contain no numeric literal other than `0` / `1` (everything in `configs/*.yaml`).
- [ ] `configs/default.yaml` declares an `expected_band` for every metric written.
- [ ] `run.sh` uses `HF_HOME=$PWD/.cache/huggingface` and is `chmod +x`.
- [ ] `exercises.md` has exactly three exercises.
- [ ] At least one smoke run completed end-to-end and the metric band passes.
- [ ] Linked from the parent course `README.md` table.
