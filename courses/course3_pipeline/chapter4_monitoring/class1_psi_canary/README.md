# Course 3 · Chapter 4 · Class 1 — Monitoring: PSI, canary set, latency

> Goal: surface distribution shift **before accuracy degrades**. Combine an early-warning embedding-drift signal (PSI) with a ground-truth quality signal (canary accuracy) and an operational signal (latency). Watch all three move together as we ramp synthetic out-of-distribution traffic from 0 % to 100 %.

---

## Psycho — the mental model

> **One-line takeaway:** *PSI tells you the input distribution drifted; canary accuracy tells you the model started hurting because of it. You need both.*

A common mistake: monitor only canary accuracy. By the time accuracy drops, real users have already received bad predictions for hours. PSI on the **embeddings of incoming traffic** can flag drift earlier — sometimes long before quality moves — because changes in input distribution are visible in the encoder's feature space without any labels at all.

But PSI alone is also wrong: it can fire on benign distribution changes (the seasonal news cycle, a new client onboarding) that don't actually hurt the model. So you need both:

- **PSI** — early warning, label-free, can have false positives.
- **Canary accuracy** — definitive, label-bound, lags real-world impact.

Together they form a 2×2 matrix: low PSI + high accuracy is "all good"; high PSI + still-high accuracy is "investigate, may be safe"; high PSI + falling accuracy is "alarm, retrain"; low PSI + falling accuracy is "your canary set is stale".

This class shows the third quadrant — PSI rises **and** accuracy falls — by ramping synthetic OOD traffic across monitoring ticks. Final metric assertion: PSI and accuracy are **negatively correlated**, proving PSI is a useful signal here.

**Common confusion to head off:** "Why not just measure accuracy?" Because most production traffic is **unlabeled**. You only have labels for the canary set, which is a small fraction of total volume. PSI works on every request.

## Academic — the metric definitions

**Population Stability Index** between two probability histograms `b` (baseline) and `l` (live):

$$
\mathrm{PSI}(b, l) = \sum_i (l_i - b_i) \log \frac{l_i}{b_i}
$$

PSI is symmetric, non-negative, and goes to 0 as `l → b`. Standard operating thresholds (Wieland & Wallace, 2008; widely reused in ML monitoring):

| PSI value | Interpretation |
|---|---|
| `< 0.10` | no significant change |
| `0.10–0.25` | moderate change, investigate |
| `> 0.25` | significant change, alarm |

We compute PSI on a **low-dimensional projection** of embeddings rather than on the raw 384-dim space, because histograms in 384 dimensions are sample-starved. The projection is the top-N right singular vectors of the centered baseline embedding matrix (PCA via `numpy.linalg.svd` — pure numpy, no sklearn dep).

References:
- [Wieland & Wallace, *Tracking Ecology with Statistical Process Control* (2008)](https://www.taylorfrancis.com/) — origin of the 0.10 / 0.25 thresholds
- [Wang et al., *A Survey on Distribution Shift Detection* (2024)](https://arxiv.org/abs/2401.06513) — modern survey covering PSI, MMD, KS, etc.
- [HF — `SentenceTransformer.encode`](https://www.sbert.net/) — what produces the embeddings we monitor

## Engineering — what the code does

[`train.py`](./train.py) (no training; "train" is preserved for class-folder uniformity):

1. **Baseline build** — encode `baseline.size` AG News rows with the encoder, fit a `n_components`-PC projection, compute the per-bin baseline histogram.
2. **Bootstrap a canary classifier** on a small AG News slice so we have something to evaluate canary accuracy against. (Could also load a pre-trained one; bootstrapping inline keeps the class self-contained.)
3. **For each monitoring tick** in `live.shift_schedule[]`:
   - Construct the live batch by mixing `(1 - ood_frac)` AG News + `ood_frac` Emotion (label-remapped). The schedule typically ramps from 0.0 to 0.8 over the ticks.
   - Encode the live batch, histogram along the projection (using the baseline's bin edges so PSI is comparable), compute PSI vs baseline.
   - Compute canary accuracy on a fixed labeled canary set.
   - Time the encode step for an operational latency reading.
4. Persist a result JSON via `shared.eval_harness.run_eval` with the full per-tick trajectory in `extras.tick_log`.

The metric band asserts `psi_max >= 0.25` (the alarm fired at peak shift) AND `accuracy_psi_correlation <= 0.0` (PSI ↑ correlates with canary accuracy ↓ — i.e. PSI is doing its job).

### Gotchas
- **Use the baseline's bin edges for the live histogram.** Without that, the two histograms are on different binnings and PSI is meaningless. `histogram_along_projection` accepts `edges=...` for this.
- **The first projected coordinate (PC1) is what we histogram.** That's deliberately a 1-D summary; richer (e.g. multivariate Wasserstein) is exercise 3.
- **Latency here is encode-only**, not full inference. For a fair latency monitor you'd time `encode + classify`. We keep it simple to keep the dashboard one JSON.

## Research — open questions / extensions

- Replace PSI with **Maximum Mean Discrepancy (MMD)** with an RBF kernel. Does it pick up the same drift moments? At what cost?
- The current canary set is fixed at startup. **Refresh it periodically** (replace the oldest 25 % each tick with new labeled data) and watch how that affects the PSI/accuracy correlation.
- The class assumes the OOD source (Emotion) — in real life you don't know what the drift looks like. Add an **adversarial OOD** sampler that picks the live batch to maximize PSI while minimizing canary accuracy (find the worst case).
- Plot the (PSI, canary_acc) trajectory over ticks. When it's a clean diagonal, monitoring works. When it's noisy, something is wrong with the projection or the canary distribution.

---

## How to run

```bash
bash courses/course3_pipeline/chapter4_monitoring/class1_psi_canary/run.sh
```

Smoke mode by default — 6 ticks, ~1 min on a single GPU.

## How to verify

`results/full/<backbone>/course3_pipeline/chapter4_monitoring_class1_psi_canary/monitoring/psi-canary-t<TICKS>.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `psi_baseline` | 0.0 | 0.05 | Baseline-against-itself sanity (should be near zero) |
| `psi_max` | 0.05 | 100.0 | Highest PSI across ticks; > 0.25 is the alarm threshold |
| `live_accuracy_baseline` | 0.40 | 1.0 | Sanity: classifier accuracy on the first (in-distribution) tick |
| `live_accuracy_min` | 0.0 | 1.0 | Lowest live accuracy across ticks (informational; falls with PSI) |
| `accuracy_psi_correlation` | -1.0 | 0.0 | The headline: PSI ↑ while live accuracy ↓ |
| `n_ticks_above_alarm` | 0 | 100 | How many ticks crossed PSI > 0.25 |
| `mean_latency_ms` | 0.0 | 600000.0 | Permissive — operational, machines vary |

`extras.tick_log[]` carries per-tick `(ood_frac, psi, live_accuracy, latency_ms)` so you can plot the divergence yourself.

We use **live accuracy** (classifier's accuracy on the labeled live batch) rather than fixed-canary accuracy because a fixed canary set is in-distribution and can't move with input drift. In a real production system you'd label a sample of incoming traffic for this — same idea.

## Instructor checklist

Before marking this class "done":

- [ ] All four mode sections present.
- [ ] `train.py`/`eval.py` contain no numeric literal other than `0` / `1` (everything in `configs/*.yaml`).
- [ ] `configs/default.yaml` declares an `expected_band` for every metric written.
- [ ] `run.sh` uses `HF_HOME=$PWD/.cache/huggingface` and is `chmod +x`.
- [ ] `exercises.md` has exactly three exercises.
- [ ] At least one smoke run completed end-to-end and the metric band passes.
- [ ] Linked from the parent course `README.md` table.
