# Course 2 · Chapter 3 · Class 1 — Elastic Weight Consolidation (EWC)

> Goal: stop catastrophic forgetting **without storing any Task A data** (unlike chapter 2's replay). Train MiniLM on AG News, compute the diagonal Fisher information matrix on Task A, then add a quadratic penalty to Task B's loss that "anchors" the parameters most important for Task A. Measure BWT against the chapter 1 baseline (−0.34) and chapter 2's replay (+0.02).

---

## Psycho — the mental model

Replay (ch2) prevents forgetting by **showing the model some of Task A while it learns Task B**. EWC prevents forgetting differently: it lets the model train freely on Task B but **adds a spring** pulling each parameter back toward its Task-A-trained value. The spring is stiffer for parameters that were *important* for Task A.

How do we know which parameters were important? The **Fisher information matrix**: roughly, "how much does Task A's loss change when we wiggle this parameter?" Big Fisher → important → strong spring. Tiny Fisher → unimportant → weak spring (parameter free to repurpose for B).

**Pros over replay:** no need to store Task A data. Useful when:
- Privacy / license / deletion mandate forbids keeping data.
- You only have model weights (not training data) for the prior task.

**Cons:** weaker than replay in practice; hyperparameter `λ` (penalty strength) is a knob that varies dramatically across tasks; requires extra GPU passes to compute Fisher.

## Academic — what's happening

Let $\theta_A^*$ be the model after training Task A. The diagonal Fisher information matrix $F$ approximates the Hessian of the negative log-likelihood at $\theta_A^*$:

$$F_i = \mathbb{E}_{(x, y) \sim D_A}\!\left[\,\left(\frac{\partial \log p_\theta(y \mid x)}{\partial \theta_i}\right)^{\!2}\,\right]_{\theta = \theta_A^*}$$

EWC's modified loss when training on Task B is:

$$\mathcal{L}_\text{EWC}(\theta) = \mathcal{L}_B(\theta) \;+\; \frac{\lambda}{2}\sum_{i} F_i \,\bigl(\theta_i - \theta_{A,i}^*\bigr)^{\!2}$$

The penalty term is the diagonal-Gaussian negative-log-prior centered at $\theta_A^*$ with precision $\lambda F$. Intuitively:

- For parameters with large $F_i$, the penalty grows rapidly as we move away — the model is reluctant to change them.
- For parameters with small $F_i$, the penalty is weak — the model is free to update them for Task B.

References:
- [Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (PNAS 2017)](https://www.pnas.org/doi/10.1073/pnas.1611835114) — the EWC paper
- [Schwarz et al., *Progress & Compress* (ICML 2018)](https://arxiv.org/abs/1805.06370) — online EWC variant
- [HF `Trainer.compute_loss` docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.compute_loss) — where we plug the EWC penalty in

## Engineering — what the code does

[`train.py`](./train.py) reuses chapter 1's pipeline but inserts an EWC step between training Task A and training Task B:

1. **Stage 0** — eval both tasks on the as-loaded model.
2. **Stage 1** — train Task A normally with HF `Trainer`.
3. **Compute Fisher** on Task A — `_compute_fisher_diagonal(model, tokenizer, task_a_eval, n_samples)`. We use the cross-entropy loss over Task A's *eval* split (a subset is fine; controlled by `ewc.fisher_n_samples`). For each batch, the Fisher contribution is `(grad of log-prob).pow(2).sum_over_batch`. Accumulated then averaged.
4. **Snapshot $\theta_A^*$** — clone every parameter's current value into `theta_star`.
5. **Stage 2** — train Task B with a custom `EWCTrainer` whose `compute_loss` adds `λ/2 * Σ F_i * (θ_i - θ_A_i)²` to the normal cross-entropy.
6. Re-eval both tasks; build a `continual.History`; compute BWT/FWT/avg-acc and write the result JSON.

The expected band asserts BWT improves over chapter 1's catastrophic baseline.

### Gotchas
- Fisher is **per-parameter**, full size of the model — a few hundred MB for typical encoders. We hold it in fp32 on the same device as the model.
- The Fisher computation must use **labels = predictions** (or true labels — both common; we use true labels for stability on small data).
- `λ` is the dominant knob. Too small → no protection. Too large → model can't learn B at all. Default 1000 works for MiniLM on this pair; expect to tune.
- We deliberately compute Fisher on the **eval** split, not the training split — using train data leaks the train distribution into both stages.

## Research — open questions

- Sweep `λ ∈ {0, 100, 1000, 10000}`. Plot BWT and `acc_B_after_B` vs λ. Where's the knee?
- Compute Fisher on Task A's *training* split instead. Does BWT change much? (Should be similar in expectation; numerically depends on size.)
- Online EWC (Schwarz et al.): instead of snapshotting `θ_A` once, maintain a running anchor. How would you implement that with multi-task chains > 2?
- Compare EWC vs replay (ch2) on the same `seed`. Which approach wins in (BWT, `acc_B_after_B`) Pareto?

---

## How to run

```bash
bash courses/course2_continual/chapter3_regularization/class1_ewc/run.sh
```

Smoke mode by default (~2048 train per task, 2 epochs each, λ=1000). Runs in ~2 min on a single GPU; computing Fisher adds ~10–20 s.

## How to verify

`results/full/<backbone>/course2_continual/chapter3_regularization_class1_ewc/two_task/ewc-l<LAMBDA>.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `avg_accuracy` | 0.55 | 1.0 | Mean across both tasks at the final stage |
| `acc_A_after_A` | 0.65 | 1.0 | Sanity: A really was learned |
| `acc_B_after_B` | 0.30 | 1.0 | Sanity: B really was learned (penalty allowed it) |
| `acc_A_after_B` | 0.50 | 1.0 | Headline: A retained better than ch1 (~0.46) |
| `BWT` | -0.30 | 0.10 | Negative but smaller in magnitude than ch1's -0.34 |
| `ewc_lambda` | 0.0 | 10000000.0 | Sanity: λ actually applied |
| `fisher_mean` | 0.0 | 1000000.0 | Sanity: Fisher computed (non-zero) |

If BWT is no better than ch1, λ is probably too small. If `acc_B_after_B` cratered, λ is too large.

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
