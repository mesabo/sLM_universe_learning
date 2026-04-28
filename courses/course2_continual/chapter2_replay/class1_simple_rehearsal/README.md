# Course 2 · Chapter 2 · Class 1 — Simple rehearsal: the cheapest defense against forgetting

> Goal: take the AG News → Emotion sequence from chapter 1 and add the simplest possible defense — when training on Task B, mix in a sample of Task A. Watch BWT improve from −0.34 (chapter 1's catastrophic baseline) toward zero as you turn up the replay ratio.

---

## Psycho — the mental model

Catastrophic forgetting comes from a one-sided update: gradient descent on Task B has no incentive to preserve Task A. The simplest fix is to **show the model some of Task A while it's learning Task B** — so the loss has a stake in both. That's *rehearsal* (also called *experience replay* in RL-flavored CL papers).

You can think of it like a student preparing for the history exam who keeps a few chemistry flashcards on the desk. Even a tiny rehearsal fraction prevents the worst of the forgetting; the cost is a slightly slower history score, and the requirement that you actually *kept the chemistry data*.

When does rehearsal NOT work?
- When you can't store Task A data (privacy, license, deletion mandate). That's where regularization (chapter 3, EWC/MAS) and parameter isolation (chapter 4, LoRA-per-task) earn their keep.
- When tasks are huge and storage budget matters; then you need a *replay buffer* with selection (reservoir sampling, herding, gradient-based selection — all in the literature).

For this class: tiny tasks, no storage constraint, vanilla rehearsal.

## Academic — what's happening

Let $D_A, D_B$ be the training sets for Task A and Task B, and let $r \in [0, 1]$ be the **replay ratio** — the fraction of Task A samples appended to $D_B$ during the second training stage:

$$
D_B^{(\text{rehearsal})} = D_B \;\cup\; \mathrm{Sample}\!\left(D_A, \lfloor r \cdot |D_B| \rfloor\right)
$$

The model is then trained on $D_B^{(\text{rehearsal})}$ as usual (single combined dataset, single optimizer). The expected effect:

- $r = 0$: the chapter 1 baseline. BWT strongly negative (~−0.34 in our smoke).
- $r = 0.1$ – $0.25$: BWT closer to zero, with negligible cost on Task B accuracy.
- $r = 0.5$ – $1.0$: BWT near zero (sometimes positive — joint training!), but training is now ~half multitask, which dilutes the "continual" framing.

Vanilla rehearsal is the baseline every continual-learning paper compares against; it remains a strong baseline. References:
- [Robins, *Catastrophic forgetting, rehearsal and pseudorehearsal* (Connection Science 1995)](https://doi.org/10.1080/09540099550039318) — the original rehearsal idea.
- [Rolnick et al., *Experience Replay for Continual Learning* (NeurIPS 2019)](https://arxiv.org/abs/1811.11682)
- [GEM (Lopez-Paz & Ranzato, NeurIPS 2017)](https://arxiv.org/abs/1706.08840) — uses an episodic memory + gradient projection; rehearsal is the simpler subset.

## Engineering — what the code does

[`train.py`](./train.py) reuses chapter 1's pipeline but adds one knob: `replay.ratio`. Concretely:

1. Stage 0 — eval both tasks (random init).
2. Stage 1 — train on Task A only (same as chapter 1).
3. **Stage 2 — train on a *mixed* dataset**: Task B's training set, augmented with `int(ratio * |B|)` randomly-sampled rows from Task A. Construction is via `datasets.concatenate_datasets`. Re-eval both tasks.
4. Compute BWT / FWT / avg-acc through `shared.continual.summarize`, exactly like chapter 1.

The result JSON's `expected_band` asserts that **BWT improves over the chapter-1 baseline** when `ratio > 0`, so the lesson reproduces. With `ratio = 0` the band falls back to chapter 1's "forgetting did happen" assertion.

### Gotchas
- Set `seed` in `dataset.shuffle` when sampling from Task A — otherwise the rehearsal subset varies run-to-run and BWT becomes noisy.
- `concatenate_datasets` requires identical features. Both tasks already share `text` + `label` after our remapping, so the concat is clean.
- Don't sample rehearsal data *before* Stage 1 — you want to draw fresh examples each time you compose a rehearsal set.
- If `ratio >= 1.0`, you're effectively doing joint training on $D_A \cup D_B$, which is the gold-standard upper bound for CL (no forgetting *possible* because both tasks are seen jointly). Useful sanity check.

## Research — open questions

- Plot BWT vs ratio (the `lora_rank_sweep`-style grid in `grids/replay_ratio_sweep.yaml` does this for you). Is the curve monotone? Does it plateau?
- Replace random sampling with **class-balanced** sampling (equal samples per Task A label). Does BWT improve at the same memory budget?
- Try **reservoir sampling** as the buffer (fixed-size, maintained online) instead of "store all of A". When does the difference matter?
- The "joint training" upper bound is $r = 1.0$ here. Train an oracle that sees A and B jointly from the start (no chapter-1 stage). How does its average accuracy compare to your best replay run?

---

## How to run

```bash
bash courses/course2_continual/chapter2_replay/class1_simple_rehearsal/run.sh
# or sweep ratio across the GPU shards:
bash shared/sharded_run.sh courses/course2_continual/grids/replay_ratio_sweep.yaml
```

Smoke mode by default (~2048 samples per task, 2 epochs each, single ratio). The sweep grid runs 4 ratios × 1 backbone × 1 seed in parallel.

## How to verify

`results/full/<backbone>/course2_continual/chapter2_replay_class1_simple_rehearsal/two_task/replay-r<RATIO>.json` per (ratio, backbone, seed). Expected band (smoke, default `ratio=0.25`):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `avg_accuracy` | 0.55 | 1.0 | Mean of both tasks' final acc — should *beat* ch1 (~0.52) |
| `acc_A_after_A` | 0.70 | 1.0 | Sanity: A really was learned |
| `acc_B_after_B` | 0.40 | 1.0 | Sanity: B really was learned (with replay diluting it) |
| `acc_A_after_B` | 0.55 | 1.0 | The headline: A retained much better than ch1 (~0.46) |
| `BWT` | -0.30 | 0.10 | Closer to zero than ch1's -0.34 |
| `replay_ratio` | 0.10 | 1.0 | Sanity: ratio actually applied |

For `ratio=0` you should reproduce chapter 1's numbers exactly (same seed, same model). For `ratio>=0.5` BWT should be near zero or positive.

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
