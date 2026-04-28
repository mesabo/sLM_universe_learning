# Course 2 · Chapter 1 · Class 1 — Measuring catastrophic forgetting

> Goal: see catastrophic forgetting happen with your own eyes. Train MiniLM (with a 4-class head) sequentially on AG News then `dair-ai/emotion`, and watch the AG-News test accuracy crater after the Emotion fine-tune. Compute BWT / FWT / average accuracy — the three numbers every continual-learning paper reports.

---

## Psycho — the mental model

> **One-line takeaway:** the model's parameters are a *finite, shared scratchpad*. Gradient descent on Task B has no incentive to preserve what was written there for Task A — so it overwrites.

Imagine a student who studies for the chemistry exam, aces it, then immediately crams for the history exam — and on Monday's chemistry retake they fail. The neurons that encoded chemistry got rewired for history. That is *catastrophic forgetting*. There's nothing pathological about it: gradient descent on Task B's loss only "knows" Task B; it has no signal that says "and please don't break Task A". Without that signal, the parameters drift toward whatever Task B rewards.

A few nuances students often miss:

- Forgetting is **task-dependent.** Similar tasks (e.g. sentiment-vs-sentiment) interfere less than dissimilar ones (news classification vs question typing). The more the two tasks need different decision boundaries, the more drift you see.
- Forgetting is **recipe-dependent.** Full fine-tuning forgets aggressively; LoRA forgets less (smaller delta); replay forgets even less (Task A is back in the loop); EWC penalizes drift in *important* parameters; isolation eliminates it. Each of those is a future chapter in this course.
- Forgetting is **measurable.** If you can't put a number on it, you can't fix it. That's what this class delivers — the BWT / FWT / avg-acc trio that every continual-learning paper reports.

**Common confusion to head off:** "Couldn't we just train on A and B together (joint training)?" Yes — and that's the upper bound. Continual learning is the constraint where you *can't* (e.g. Task B's data arrives later, or Task A's data was deleted). Otherwise, joint training wins.

## Academic — what's being measured

Following Lopez-Paz & Ranzato (NeurIPS 2017), let $R \in [0,1]^{(T+1) \times T}$ be the accuracy matrix where $R_{k,i}$ is the test accuracy on task $i$ after stage $k$. Stage $0$ is *before any training* (the model as loaded). Stage $k > 0$ is *after training task $k-1$*.

For $T = 2$ tasks (this class), the three headline metrics are:

$$
\begin{aligned}
\mathrm{Avg\text{-}Acc} &= \tfrac{1}{T}\sum_{i=0}^{T-1} R_{T,i} \\
\mathrm{BWT} &= \tfrac{1}{T-1}\sum_{i=0}^{T-2} \big(R_{T,i} - R_{i+1,i}\big) \\
\mathrm{FWT} &= \tfrac{1}{T-1}\sum_{i=1}^{T-1} \big(R_{i,i} - b_i\big)
\end{aligned}
$$

where $b_i$ is a baseline (default: stage-0 accuracy on task $i$, i.e. the as-loaded model). **Negative BWT means catastrophic forgetting.** Positive BWT — rare — means later tasks improve earlier-task performance.

Why this dataset pair? **AG News (4-class news topics) and `dair-ai/emotion` (6 emotion classes)** share *no domain overlap* (news headlines vs first-person emotional statements) and we use a single 4-output classification head, so the forgetting signal is sharp. We map Emotion down to four labels (sadness / joy / love / anger), dropping `fear` and `surprise`, so the heads align.

References:
- [Lopez-Paz & Ranzato — *Gradient Episodic Memory for Continual Learning* (NeurIPS 2017)](https://arxiv.org/abs/1706.08840) — defines BWT / FWT
- [Kemker et al. — *Measuring catastrophic forgetting in neural networks* (AAAI 2018)](https://arxiv.org/abs/1708.02072)
- [HF — `AutoModelForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/auto)

## Engineering — what the code does

[`train.py`](./train.py) runs the following pipeline:

1. Load both tasks via the HF `datasets` library (AG News and `dair-ai/emotion`), capping sizes per `mode`. Emotion's six labels are remapped to `{0..3}` (fear and surprise dropped) to share a 4-output head with AG News.
2. Load the encoder backbone with `AutoModelForSequenceClassification(num_labels=4)`.
3. **Stage 0** — Evaluate accuracy on each task → `R[0]`.
4. **Stage 1** — Fine-tune on Task A (AG News) for `epochs_per_task` epochs via HF `Trainer`. Re-evaluate both tasks → `R[1]`.
5. **Stage 2** — Continue training on Task B (Emotion) without resetting weights. Re-evaluate both tasks → `R[2]`.
6. Build a `continual.History`, compute `BWT`, `FWT`, average accuracy via `shared.continual.summarize`.
7. Persist a result JSON via `shared.eval_harness.run_eval` and assert the metric band — in this class the band asserts that **forgetting did happen** (BWT below a threshold) so the lesson is reproducible.

### Gotchas
- **The head is shared.** Emotion's class 0 (sadness) is *not* AG News's class 0 (World); they're remapped semantically. That's intentional — the dramatic forgetting comes from the head's rewriting, plus the backbone drifting toward Emotion's distribution.
- **Eval first, train later, eval again** — without the stage-0 eval you can't compute FWT.
- **`Trainer` mutates the model in place.** Stages 1 and 2 reuse the same `model` object; do NOT `from_pretrained` again between tasks (that would be a fresh restart, not continual learning).
- AG News (~120 k rows) is much larger than Emotion (~16 k rows). Smoke mode caps both at 512 train samples each — runs in ~1 min on a single GPU.

## Research — open questions / extensions

- Replace Emotion with a **closer-domain** Task B (e.g. `Yelp/yelp_review_full` star ratings 0–4 mapped to 4 buckets). How much less BWT do you measure? This is the "task similarity" axis.
- Re-run with `freeze_backbone: true` (only the head trains). Does forgetting go away? (Hint: yes — but accuracy on Task A also caps lower. The classic capacity-vs-stability trade-off.)
- Swap the encoder to BGE-small or GTE-small. Does forgetting differ across encoders of similar size? Is it correlated with how "specialized" the encoder is to retrieval pre-training?
- Add a third task and watch BWT compound. Do later tasks erase earlier tasks faster, or does the drift saturate?

---

## How to run

```bash
bash courses/course2_continual/chapter1_measurement/class1_two_task_forgetting/run.sh
```

Smoke mode by default (~512 samples per task, 1 epoch each, runs in ~1–2 min on a single GPU). `MODE=full` uses the full splits.

## How to verify

`results/full/<backbone>/course2_continual/chapter1_measurement_class1_two_task_forgetting/two_task/full-ft.json`. Expected band (smoke):

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `avg_accuracy` | 0.50 | 1.0 | Mean test accuracy across both tasks at the final stage |
| `acc_A_after_A` | 0.70 | 1.0 | AG News accuracy right after Task A — sanity that learning happened |
| `acc_B_after_B` | 0.40 | 1.0 | Emotion accuracy after Task B — same sanity |
| `acc_A_after_B` | 0.0 | 0.85 | AG News accuracy after also training on B — should drop substantially |
| `BWT` | -1.0 | -0.05 | Negative = forgetting; band asserts it really happened |

If BWT is near zero, the experiment didn't reproduce — usually because the smoke epoch was too short to overwrite the head, or the tasks happened to share useful features. Open NOTES.md and explain what you saw.

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
