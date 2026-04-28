# Course 2 · Chapter 5 · Class 1 — Synthesis: which defense actually wins?

> Goal: read the four prior chapters' result JSONs as one comparison matrix and decide which catastrophic-forgetting defense to reach for in which situation. The lesson is **the synthesis**, not a new training method — Course 2 has nothing left to teach about technique. Then optionally launch the same four defenses across all five backbones via the launcher, and watch the matrix grow.

---

## Psycho — the mental model

By now you've trained four models on the same AG News → Emotion sequence:

- **ch1 baseline** — full fine-tuning, no defense. Forgetting is brutal.
- **ch2 replay** — mix Task A samples into Task B training. Cheap, strongest BWT.
- **ch3 EWC** — penalize drift in Task-A-important parameters. No data needed; weakest BWT improvement.
- **ch4 isolation** — LoRA per task, swap at eval. BWT = 0 by construction; loses positive transfer.

There is no single "best" — there's a **decision tree**:

1. Can you keep Task A's data? → **replay** (ch2). It usually wins on raw BWT.
2. Privacy / license / deletion mandate forbids storing data? → **EWC** (ch3) or **isolation** (ch4).
3. Adding many tasks over time, must hot-swap at inference? → **isolation** (ch4).
4. Want some shared representation learning across tasks? → **EWC** (ch3) — it constrains drift but doesn't forbid it.

This class makes the trade-offs concrete by reading the JSONs you already produced in ch1–ch4 and printing them as one table. No new model training.

## Academic — what we're aggregating

For each (backbone, defense) pair, the result JSON in `results/full/<backbone>/course2_continual/<class>/two_task/<method>.json` contains:

- `metrics.acc_A_after_A` — task A accuracy right after training A
- `metrics.acc_A_after_B` — task A accuracy after also training B (the forgetting signal)
- `metrics.acc_B_after_B` — task B accuracy after training B
- `metrics.BWT` — Lopez-Paz & Ranzato backward transfer: `R[T][i] - R[i+1][i]` averaged over prior tasks
- `metrics.avg_accuracy` — mean across tasks at the final stage
- `extras.history_matrix` — full 3×2 accuracy matrix per stage

[`aggregate.py`](./aggregate.py) walks `results/full/**/course2_continual/**/two_task/*.json`, groups by (backbone, method), and emits a Markdown table sorted to highlight the BWT vs avg-accuracy Pareto frontier.

## Engineering — what the code does

This class has **no training step**. Instead:

1. [`aggregate.py`](./aggregate.py) is the work — `python aggregate.py --config configs/default.yaml [--output PATH]`. It:
   - Walks `results/full/<backbone>/course2_continual/**/two_task/*.json` (configurable glob).
   - For each result, extracts `(backbone, course, class, method, metrics.{BWT, acc_A_after_A, acc_A_after_B, acc_B_after_B, avg_accuracy})`.
   - Sorts by (backbone, method) and emits a Markdown table to stdout (and optionally a file).
   - Writes its own result JSON (`extras.aggregated[]` with all the rows; metric `n_methods_aggregated` for the verification band).
2. The four grid YAMLs live in [`courses/course2_continual/grids/`](../../grids/) — one per defense (`baseline_cross_backbone.yaml`, `replay_cross_backbone.yaml`, `ewc_cross_backbone.yaml`, `isolation_cross_backbone.yaml`). Each fans out across the 3 encoder backbones (decoder backbones don't classify well in this fast-train setup) using `shared/sharded_run.sh`.

### Gotchas
- **Run the underlying chapters first.** This class only aggregates — if you haven't run ch1–4, the table will be empty (the band will fail).
- The decoder backbones (SmolLM2-135M / 360M) are excluded by default from the grids: causal LMs technically support `AutoModelForSequenceClassification` but with random head + few steps they don't learn classification well. Override `backbones=...` in the grid to include them if you want the data.
- `aggregate.py` doesn't dedupe — re-running a (backbone, method) cell overwrites the JSON, and aggregate just reads what's there. If you want multiple seeds, use the `seed=` override and the result JSONs collide; bring back per-seed paths if needed.

## Research — open questions

- The defaults in ch1-4 use the same seed and data caps. Sweep `seed ∈ {42, 1337, 7}` for each (backbone, method) and add error bars to the table. Do the rankings hold under noise?
- Add a "**joint training**" oracle row (train on `ag_news ∪ emotion` from scratch; no continual). It's the upper bound on average accuracy. How close do replay / EWC / isolation get?
- Add a third task (e.g. `SetFit/sst2`). The two-task summary metrics generalize but the table grows. Which defenses scale?
- Build a **cost vs benefit** scatter: x-axis = "extra train time per task" (replay > EWC > isolation in your measurements?); y-axis = `acc_A_after_B`. Where's the Pareto frontier?

---

## How to run

```bash
# (1) After running ch1, ch2, ch3, ch4 (each emits a JSON):
bash courses/course2_continual/chapter5_recipes/class1_synthesis/run.sh

# (2) Optional — fan all four defenses across the 3 encoder backbones via the sharder:
bash shared/sharded_run.sh courses/course2_continual/grids/baseline_cross_backbone.yaml
bash shared/sharded_run.sh courses/course2_continual/grids/replay_cross_backbone.yaml
bash shared/sharded_run.sh courses/course2_continual/grids/ewc_cross_backbone.yaml
bash shared/sharded_run.sh courses/course2_continual/grids/isolation_cross_backbone.yaml

# Then re-run the aggregator to see the full matrix.
bash courses/course2_continual/chapter5_recipes/class1_synthesis/run.sh
```

## How to verify

`results/full/_aggregator_/course2_continual/chapter5_recipes_class1_synthesis/two_task/synthesis.json` (a synthetic "backbone" name `_aggregator_` is used because this class doesn't bind to one particular backbone). Expected band:

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `n_methods_aggregated` | 1 | 100 | At least one ch1-4 result was found |
| `unique_methods` | 1 | 100 | Sanity: distinct method tags counted |
| `unique_backbones` | 1 | 100 | Sanity: distinct backbones counted |

If the band fails because no JSONs were found, run `bash courses/course2_continual/chapter1_measurement/.../run.sh` (and ch2, ch3, ch4) first. That's the prerequisite.

## Instructor checklist

Before marking this class "done":

- [ ] All four mode sections (Psycho / Academic / Engineering / Research) are present and ≥ 2 paragraphs each.
- [ ] Every reference link points at an official source (paper / HF doc / repo) where one exists.
- [ ] `aggregate.py` contains no numeric literal other than `0` / `1` (everything in `configs/*.yaml`).
- [ ] `configs/default.yaml` declares an `expected_band` for every metric written by `aggregate.py`.
- [ ] `run.sh` uses `HF_HOME=$PWD/.cache/huggingface` and is `chmod +x`.
- [ ] `exercises.md` has exactly three exercises (warm-up / apply / stretch).
- [ ] At least one smoke run produced a non-empty table.
- [ ] Linked from the parent course `README.md` table.
