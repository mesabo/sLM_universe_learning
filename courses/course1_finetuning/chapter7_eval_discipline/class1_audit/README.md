# Course 1 · Chapter 7 · Class 1 — Eval discipline audit

> Goal: scan every result JSON in `results/full/` and flag the most common eval-discipline violations — missing seeds, single-seed cells (no error bars possible), undeclared metric bands, missing `config_hash`, and (method × backbone × task) cells with only one run when the lesson said to sweep. Run this before publishing any cross-method comparison so reviewers don't catch the holes for you.

---

## Psycho — the mental model

Most "X beats Y" results in ML are noise. Two reasons:

1. **One seed**: the ranking might flip on a different RNG state. You need at least 3 seeds to claim a winner.
2. **No band**: if you didn't decide what a "good" number looks like *before* the run, you'll talk yourself into accepting whatever you got.

This class doesn't train anything — it audits the discipline you've already imposed (or failed to impose) on yourself. Run it after a sweep, before you write a paper / blog post / commit message / PR description claiming X beats Y.

## Academic — what we audit

For every JSON under `results/full/`, the audit checks:

- **`config_hash` present** — required for "did this run match its config?" reproducibility.
- **`seed` recorded in `config`** — without it, no rerun can be exact.
- **`expected_band` declared for every metric** in `metrics`. A metric without a band is a hope, not an assertion.
- **All `metrics` values inside `expected_band`** — sanity that the lesson reproduced.
- **Per (course, class, task, method, backbone) cell: more than one seed** when the cell is part of a published comparison. Ratings of significance need ≥ 3.
- **Path collisions** — two JSONs with the same `(backbone, course, class, task, method)` overwriting each other (only the last write survives).

Each violation gets a short tag (`missing_band`, `single_seed`, `out_of_band`, `path_collision`, ...) and a one-line description. The audit emits a Markdown report and a result JSON whose `metrics.n_violations` is the headline count.

References:
- [Henderson et al., *Deep Reinforcement Learning that Matters* (AAAI 2018)](https://arxiv.org/abs/1709.06560) — the seminal "your one-seed result is noise" paper
- [Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks* (MLSys 2021)](https://arxiv.org/abs/2103.03098) — practical reproducibility guidance
- [HF — config_hash + manifest patterns](https://huggingface.co/docs/transformers/main/en/internal/audio_utils) (we roll our own in `shared/repro.py`)

## Engineering — what the code does

[`audit.py`](./audit.py):

1. Walks `results/full/**/*.json` (configurable glob).
2. For each JSON, runs the checks listed above. Flags get appended with a 60-char description, the file path, and the rule key.
3. Groups by `(backbone, course, class, task, method)` to detect single-seed cells and path collisions.
4. Renders a Markdown report (one row per violation), prints to stdout, optionally writes to disk.
5. Writes the standard result JSON via `shared.eval_harness.run_eval` with `metrics.n_violations`, `metrics.n_files_audited`, `metrics.n_unique_cells` and bands like `n_violations <= MAX_FOR_SMOKE`.

Note: the audit deliberately does NOT touch the audited JSONs — it's read-only. It also doesn't auto-fix anything; the lesson is "see the violations, then fix the upstream lessons."

### Gotchas
- The audit's own result JSON is excluded from its own audit (otherwise the audit always shows a single-seed cell for itself).
- A single-seed warning isn't always a violation — for sanity classes (Course 0) the seed sweep doesn't add information. The config has `single_seed_excludes` for this.
- Path collision detection compares the JSON path strings, not their `config_hash`. Two runs with the same identifiers but different configs would still collide; that's a bug in the upstream class's auto-method-tag derivation.

## Research — open questions / extensions

- The audit currently treats every seed value as significant. In practice, seed=42 and seed=43 might produce indistinguishable results on small encoders. Add a "seed equivalence class" via clustering on `metrics`.
- Output a markdown leaderboard sorted by (n_violations, then headline metric). That's a different lens — "which result is both well-disciplined AND best?".
- Wire the audit into a pre-commit hook so any commit that adds a result JSON without a band gets blocked.

---

## How to run

```bash
bash courses/course1_finetuning/chapter7_eval_discipline/class1_audit/run.sh
```

Smoke mode by default — audits everything under `results/full/` and emits a Markdown report to stdout + `results/full/_audit_/.../report.md`.

## How to verify

`results/full/_audit_/course1_finetuning/chapter7_eval_discipline_class1_audit/discipline/audit.json`. Expected band:

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `n_files_audited` | 1 | 100000 | Sanity: at least one JSON found |
| `n_unique_cells` | 1 | 100000 | Sanity: distinct (backbone, course, class, task, method) cells counted |
| `n_violations` | 0 | 100000 | Permissive (the *count* is informational; the lesson is in the report) |

The headline output is the report itself — read the Markdown table, fix the violations upstream, re-run.

## Instructor checklist

Before marking this class "done":

- [ ] All four mode sections present.
- [ ] `audit.py` contains no numeric literal other than `0` / `1` (everything in `configs/*.yaml`).
- [ ] `configs/default.yaml` declares an `expected_band` for every metric written.
- [ ] `run.sh` uses `HF_HOME=$PWD/.cache/huggingface` and is `chmod +x`.
- [ ] `exercises.md` has exactly three exercises.
- [ ] At least one smoke run produced a non-empty audit report.
- [ ] Linked from the parent course `README.md` table.
