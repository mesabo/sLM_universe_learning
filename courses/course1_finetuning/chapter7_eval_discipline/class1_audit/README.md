# Course 1 · Chapter 7 · Class 1 — Eval discipline audit

> **Goal:** Run an automated audit of your research results. You will scan all your experiment data to find "violations" of scientific discipline, such as missing random seeds or undeclared passing ranges. This is the final step before you can trust your own conclusions.

---

## 🧭 The 5 W's & 1 H (Foundations)

### WHAT are we doing?
We are performing a **Scientific Audit**.
*   **The Target:** Every experiment result (JSON file) you've generated in this course under `results/full/`.
*   **The Goal:** To flag common mistakes that make AI research unreliable.
*   **The Tool:** A script (`audit.py`) that reads your data and looks for specific red flags.

### WHY are we doing this?
*   **Avoiding "Lucky" Results:** If you only run an experiment once, you might have just gotten a "lucky" random seed. A disciplined scientist runs each experiment multiple times to be sure.
*   **Reproducibility:** If your config file doesn't record the `seed` or the `hash`, nobody (not even you) can perfectly recreate your result.
*   **Intellectual Honesty:** By deciding on an "Expected Band" *before* you run the code, you prevent yourself from making excuses for poor performance after the fact.

### WHEN should you use this?
*   Use it after a long weekend of training models.
*   Use it before you publish a blog post or share a report claiming "Method A is 5% better than Method B."
*   Use it as a final "Sanity Check" before moving on to Course 2.

### WHERE do the violations hide?
Violations hide in your **JSON metadata**.
1.  **Config Hash:** Did the code that ran actually match the settings you wrote down?
2.  **Passing Range:** Did you define what "Success" looks like before you started?
3.  **Seed Diversity:** Did you run the same experiment with at least 3 different random seeds?

### HOW does it work (The Pipeline)?
1.  **Scraping:** The audit script "walks" through your results folder and opens every JSON file it finds.
2.  **Validation:** It checks each file against a list of 10+ rules (e.g., "Does this file have a `config_hash`?").
3.  **Grouping:** It groups results by model and task to see if you have enough data points to claim a trend.
4.  **Reporting:** It generates a Markdown table that lists every violation, showing you exactly which file needs fixing.

---

## 🧠 Psycho — the mental model

Most "X beats Y" results in ML are noise. Two reasons:

1. **One seed**: the ranking might flip on a different RNG state. You need at least 3 seeds to claim a winner.
2. **No band**: if you didn't decide what a "good" number looks like *before* the run, you'll talk yourself into accepting whatever you got.

This class doesn't train anything — it audits the discipline you've already imposed (or failed to impose) on yourself. Run it after a sweep, before you write a paper / blog post / commit message / PR description claiming X beats Y.

---

## 🎓 Academic — what we audit

For every JSON under `results/full/`, the audit checks:

- **`config_hash` present** — required for "did this run match its config?" reproducibility.
- **`seed` recorded in `config`** — without it, no rerun can be exact.
- **`expected_band` declared for every metric** in `metrics`. A metric without a band is a hope, not an assertion.
- **All `metrics` values inside `expected_band`** — sanity that the lesson reproduced.
- **Per (course, class, task, method, backbone) cell: more than one seed** when the cell is part of a published comparison. Ratings of significance need ≥ 3.
- **Path collisions** — two JSONs with the same `(backbone, course, class, task, method)` overwriting each other (only the last write survives).

Each violation gets a short tag (`missing_band`, `single_seed`, `out_of_band`, `path_collision`, ...) and a one-line description. The audit emits a Markdown report and a result JSON whose `metrics.n_violations` is the headline count.

**Key Terms for Students:**
*   **Config Hash:** A digital fingerprint of your settings. If you change even one number in your YAML, the hash changes.
*   **Seed:** The starting number for the random number generator. Using the same seed makes your results "Stable."
*   **Metric Band:** A high/low range (e.g., 0.85 to 1.0) that defines what counts as a "Pass."

References:
- [Henderson et al., *Deep Reinforcement Learning that Matters* (AAAI 2018)](https://arxiv.org/abs/1709.06560) — the seminal "your one-seed result is noise" paper
- [Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks* (MLSys 2021)](https://arxiv.org/abs/2103.03098) — practical reproducibility guidance
- [HF — config_hash + manifest patterns](https://huggingface.co/docs/transformers/main/en/internal/audio_utils) (we roll our own in `shared/repro.py`)

---

## 🛠️ Engineering — what the code does

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

---

## 🧪 Research — open questions / extensions

- The audit currently treats every seed value as significant. In practice, seed=42 and seed=43 might produce indistinguishable results on small encoders. Add a "seed equivalence class" via clustering on `metrics`.
- Output a markdown leaderboard sorted by (n_violations, then headline metric). That's a different lens — "which result is both well-disciplined AND best?".
- Wire the audit into a pre-commit hook so any commit that adds a result JSON without a band gets blocked.

---

## 🚀 How to run

```bash
bash courses/course1_finetuning/chapter7_eval_discipline/class1_audit/run.sh
```

Smoke mode by default — audits everything under `results/full/` and emits a Markdown report to stdout + `results/full/_audit_/.../report.md`.

## ✔ How to verify

`results/full/_audit_/course1_finetuning/chapter7_eval_discipline_class1_audit/discipline/audit.json`. Expected band:

| Metric | Passing Range | Meaning |
|---|---|---|
| `n_files_audited` | 1+ | Did the script find any data? |
| `n_unique_cells` | 1+ | Did it find unique experiments? |
| `n_violations` | Any | Information count of discipline errors. |

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
