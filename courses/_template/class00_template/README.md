# Course X · Chapter Y · Class Z — <one-line lesson title>

> Goal: <one-sentence statement of what the student can do after this class>.

---

## Psycho — the mental model

<2–4 paragraphs of intuition. Analogies, mental shortcuts, "why this matters". No equations, no code. The reader should leave this section with a mental picture, not a recipe.>

## Academic — what's actually happening

<Theory: equations in `$...$`, definitions, named results. Cite official sources via inline links — papers (arXiv / DOI) and the relevant HF / PEFT / TRL docs. Extra blog links only when no official source exists.>

References:
- <Paper / official doc link 1>
- <Paper / official doc link 2>

## Engineering — what the code does

[`train.py`](./train.py) (or whichever script the lesson uses):

1. <Step 1 — load via `shared.*`>
2. <Step 2 — main computation>
3. <Step 3 — write result JSON via `shared.eval_harness.run_eval`>

### Gotchas
- <Concrete pitfalls a junior engineer would hit. Use code-style for exact symbols.>

## Research — open questions / extensions

- <Question 1 — concrete, runnable as an exercise>
- <Question 2>
- <Question 3>

---

## How to run

```bash
bash courses/<courseX>/<chapterY>/<classZ>/run.sh
```

## How to verify

`results/full/<backbone>/<course>/<class>/<task>/<method>.json` — expected band:

| Metric | Lo | Hi | Meaning |
|---|---|---|---|
| `metric_a` | <lo> | <hi> | <plain-English meaning> |
| `metric_b` | <lo> | <hi> | <plain-English meaning> |

`eval.py` exits non-zero if any metric falls outside its band — that is the per-class verification contract.

---

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
