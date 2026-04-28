# Exercises — Course 1 · ch 7 · class 1 (eval discipline audit)

## 1. Warm-up — read your own report

After running `bash run.sh`, open `results/full/_audit_/.../report.md`. Pick the most-frequent rule and decide: is it a real problem (you'd lose a paper review over it) or a false alarm given the lesson's pedagogical scope? Document your decision in `NOTES.md`.

## 2. Apply — fix one cell

Find a `single_seed` cell in the report (likely most of them). Re-run that lesson with `seed=42` and `seed=1337` and `seed=7`. Re-run the audit. Did the cell drop off the violation list?

## 3. Stretch — pre-commit hook

Wire `audit.py` into a pre-commit hook (`.git/hooks/pre-commit`) that runs the audit and refuses the commit if `n_violations` is over a threshold YOU declare. What threshold makes sense for this repo, and why?
