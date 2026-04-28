# `_template/` — start every new class from here

This directory is the **canonical skeleton** every class folder is expected to follow. It is referenced by `CLAUDE.md` and by the four-mode pedagogical contract.

## How to use it

```bash
# From the repo root, when starting a new class:
cp -r courses/_template/class00_template courses/courseX/chapterY/classZ_<slug>
$EDITOR courses/courseX/chapterY/classZ_<slug>/README.md
```

Then fill in:
1. The four-mode README (Psycho / Academic / Engineering / Research) — never delete a section, even if it's brief.
2. `configs/default.yaml` — every numeric / string knob lives here. No literals other than `0`/`1` in `train.py`.
3. `train.py` and `eval.py` — wired to `shared.*` for backbone loading, dataset specs, eval harness.
4. `run.sh` — one-line entrypoint. CUDA env vars and `HF_HOME` are set in the script.
5. `exercises.md` — three exercises (warm-up / apply / stretch).

## What NOT to put in a class folder

- Large datasets (use `shared.datasets` + the canonical YAML spec).
- Vendored model weights (HF cache lives at `<repo>/.cache/huggingface/`).
- Notebooks (the project is `.py` + Markdown only — sharder doesn't run `.ipynb`).
- Anything in `__pycache__` / `runs/` / `checkpoints/` (gitignored).
