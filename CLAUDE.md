# sLM Universe Learning — Claude Guidelines

This file is the standing brief for any Claude session in this directory. Read it before doing anything.

---

## What this project is

A self-taught, multi-chapter course on **small language models (sLM)**. Eight pillars:

1. Foundations bridge (only what's necessary — no transformer-from-scratch derivation)
2. Fine-tuning sLM (full FT, LoRA, QLoRA, DPO, embedding FT, **RAG**)
3. Preventing catastrophic forgetting
4. Pipelines (deployment, active learning, auto-update, monitoring)
5. Architecture (RMSNorm, RoPE, GQA, KV cache, quantization formats — read modern code, don't re-derive)
6. Auto-research / auto-learn (Karpathy's `autoresearch` as primary text)
7. Multi-agent collaboration & learning (framework-agnostic core + one Anthropic Agent SDK comparison)

Folder hierarchy: `courses/courseX/chapterY/classZ_<slug>/`.

---

## Hard rules — non-negotiable

### Git artifacts (NO AI TRACE — EVER)

- **Never** add `Co-Authored-By: Claude …` (or any AI co-author) to commit messages.
- **Never** add `🤖 Generated with Claude Code` (or similar) footers anywhere.
- **Never** add `.claude/`, `claude*`, `anthropic*`, or other AI-tool entries to `.gitignore` (use a global gitignore for tool dirs).
- PR / commit messages are written in plain human voice describing the change.
- Applies retroactively: clean traces from any commit before pushing.

### Shared parent directory

`/home/Aboya_25R9803/projects/extra/learning/` is **shared with other AI tools** (Codex owns the sibling `learning-llms/`).
- **Never** write files at the parent level. Stay inside `learning_slm_claude/`.
- Don't touch the parent's `CLAUDE.md`, `.codex`, `.claude/`, `.remember/`, or `learning-llms/`.

### Production research project is off-limits

The user's production research at `/home/Aboya_25R9803/projects/perso/LLMium/projects/02-SLM-Foundational/` may be **referenced and linked** but **never imported, modified, or copied from**. This course re-implements its own infra.

### Smoke test first

Before launching any new training script at full epochs/seeds, run a 1–2 epoch / 1-seed / smallest-case smoke run synchronously and confirm exit 0. Bugs in calling conventions surface in the first 30 seconds.

---

## Pedagogical contract — every class

Each `class*/README.md` MUST present the lesson through four "modes":

| Mode | What it covers |
|---|---|
| **Psycho** | Intuition, mental models, analogies, "why this matters" |
| **Academic** | Theory, equations (in `$...$`), citations to official papers (DOI/arXiv) |
| **Engineering** | Code walk-through, configs, gotchas, reproducibility notes |
| **Research** | Open questions, ablation menu, recent papers / extensions |

Then end with:
- **How to run** (single bash command)
- **How to verify** (expected metric band; eval exits non-zero outside band)

Prefer **official sources**: HuggingFace docs, original paper PDFs, official repos. Add extra links only when an official one doesn't exist.

---

## Code rules

- `train.py` and `eval.py` may **not** contain numeric literals other than `0`/`1`. Everything else lives in YAML.
- Models, tokenizers, datasets are loaded via `shared/`. Classes never call `AutoModel.from_pretrained` directly except in Course 0 ch1 (the lesson is *how* to do it).
- HF cache is project-local at `.cache/huggingface/` via `HF_HOME` (set in `.env`). Never leak to `~/.cache/huggingface`.
- Distinguish **encoder/sentence-encoder** sLMs (MiniLM, BGE-small, GTE-small) from **decoder** sLMs (SmolLM2-135M / 360M). Same topic, different recipes.

---

## Hardware

- 4 GPUs available: CUDA `4,5,6,7` (see `configs/hardware.yaml`).
- The launcher shards `(method × backbone × seed)` grids round-robin via `xargs -P` over those 4 devices, one job per GPU.
- CPU env (`env/slm-cpu.yml`) supports the smoke tests and encoder-model classes; GPU env (`env/slm-gpu.yml`) is needed for QLoRA / SmolLM2 training.

---

## Backbones (single source of truth: `configs/backbones.yaml`)

| Name | Kind | Use |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` *(default)* | sentence-encoder | classification, retrieval, smoke tests |
| `BAAI/bge-small-en-v1.5` | sentence-encoder | retrieval, MTEB-style eval |
| `thenlper/gte-small` | sentence-encoder | retrieval comparison |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | decoder | SFT, DPO, agents |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | decoder | larger-scale SFT, autoresearch, agents |

All experiments evaluated across all five (where applicable). Results land at `results/full/<backbone>/<task>/<method>.json`.

---

## When in doubt

- Start a class with the `class00_template/` skeleton (when it exists) and never deviate from the four-mode + how-to-run + how-to-verify structure.
- Smoke test before launching anything heavy.

## Supervision TODOs

- [ ] Confirm the `four-mode` README template exists in `class00_template/` and is referenced by maintainers.
- [ ] Add per-class instructor checklist examples (short bullets) under the pedagogical contract.
- [ ] Ensure the `no AI trace` rules are copied to `learning_llms_codex` README for consistency.

- Ask the user before destructive actions (rm, force-push, env destroy).
