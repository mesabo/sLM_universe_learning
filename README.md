# sLM Universe Learning

A self-taught, multi-chapter course on **small language models** (sLM). Combines theory and practice across seven topical pillars and is delivered through four pedagogical "modes": **Psycho** (intuition), **Academic** (theory + citations), **Engineering** (clean documented code), **Research** (open questions + ablations).

> Repo: `https://github.com/mesabo/sLM_universe_learning.git`
> Lives in: `/home/Aboya_25R9803/projects/extra/learning/learning_slm_claude/`
> Sibling `../learning-llms/` and `../learning_llms_codex/` are owned by other AI tools — do not touch them.

---

## Quick start

```bash
# 1. Create the conda env (pick one)
conda env create -f env/slm-gpu.yml      # NVIDIA GPU (CUDA 12.1)
conda env create -f env/slm-cpu.yml      # no GPU
conda activate slm-gpu                    # or slm-cpu

# 2. Configure project-local HF cache
cp .env.example .env
set -a; source .env; set +a

# 3. Install the in-repo package (editable)
pip install -e .

# 4. Smoke test
pytest -q
```

Then open the first class:

```bash
$EDITOR courses/course0_bridge/chapter1_hf_tour/class1_automodel/README.md
bash    courses/course0_bridge/chapter1_hf_tour/class1_automodel/run.sh
```

---

## Curriculum

| # | Course | Pillar |
|---|---|---|
| 0 | `course0_bridge/` | Just-enough HF + transformer refresher |
| 1 | `course1_finetuning/` | Full FT, LoRA, QLoRA, DPO, embedding FT, **RAG**, evaluation |
| 2 | `course2_continual/` | Catastrophic forgetting: measure, replay, regularize, isolate |
| 3 | `course3_pipeline/` | Serving, active learning, auto-update, monitoring |
| 4 | `course4_architecture/` | RMSNorm, RoPE, GQA, KV cache, quantization formats |
| 5 | `course5_autoresearch/` | Karpathy `autoresearch` read-along + reimplementation |
| 6 | `course6_multiagent/` | Multi-agent collaboration & cooperative learning |

Each `class*/` contains: `README.md` (lesson), `configs/*.yaml`, `train.py`, `eval.py`, `run.sh`, `exercises.md`, `results/`.

---

## Layout

```
learning_slm_claude/
  README.md                  # this file
  CLAUDE.md                  # standing guidelines for any Claude session
  pyproject.toml             # ruff + pytest config; installs `shared/` as a package
  .env.example               # HF_HOME pointing at .cache/huggingface
  .gitignore                 # blocks results/, .cache/, .env, etc.
  .python-version            # 3.11
  env/
    slm-cpu.yml              # CPU conda env
    slm-gpu.yml              # GPU conda env
    README.md                # how to create / update
  configs/
    backbones.yaml           # the 5 sLM backbones, single source of truth
    hardware.yaml            # cuda_devices: [4,5,6,7]
    defaults.yaml            # seeds, dtype, output roots
  shared/                    # importable utilities (in-repo package)
  datasets/                  # tiny canonical samples only
  courses/                   # the lessons
  results/                   # gitignored experiment artifacts
  tests/                     # smoke tests for shared/
  .cache/huggingface/        # project-local HF cache (HF_HOME)
```

---

## Backbones

All experiments evaluate across these five (where applicable):

| Name | Kind | Params |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` *(default)* | sentence-encoder | 22 M |
| `BAAI/bge-small-en-v1.5` | sentence-encoder | 33 M |
| `thenlper/gte-small` | sentence-encoder | 33 M |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | decoder | 135 M |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | decoder | 360 M |

Edit `configs/backbones.yaml` to add/swap.

---

## Hardware

The launcher shards `(method × backbone × seed)` grids round-robin via `xargs -P` over **CUDA 4, 5, 6, 7**, one job per GPU. Override via env var `CUDA_DEVICES=4,5` for partial use.

---

## How verification works

Every class declares an **expected metric band** in its `README.md`. `eval.py` exits non-zero if a result falls outside the band — that's the contract. Plus:

- `pytest tests/` smoke-tests `shared/` modules.
- `manifest.json` (config hash + env + seed) is written next to each result for reproducibility.

No CI; the metrics-band check is the contract.

---

## Reference (do NOT import or modify)

For production-grade few-shot transfer matrices, causal LLM ensembles, etc., see the user's research project at `/home/Aboya_25R9803/projects/perso/LLMium/projects/02-SLM-Foundational/`. This course re-implements its own infra; that project is for inspiration only.
