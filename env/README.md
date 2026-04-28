# Conda environments

Two interchangeable environments — pick the one that matches your machine.

| File | When to use |
|---|---|
| `slm-cpu.yml` | No GPU, or you only want to run smoke tests / encoder-model experiments. PyTorch CPU build, no `bitsandbytes`. |
| `slm-gpu.yml` | NVIDIA GPU available (course assumes CUDA 4–7). PyTorch + CUDA 12.1 build, includes `bitsandbytes` for QLoRA / 4-bit. |

## Create

```bash
# CPU
conda env create -f env/slm-cpu.yml
conda activate slm-cpu

# GPU
conda env create -f env/slm-gpu.yml
conda activate slm-gpu
```

## Update after editing the YAML

```bash
conda env update -f env/slm-gpu.yml --prune
```

## Notes

- The course never imports `bitsandbytes` at module top-level — QLoRA classes import it lazily and skip with a clear error in the CPU env.
- HF cache is set per-project via `HF_HOME` (see `.env.example`); the env files do NOT touch your global cache.
- All ML libraries (transformers, peft, trl, accelerate, sentence-transformers, …) come from `pip` even inside the conda env, because the conda channels lag behind PyPI for these packages.
