#!/usr/bin/env bash
# Course 0 / ch1 / class 1 — HF ecosystem sanity check.
# Runs the default backbone, then loops over all five.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG="courses/course0_bridge/chapter1_hf_tour/class1_automodel/configs/default.yaml"

# Default backbone (configured in YAML).
python courses/course0_bridge/chapter1_hf_tour/class1_automodel/train.py --config "${CONFIG}"

# Optional: sweep all five backbones. Comment out to skip.
for bb in \
  "sentence-transformers/all-MiniLM-L6-v2" \
  "BAAI/bge-small-en-v1.5" \
  "thenlper/gte-small" \
  "HuggingFaceTB/SmolLM2-135M-Instruct" \
  "HuggingFaceTB/SmolLM2-360M-Instruct"
do
  echo "[run] backbone=${bb}"
  python courses/course0_bridge/chapter1_hf_tour/class1_automodel/train.py \
    --config "${CONFIG}" backbone="${bb}"
done
