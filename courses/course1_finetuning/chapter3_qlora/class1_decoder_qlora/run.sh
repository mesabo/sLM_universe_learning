#!/usr/bin/env bash
# Course 1 / ch3 / class 1 — QLoRA on SmolLM2-360M.
# Requires the GPU env (bitsandbytes). Will exit non-zero on CPU-only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODE="${MODE:-smoke}"
CONFIG="courses/course1_finetuning/chapter3_qlora/class1_decoder_qlora/configs/default.yaml"

python courses/course1_finetuning/chapter3_qlora/class1_decoder_qlora/train.py \
  --config "${CONFIG}" mode="${MODE}"
