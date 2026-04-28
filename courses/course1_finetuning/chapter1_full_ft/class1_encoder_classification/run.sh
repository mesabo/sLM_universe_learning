#!/usr/bin/env bash
# Course 1 / ch1 / class 1 — full FT encoder classification.
# MODE=smoke (default) or MODE=full

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODE="${MODE:-smoke}"
CONFIG="courses/course1_finetuning/chapter1_full_ft/class1_encoder_classification/configs/default.yaml"

python courses/course1_finetuning/chapter1_full_ft/class1_encoder_classification/train.py \
  --config "${CONFIG}" mode="${MODE}"
