#!/usr/bin/env bash
# Course 4 / ch1 / class 1 — LCEL chains (invoke, batch, stream)
#
# Usage:
#   bash courses/course4_langchain_ecosystem/chapter1_core/class1_lcel_chains/run.sh
#   MODE=full bash courses/course4_langchain_ecosystem/chapter1_core/class1_lcel_chains/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"

MODE="${MODE:-smoke}"
CONFIG="courses/course4_langchain_ecosystem/chapter1_core/class1_lcel_chains/configs/default.yaml"

python "courses/course4_langchain_ecosystem/chapter1_core/class1_lcel_chains/train.py" --config "${CONFIG}" mode="${MODE}"
