#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODE="${MODE:-smoke}"
CONFIG="courses/course2_continual/chapter1_measurement/class1_two_task_forgetting/configs/default.yaml"

python courses/course2_continual/chapter1_measurement/class1_two_task_forgetting/train.py \
  --config "${CONFIG}" mode="${MODE}"
