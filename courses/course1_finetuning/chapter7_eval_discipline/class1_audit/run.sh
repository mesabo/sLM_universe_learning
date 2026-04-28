#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

MODE="${MODE:-smoke}"
CONFIG="courses/course1_finetuning/chapter7_eval_discipline/class1_audit/configs/default.yaml"

python courses/course1_finetuning/chapter7_eval_discipline/class1_audit/audit.py \
  --config "${CONFIG}" mode="${MODE}"
