#!/usr/bin/env bash
# Course 0 / ch2 / class 1 — encoder vs decoder side by side.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG="courses/course0_bridge/chapter2_encoder_vs_decoder/class1_side_by_side/configs/default.yaml"
python courses/course0_bridge/chapter2_encoder_vs_decoder/class1_side_by_side/train.py --config "${CONFIG}"
