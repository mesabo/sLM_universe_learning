#!/usr/bin/env bash
# Class entrypoint. Replace <courseX>/<chapterY>/<classZ> when copying.
#
# Usage:
#   bash courses/<courseX>/<chapterY>/<classZ>/run.sh
#   MODE=full bash courses/<courseX>/<chapterY>/<classZ>/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

MODE="${MODE:-smoke}"
CONFIG="courses/<courseX>/<chapterY>/<classZ>/configs/default.yaml"

python "courses/<courseX>/<chapterY>/<classZ>/train.py" --config "${CONFIG}" mode="${MODE}"
