#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CONFIG="${1:-${SCRIPT_DIR}/configs/smoke.yaml}"
conda run -n slm-gpu python "${SCRIPT_DIR}/train.py" --config "${CONFIG}"
