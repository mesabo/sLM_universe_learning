#!/usr/bin/env bash
# Round-robin sharded launcher.
#
# Usage:
#   bash shared/sharded_run.sh path/to/grid.yaml
#
# Reads `cuda_devices` and `jobs_per_gpu` from configs/hardware.yaml
# (env CUDA_DEVICES=4,5 overrides). Expands the grid spec via the Python
# launcher, then assigns each job to a GPU in round-robin and runs them
# in parallel via xargs -P.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <grid.yaml>" >&2
  exit 2
fi

GRID="$1"

# Resolve project root by walking up from this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Project-local HF cache.
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Read hardware config (env override wins).
if [[ -n "${CUDA_DEVICES:-}" ]]; then
  IFS=',' read -r -a DEVICES <<<"${CUDA_DEVICES}"
else
  mapfile -t DEVICES < <(
    python - <<'PY'
from shared.config import load_hardware_config
for d in load_hardware_config()["cuda_devices"]:
    print(d)
PY
  )
fi

JOBS_PER_GPU="${JOBS_PER_GPU:-$(python - <<'PY'
from shared.config import load_hardware_config
print(load_hardware_config().get("jobs_per_gpu", 1))
PY
)}"

PARALLEL=$(( ${#DEVICES[@]} * JOBS_PER_GPU ))
echo "[launcher] devices=${DEVICES[*]} jobs_per_gpu=${JOBS_PER_GPU} parallel=${PARALLEL}" >&2

# Expand grid to job lines.
mapfile -t JOBS < <(python -m shared.launcher "${GRID}")
TOTAL=${#JOBS[@]}
if (( TOTAL == 0 )); then
  echo "[launcher] no jobs in ${GRID}" >&2
  exit 0
fi
echo "[launcher] ${TOTAL} jobs from ${GRID}" >&2

# Round-robin device assignment via line index.
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
for i in "${!JOBS[@]}"; do
  device="${DEVICES[$(( i % ${#DEVICES[@]} ))]}"
  printf 'CUDA_VISIBLE_DEVICES=%s %s\n' "${device}" "${JOBS[$i]}" >>"$TMP"
done

# Fan out.
xargs -a "$TMP" -P "${PARALLEL}" -I{} bash -c '{}'
