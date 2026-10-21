#!/usr/bin/env bash
# Creates only this repository's rollout venv; does not install system drivers.
set -euo pipefail
DP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DP_PYTHON="${DP_PYTHON:-3.11}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${DP_ROOT}/.cache/uv}"
if [[ ! -x "${DP_ROOT}/.venv/bin/python" ]]; then
  uv venv --python "$DP_PYTHON" "${DP_ROOT}/.venv"
fi
uv pip install --python "${DP_ROOT}/.venv/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128'
uv pip install --python "${DP_ROOT}/.venv/bin/python" \
  -r "${DP_ROOT}/requirements-rollout.txt"
uv pip check --python "${DP_ROOT}/.venv/bin/python"
printf 'Activate with: source %s/scripts/activate_rollout_env.sh\n' "$DP_ROOT"
