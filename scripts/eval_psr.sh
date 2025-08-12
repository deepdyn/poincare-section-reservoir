#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run an evaluation-only style experiment (same pipeline; prints NRMSE by horizons).
# You can pass the same args as train, e.g.:
#   scripts/eval_psr.sh --config configs/default.yaml --set eval.horizons="[200,400,1200]"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

if ! command -v psr >/dev/null 2>&1; then
  python -m psr.pipelines.cli train "$@"
else
  psr train "$@"
fi
