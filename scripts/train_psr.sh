#!/usr/bin/env bash
set -euo pipefail

# Train + evaluate a PSR experiment.
# Usage examples:
#   scripts/train_psr.sh --config configs/default.yaml
#   scripts/train_psr.sh --config configs/default.yaml --model configs/model/psr.yaml --set model.bins=64 train.seed=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Ensure local package is discoverable if not installed in editable mode
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

if ! command -v psr >/dev/null 2>&1; then
  echo "[info] 'psr' CLI not found in PATH. Using module call."
  python -m psr.pipelines.cli train "$@"
else
  psr train "$@"
fi
