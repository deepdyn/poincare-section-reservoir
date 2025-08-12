#!/usr/bin/env bash
set -euo pipefail

# Show where figures from the latest run are saved (or a specific run).
# Usage:
#   scripts/make_figure_overlays.sh                       # auto-detect latest run
#   scripts/make_figure_overlays.sh artifacts/runs/lorenz63/2025-08-12_11-10-03

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-}"

if [[ -z "${RUN_DIR}" ]]; then
  DS="${2:-lorenz63}"
  BASE="${ROOT_DIR}/artifacts/runs/${DS}"
  if [[ ! -d "${BASE}" ]]; then
    echo "No runs found in ${BASE}"
    exit 1
  fi
  # Pick most recent directory
  RUN_DIR="$(ls -td "${BASE}"/* | head -n1)"
fi

if ! command -v psr >/dev/null 2>&1; then
  python -m psr.pipelines.cli plot --run "${RUN_DIR}"
else
  psr plot --run "${RUN_DIR}"
fi
