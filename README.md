# Poincaré Section Reservoirs (PSR)

Geometry-inspired reservoir computing built from Poincaré sections of chaotic flows.
This repo provides a clean, modular implementation with end-to-end experiments and plotting.

## Install

```bash
# from the repo root
pip install -e .[dev]
pre-commit install


## Quickstart

import numpy as np
from psr.data.generators import lorenz63
from psr.pipelines.experiment import run_experiment
from psr.config import load_config

cfg = load_config("configs/default.yaml")  # edit dataset/model/train sections
result = run_experiment(cfg)
print(result.metrics)


## CLI

# Train + evaluate on Lorenz (closed-loop)
psr train --config configs/default.yaml dataset=dataset/lorenz63.yaml train=train/closed_loop.yaml

# Make paper-quality figures from a run
psr plot --run artifacts/runs/lorenz63/2025-08-12_001


## Repo layout (high level)

src/psr/           # package (geometry, graphs, reservoir, readout, metrics, viz, pipelines)
configs/           # YAML configs for dataset/model/train/eval
notebooks/         # small, focused demos/figures
tests/             # unit tests
artifacts/         # runs & figures (gitignored)
