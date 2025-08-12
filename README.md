Here’s a polished, professional `README.md` you can drop in. It aligns with your repo structure, clarifies the method and theory, and fixes the CLI examples to match the scaffolded `psr` tool.

---

### `README.md`

````markdown
# Topologically Informed Echo State Networks via Poincaré Return Maps

**Poincaré-Section Reservoirs (PSR)** is a geometry-driven reservoir computing library that builds the recurrent graph **directly from data** via Poincaré return maps. It provides a clean, modular implementation with end-to-end experiments, plotting, and tests.

> **Research status.** This software accompanies the paper  
> **“Topologically Informed Echo State Networks via Poincaré Return Maps for Chaotic Time-Series”** (under review at *Neural Networks*, Elsevier).  
> *Short title:* Topologically Informed Reservoirs.  
> *Authors:* Pradeep Singh\* (corresponding), Ashutosh Kumar, Sutirtha Ghosh, Balasubramanian Raman  
> *Affiliation:* Machine Intelligence Lab, Dept. of CSE, IIT Roorkee, India  
> *Contact:* pradeep.cs@sric.iitr.ac.in

---

## Why PSR?

Classical Echo State Networks (ESNs) draw a random sparse recurrent matrix \(W\) and tune only the readout. PSR replaces that heuristic with a **topologically informed** construction:

1. Slice a single trajectory \(x_t\) by a transverse hyperplane (Poincaré section) to obtain a sequence of **return points** on the plane.
2. **Coarse-grain** those points into bins/symbols on the section.
3. Compute **empirical transition frequencies** between successive symbols to form a row-stochastic matrix \(T\).
4. **Spectrally rescale** \(T\) to target radius \(\rho\) to obtain the reservoir adjacency \(W\).

Each neuron corresponds to a concrete region on the attractor; each edge weight reflects an observed return probability—yielding a lean, interpretable reservoir without gradient training or equation knowledge.

**Theory (paper):** As the partition is refined, \(W\) converges in operator norm to a scaled Perron–Frobenius operator of the true Poincaré map, providing a consistency guarantee typically unavailable to vanilla ESNs.

**Practice (paper):** With a once-trained linear/quadratic readout, a 300-node PSR competitively extends valid prediction time on Lorenz, Rössler, Chen–Ueta, and related chaotic benchmarks—without hyper-parameter sweeps. (See the paper for complete protocol and statistics.)

---

## Features

- **Deterministic graph from data** via Poincaré return maps and Ulam discretization.
- **Geometry tools** for sections, ensembles of planes, projections, and return-map diagnostics.
- **Graphs & scaling**: binning, transition adjacencies, small-world baselines, spectral-radius rescaling.
- **Reservoir core**: leaky-integrator ESN with clean state-update API.
- **Readouts**: ridge, lasso, and RLS (online), with open/closed-loop evaluation.
- **Metrics**: NRMSE, VPT, horizon-wise evaluation; **Viz**: overlays, phase portraits, eigenspectra, section plots.
- **Reproducible pipeline** with YAML configs and a simple CLI.

---

## Installation

```bash
# from the repo root
pip install -e .[dev]
pre-commit install
````

**Requirements:** Python ≥ 3.9; dependencies are specified in `pyproject.toml` (NumPy, SciPy, scikit-learn, matplotlib, seaborn, PyYAML, networkx, pandas, etc.).

---

## Quickstart

```python
# notebooks/01_quickstart.ipynb mirrors this example

from psr.config import load_config
from psr.pipelines.experiment import run_experiment

cfg = load_config("configs/default.yaml")   # dataset/model/train/eval knobs
result = run_experiment(cfg)

print("NRMSE by horizon:", result.metrics)
print("Run directory:", result.run_dir)     # figures & artifacts saved here
```

---

## CLI

```bash
# Train + evaluate on Lorenz (closed-loop)
psr train \
  --config configs/default.yaml \
  --dataset configs/dataset/lorenz63.yaml \
  --train configs/train/closed_loop.yaml

# Override knobs on the fly
psr train \
  --config configs/default.yaml \
  --model configs/model/psr.yaml \
  --set model.bins=64 model.sections=3 train.seed=1

# Show where figures from a run were saved
psr plot --run artifacts/runs/lorenz63/<timestamped-run-dir>
```

**Notes.**

* In PSR, the effective reservoir size is `bins_u × bins_v`. With `bins=32`, the ESN ends up with $N=1024$ units.
* The Poincaré plane defaults to a transverse plane through the mean with normal along $z$ (3-D). Adjust `model.section_rotation_deg` and `model.sections` for a small ensemble.

---

## Minimal API Sketch

```python
from psr.geometry import PoincarePlane, SectionEnsemble
from psr.graphs import adjacency_from_section_points, set_spectral_radius
from psr.reservoir import ESN, MSPSR
from psr.readout import RidgeReadout
```

* `MSPSR.build_from_trajectory(X) -> (ESN, info)` wires: trajectory → section hits → bins → adjacency $T$ → rescaled $W$ → ESN.
* `RidgeReadout.fit(states, targets).predict(states)` handles the linear readout.
* `psr.viz` contains convenience plotting for overlays, eigenspectra, and section diagnostics.

---

## Repository Layout

```
src/psr/
  data/        # ODE generators (Lorenz/Rössler/Chen–Ueta), loaders, transforms
  geometry/    # Poincaré plane, ensembles, return maps
  graphs/      # binning, transition adjacencies, scaling, small-world baselines
  reservoir/   # ESN cells, core ESN, MS-PSR wiring
  readout/     # ridge, lasso, RLS + trainer utilities
  metrics/     # NRMSE, VPT, horizon evaluators
  viz/         # overlays, phase portraits, section plots
  pipelines/   # run_experiment + CLI

configs/       # dataset/model/train/eval YAMLs
scripts/       # train/eval wrappers and figure helpers
notebooks/     # 01_quickstart / 02_ablation / 03_figures
tests/         # pytest unit tests
artifacts/     # runs & figures (gitignored)
```

---

## Reproducibility

* **Seeding:** Use `train.seed` (`configs/train/*.yaml`) for deterministic ODE initializations and random weights drawn on the PSR graph.
* **Artifacts:** Every run creates a timestamped directory under `artifacts/runs/<dataset>/...` containing configs, predictions (`.npz`), and figures (`.png`).
* **Testing:** Run `pytest -q` from the repo root.

---

## Citing

If this software or its ideas are useful in your work, please cite:

* **Software**

  ```bibtex
  @software{Singh_PSR_2025,
    author  = {Pradeep Singh and Ashutosh Kumar and Sutirtha Ghosh and Balasubramanian Raman},
    title   = {Poincar{\'e} Section Reservoirs (PSR): Topologically Informed Echo State Networks},
    year    = {2025},
    version = {0.1.0},
    url     = {https://github.com/deepdyn/poincare-section-reservoir}
  }
  ```

  Or use the included `CITATION.cff`.

* **Paper (under review)**

  ```
  Singh, P.*, Kumar, A., Ghosh, S., Raman, B.
  "Topologically Informed Echo State Networks via Poincaré Return Maps for Chaotic Time-Series",
  under review at Neural Networks (Elsevier), 2025.
  ```

---

## License

MIT License — see `LICENSE`.

---

## Acknowledgments

This project emerged at the Machine Intelligence Lab, IIT Roorkee. We thank colleagues and students for feedback on early prototypes and ablations.

