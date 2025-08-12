from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..config import Config
from ..data.generators import lorenz63, rossler, chen_ueta
from ..geometry.poincare import plane_from_point_normal, PoincarePlane
from ..geometry.sections import SectionEnsemble
from ..metrics import nrmse, evaluate_horizons
from ..readout.trainer import fit_open_loop, predict_closed_loop
from ..reservoir.ms_psr import MSPSR
from ..utils.io import make_run_dir, save_npz, save_yaml
from ..viz.plots import plot_overlay, plot_phase3d, plot_eigenspectrum
from ..viz.sections_viz import plot_section_points, plot_section_bins, plot_section_transitions


@dataclass
class RunResult:
    run_dir: Path
    metrics: Dict[int, float]
    y_true: np.ndarray
    y_hat: np.ndarray


def _make_planes(X: np.ndarray, sections: int, rotation_deg: float) -> SectionEnsemble:
    """
    Build a base plane through mean(X) with normal along e_z,
    then add small rotations around x-axis to create an ensemble.
    """
    mu = X.mean(axis=0)
    d = X.shape[1]
    n = np.zeros(d, dtype=float)
    n[min(2, d - 1)] = 1.0  # use z if available, else last axis
    base = plane_from_point_normal(mu, n)

    if sections <= 1:
        return SectionEnsemble([base])

    # symmetric small rotations around x-axis (in R^3)
    angles = np.linspace(-rotation_deg, rotation_deg, sections)
    angles = np.deg2rad(angles)
    axes = [np.array([1.0, 0.0, 0.0])] * sections
    rotations = list(zip(axes, angles))
    return SectionEnsemble.from_plane_and_rotations(base, rotations)


def _pick_dataset(cfg: Config):
    name = cfg.dataset.name.lower()
    if name in {"lorenz", "lorenz63"}:
        res = lorenz63(T=cfg.dataset.params.get("T", 20_000), dt=cfg.dataset.params.get("dt", 0.01),
                       seed=cfg.train.seed)
    elif name in {"rossler", "rössler"}:
        res = rossler(T=cfg.dataset.params.get("T", 20_000), dt=cfg.dataset.params.get("dt", 0.05),
                      seed=cfg.train.seed)
    elif name in {"chen", "chen_ueta", "chen–ueta", "chen-ueta"}:
        res = chen_ueta(T=cfg.dataset.params.get("T", 20_000), dt=cfg.dataset.params.get("dt", 0.01),
                        seed=cfg.train.seed)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    return res.t, res.x  # (t, X)


def run_experiment(cfg: Config) -> RunResult:
    """
    End-to-end:
      - generate dataset trajectory
      - build planes & MS-PSR graph → ESN
      - fit ridge readout (open-loop)
      - closed-loop rollout for max horizon and compute NRMSE@H
      - save artifacts and a few plots
    """
    t, X = _pick_dataset(cfg)

    # Planes & ensemble
    ensemble = _make_planes(X, sections=cfg.model.sections, rotation_deg=cfg.model.section_rotation_deg)

    # Build reservoir from section hits
    psr = MSPSR(
        ensemble=ensemble,
        bins=(cfg.model.bins, cfg.model.bins),
        leak=cfg.model.leak,
        spectral_radius=cfg.model.spectral_radius,
        input_scale=cfg.model.input_scale,
        activation=cfg.model.activation,
    )
    esn, info = psr.build_from_trajectory(X, U=None, oriented=False, random_state=cfg.train.seed)

    # Train readout in open-loop (teacher forcing)
    readout, train_info = fit_open_loop(esn, series=X, washout=cfg.train.washout, alpha=cfg.train.ridge_alpha)

    # Closed-loop rollout from y0 at washout
    Hs = cfg.eval.horizons
    Hmax = int(max(Hs))
    y0 = X[cfg.train.washout]
    yhat = predict_closed_loop(esn, readout, y0=y0, steps=Hmax)
    ytrue = X[cfg.train.washout + 1 : cfg.train.washout + 1 + Hmax]

    # Metrics
    metrics = evaluate_horizons(ytrue, yhat, Hs, metric=lambda a, b: float(nrmse(a, b)))

    # Save artifacts
    run_dir = make_run_dir(root=cfg.artifacts_dir, dataset=cfg.dataset.name)
    save_npz(run_dir / "predictions.npz", y_true=ytrue, y_hat=yhat, t=t)
    save_yaml(run_dir / "config.yaml", {"config": cfg.__dict__, "train_info": train_info})

    # Plots
    try:
        plot_phase3d(X).savefig(run_dir / "phase3d.png", dpi=300, bbox_inches="tight")
        plot_overlay(ytrue, yhat, dims=(0, 1, 2)).savefig(run_dir / "overlay.png", dpi=300, bbox_inches="tight")
        plot_eigenspectrum(esn.W).savefig(run_dir / "eigenspectrum.png", dpi=300, bbox_inches="tight")
        plot_section_points(info["uv"]).savefig(run_dir / "section_points.png", dpi=300, bbox_inches="tight")
        plot_section_bins(info["uv"], bins=(cfg.model.bins, cfg.model.bins), ranges=info["ranges"]).savefig(
            run_dir / "section_bins.png", dpi=300, bbox_inches="tight"
        )
        plot_section_transitions(info["bin_ids"]).savefig(run_dir / "section_transitions.png", dpi=300, bbox_inches="tight")
    except Exception:
        pass  # plotting is best-effort

    return RunResult(run_dir=run_dir, metrics=metrics, y_true=ytrue, y_hat=yhat)
