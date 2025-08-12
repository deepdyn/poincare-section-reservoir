from __future__ import annotations

import numpy as np

from ..graphs.scaling import effective_radius_bound


def vpt(y: np.ndarray, yhat: np.ndarray, *, tol: float = 0.1) -> int:
    """
    Valid Prediction Time: first index t where relative error exceeds tol.
      rel_err(t) = ||y_t - yhat_t|| / (||y_t|| + 1e-12)
    Returns T if tolerance never exceeded.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    e = np.linalg.norm(y - yhat, axis=-1) / (np.linalg.norm(y, axis=-1) + 1e-12)
    bad = np.nonzero(e > tol)[0]
    return int(bad[0]) if bad.size > 0 else len(e)


def spectral_radius_bound(rho_W: float, leak: float) -> float:
    """
    ρ((1-λ)I + λW) ≤ (1-λ) + λ ρ(W)
    """
    return effective_radius_bound(rho_W, leak)
