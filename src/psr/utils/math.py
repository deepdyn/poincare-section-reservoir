from __future__ import annotations

from typing import Optional

import numpy as np


def spectral_radius(W: np.ndarray) -> float:
    """
    Compute the spectral radius (max |eigenvalue|) of a square matrix.
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square 2D array")
    eigvals = np.linalg.eigvals(W)
    return float(np.max(np.abs(eigvals)))


def rescale_spectral_radius(W: np.ndarray, target_rho: float, current_rho: Optional[float] = None):
    """
    Scale W so that its spectral radius becomes target_rho.

    Returns:
        W_scaled, scale_factor
    """
    if current_rho is None:
        current_rho = spectral_radius(W)
    if current_rho == 0:
        return W.copy(), 1.0
    scale = target_rho / current_rho
    return (W * scale).astype(W.dtype, copy=False), float(scale)


def l2_normalize(x: np.ndarray, axis: Optional[int] = None, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def zscore(x: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    mu = np.mean(x, axis=axis, keepdims=True)
    sig = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / np.maximum(sig, eps)


def minmax_scale(x: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    xmin = np.min(x, axis=axis, keepdims=True)
    xmax = np.max(x, axis=axis, keepdims=True)
    return (x - xmin) / np.maximum(xmax - xmin, eps)


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def crossing_time_linear(f_prev: float, f_curr: float) -> float:
    """
    Linear-in-time crossing fraction t* in [0,1] at which f crosses zero on [prev,curr].
    Assumes f_prev and f_curr have opposite signs.
        f(t) = (1 - t) * f_prev + t * f_curr; find t s.t. f(t)=0 => t* = f_prev / (f_prev - f_curr)
    """
    denom = (f_prev - f_curr)
    if denom == 0:
        return 0.5  # degenerate; return midpoint
    t = f_prev / denom
    # Clamp for numerical safety
    if not np.isfinite(t):
        return 0.5
    return float(np.clip(t, 0.0, 1.0))
