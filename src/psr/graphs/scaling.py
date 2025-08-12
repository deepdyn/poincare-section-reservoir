from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..utils.math import rescale_spectral_radius as _rescale_dense


def spectral_radius_power(A: sp.csr_matrix | np.ndarray, iters: int = 200) -> float:
    """
    Power-iteration estimate of spectral radius (largest |eigenvalue|).
    Works for sparse or dense nonnegative/real matrices; for general matrices it estimates ||A||.
    """
    n = A.shape[0]
    x = np.random.default_rng(0).normal(size=n)
    x /= np.linalg.norm(x) + 1e-12
    for _ in range(iters):
        x = A @ x if not sp.issparse(A) else A.dot(x)
        nrm = np.linalg.norm(x)
        if nrm == 0:
            return 0.0
        x /= nrm
    y = A @ x if not sp.issparse(A) else A.dot(x)
    num = float(np.linalg.norm(y))
    den = float(np.linalg.norm(x))
    return num / max(den, 1e-12)


def set_spectral_radius(
    A: sp.csr_matrix | np.ndarray,
    target_rho: float,
    current_rho: Optional[float] = None,
) -> sp.csr_matrix | np.ndarray:
    """
    Scale A so that spectral radius ≈ target_rho (uses dense exact or power iteration).
    """
    if sp.issparse(A):
        rho = spectral_radius_power(A) if current_rho is None else float(current_rho)
        if rho == 0:
            return A.copy()
        scale = target_rho / rho
        return (A * scale).tocsr()
    else:
        W, scale = _rescale_dense(np.asarray(A), target_rho=target_rho, current_rho=current_rho)
        return W


def row_stochastic(A: sp.csr_matrix | np.ndarray) -> sp.csr_matrix | np.ndarray:
    """
    Row-normalize so each row sums to 1 (if sum>0).
    """
    if sp.issparse(A):
        d = np.array(A.sum(axis=1)).ravel()
        d[d == 0] = 1.0
        Dinv = sp.diags(1.0 / d)
        return Dinv @ A
    else:
        d = A.sum(axis=1, keepdims=True)
        d[d == 0] = 1.0
        return A / d


def effective_radius_bound(rho_W: float, leak: float) -> float:
    """
    Upper bound on spectral radius of (1-λ)I + λW:
    ρ((1-λ)I + λW) ≤ (1-λ) + λ ρ(W).
    """
    lam = float(leak)
    return (1.0 - lam) + lam * float(rho_W)
