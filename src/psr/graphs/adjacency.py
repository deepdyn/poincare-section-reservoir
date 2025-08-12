from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp


def bins_from_points_rect(
    uv: np.ndarray,
    bins: Tuple[int, int] = (64, 64),
    ranges: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Map 2D points to rectangular bins.
    Returns bin indices in [0, B-1] (row-major) and the ranges used.
    """
    u = uv[:, 0]
    v = uv[:, 1]
    if ranges is None:
        umin, umax = np.min(u), np.max(u)
        vmin, vmax = np.min(v), np.max(v)
        # pad slightly to include boundary points
        pad_u = 1e-9 * max(1.0, abs(umax - umin))
        pad_v = 1e-9 * max(1.0, abs(vmax - vmin))
        ranges = ((umin - pad_u, umax + pad_u), (vmin - pad_v, vmax + pad_v))
    (umin, umax), (vmin, vmax) = ranges
    nu, nv = bins

    # normalized positions in [0,1)
    U = (u - umin) / max(umax - umin, 1e-12)
    V = (v - vmin) / max(vmax - vmin, 1e-12)
    iu = np.minimum((U * nu).astype(int), nu - 1)
    iv = np.minimum((V * nv).astype(int), nv - 1)
    bin_ids = iu * nv + iv
    return bin_ids, ranges


def build_transition_adjacency(
    bin_ids: np.ndarray,
    n_bins: int,
    directed: bool = True,
    normalize: Optional[str] = "row",
    sparse: bool = True,
) -> sp.csr_matrix | np.ndarray:
    """
    Count transitions between successive bin IDs to build adjacency.

    normalize: None | 'row' | 'sym'
    """
    b = np.asarray(bin_ids, dtype=int)
    if len(b) < 2:
        A = sp.csr_matrix((n_bins, n_bins)) if sparse else np.zeros((n_bins, n_bins), dtype=float)
        return A

    i = b[:-1]
    j = b[1:]
    if not directed:
        # undirected: symmetrize transitions
        ii = np.concatenate([i, j])
        jj = np.concatenate([j, i])
        data = np.ones_like(ii, dtype=float)
    else:
        ii = i
        jj = j
        data = np.ones_like(i, dtype=float)

    A = sp.csr_matrix((data, (ii, jj)), shape=(n_bins, n_bins)) if sparse else np.zeros((n_bins, n_bins), dtype=float)
    if not sparse:
        for s, t in zip(ii, jj):
            A[s, t] += 1.0

    if normalize:
        A = normalize_adjacency(A, mode=normalize)
    return A


def adjacency_from_section_points(
    uv: np.ndarray,
    bins: Tuple[int, int] = (64, 64),
    directed: bool = True,
    normalize: Optional[str] = "row",
    sparse: bool = True,
):
    """
    Convenience pipeline: points -> bins -> transition adjacency.
    """
    bin_ids, _ = bins_from_points_rect(uv, bins=bins, ranges=None)
    n_bins = bins[0] * bins[1]
    return build_transition_adjacency(bin_ids, n_bins=n_bins, directed=directed, normalize=normalize, sparse=sparse)


def normalize_adjacency(A: sp.csr_matrix | np.ndarray, mode: str = "row") -> sp.csr_matrix | np.ndarray:
    """
    mode: 'row' -> row-stochastic; 'sym' -> D^{-1/2} A D^{-1/2}
    """
    if sp.issparse(A):
        if mode == "row":
            d = np.array(A.sum(axis=1)).ravel()
            d[d == 0] = 1.0
            Dinv = sp.diags(1.0 / d)
            return Dinv @ A
        elif mode == "sym":
            d = np.array(A.sum(axis=1)).ravel()
            d[d == 0] = 1.0
            Dm = sp.diags(1.0 / np.sqrt(d))
            return Dm @ A @ Dm
        else:
            return A
    else:
        if mode == "row":
            d = A.sum(axis=1, keepdims=True)
            d[d == 0] = 1.0
            return A / d
        elif mode == "sym":
            d = A.sum(axis=1)
            d[d == 0] = 1.0
            Dm = 1.0 / np.sqrt(d)
            return (A * Dm[:, None]) * Dm[None, :]
        else:
            return A
