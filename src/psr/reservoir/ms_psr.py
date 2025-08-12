from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from ..geometry.sections import SectionEnsemble
from ..graphs.adjacency import adjacency_from_section_points, bins_from_points_rect
from ..graphs.scaling import set_spectral_radius
from ..utils.math import l2_normalize
from .esn import ESN


def build_W_from_adjacency(
    A: sp.csr_matrix | np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    weight_scale: float = 1.0,
    signed: bool = True,
) -> np.ndarray:
    """
    Replace nonzero entries with random weights (signed or positive).
    """
    rng = np.random.default_rng() if rng is None else rng
    if sp.issparse(A):
        A = A.tocsr()
        W = A.copy().astype(float)
        W.data = (rng.uniform(-1.0, 1.0, size=W.nnz) if signed else rng.random(W.nnz)) * weight_scale
        return W.toarray()
    else:
        W = np.zeros_like(A, dtype=float)
        nz = np.nonzero(A)
        vals = rng.uniform(-1.0, 1.0, size=len(nz[0])) if signed else rng.random(len(nz[0]))
        W[nz] = vals * weight_scale
        return W


@dataclass
class MSPSR:
    """
    Multi-Section Poincaré Section Reservoir wiring:
      trajectory -> section hits (uv) -> bins -> adjacency -> W -> ESN
    """
    ensemble: SectionEnsemble
    bins: Tuple[int, int] = (64, 64)
    leak: float = 0.3
    spectral_radius: float = 0.9
    input_scale: float = 1.0
    activation: str = "tanh"

    def build_from_trajectory(
        self,
        X: np.ndarray,             # (T, d) full state trajectory used to define section hits
        U: Optional[np.ndarray] = None,  # (T, m) driving inputs (defaults to using state as input)
        *,
        oriented: bool = False,
        random_state: Optional[int] = 0,
    ) -> tuple[ESN, dict]:
        """
        Returns:
            esn: instantiated ESN wired from PSR graph
            info: dict with uv, bin_ids, adjacency, ranges, etc.
        """
        hits = self.ensemble.find_hits(X, oriented=oriented)
        if len(hits) == 0:
            raise ValueError("No section hits detected; adjust plane or trajectory length.")

        uv = np.stack([h.uv for h in hits], axis=0)  # (M,2)

        # Binning on the plane → transitions → adjacency
        bin_ids, ranges = bins_from_points_rect(uv, bins=self.bins, ranges=None)
        A = adjacency_from_section_points(uv, bins=self.bins, directed=True, normalize="row", sparse=True)

        # Adjacency → random-weight recurrent W, scaled to target spectral radius
        rng = np.random.default_rng(random_state)
        W_raw = build_W_from_adjacency(A, rng=rng, signed=True, weight_scale=1.0)
        W = set_spectral_radius(W_raw, target_rho=self.spectral_radius)

        # Input mapping Win: if U is None, use the state X as input (m = d)
        if U is None:
            U = X
        m = U.shape[1]
        N = W.shape[0]
        Win = rng.normal(scale=1.0 / np.sqrt(m), size=(N, m))

        esn = ESN(
            W=W,
            Win=Win,
            bias=np.zeros(N, dtype=float),
            leak=self.leak,
            activation=self.activation,
            mask=None,
            input_scale=self.input_scale,
        )

        info = {
            "uv": uv,
            "bin_ids": bin_ids,
            "ranges": ranges,
            "adjacency": A,
            "W_raw": W_raw,
            "W": W,
        }
        return esn, info

    # Convenience pass-throughs

    def states(self, esn: ESN, U: np.ndarray, *, washout: int = 0) -> np.ndarray:
        X = esn.forward(U)
        return X[washout:]

    def normalize_inputs(self, U: np.ndarray) -> np.ndarray:
        return l2_normalize(U, axis=0)
