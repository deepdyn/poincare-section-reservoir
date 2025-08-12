from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .poincare import PoincarePlane


@dataclass(frozen=True)
class SectionHit:
    seg_index: int          # i such that crossing occurs on [X[i], X[i+1]]
    tstar: float            # fraction in [0,1]
    point: np.ndarray       # (d,)
    uv: np.ndarray          # (2,)
    plane_id: int           # which plane


class SectionEnsemble:
    def __init__(self, planes: Iterable[PoincarePlane]):
        self.planes: List[PoincarePlane] = list(planes)

    def find_hits(self, X: np.ndarray, oriented: bool = False) -> List[SectionHit]:
        """
        Collect and merge hits across all planes. Returns a list sorted by segment index then t*.
        """
        hits: List[SectionHit] = []
        for pid, pl in enumerate(self.planes):
            idx, tstar, pts, uv = pl.intersect_trajectory(X, oriented=oriented)
            for k, i in enumerate(idx):
                hits.append(
                    SectionHit(
                        seg_index=int(i),
                        tstar=float(tstar[k]),
                        point=pts[k],
                        uv=uv[k],
                        plane_id=pid,
                    )
                )
        hits.sort(key=lambda h: (h.seg_index, h.tstar))
        return hits

    def project(self, points: np.ndarray, plane_id: int = 0) -> np.ndarray:
        return self.planes[plane_id].project(points)

    @staticmethod
    def from_plane_and_rotations(
        base: PoincarePlane,
        rotations: Iterable[Tuple[np.ndarray, float]],
    ) -> "SectionEnsemble":
        """
        Construct planes by rotating base normal around given axes by angles (radians).
        Only meaningful in R^3.
        """
        planes = [base]
        n0 = base.n
        p0_on_plane = base.d * base.n
        for axis, angle in rotations:
            axis = np.asarray(axis, dtype=float).reshape(-1)
            axis /= max(np.linalg.norm(axis), 1e-12)
            R = _rotation_matrix(axis, angle)
            n_new = R @ n0
            d_new = float(np.dot(n_new, p0_on_plane))  # keep passing through same p0
            planes.append(PoincarePlane(n=n_new, d=d_new))
        return SectionEnsemble(planes)


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Rodrigues' rotation formula (3D).
    """
    ax = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = ax
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    I = np.eye(3)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
