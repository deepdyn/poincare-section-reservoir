from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PoincarePlane:
    """
    Plane defined by unit normal n and offset d such that f(x) = n·x - d = 0.
    """
    n: np.ndarray  # (d,)
    d: float       # scalar

    def __post_init__(self):
        n = np.asarray(self.n, dtype=float).reshape(-1)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError("Normal cannot be zero.")
        object.__setattr__(self, "n", n / norm)
        object.__setattr__(self, "d", float(self.d) / norm)

    def f(self, x: np.ndarray) -> float:
        return float(np.dot(self.n, x) - self.d)

    def signed_values(self, x: np.ndarray) -> np.ndarray:
        return x @ self.n - self.d

    def hits(self, x_prev: np.ndarray, x_curr: np.ndarray, *, oriented: bool = False) -> bool:
        """
        Returns True if segment [x_prev, x_curr] crosses the plane.
        If oriented=True, requires f(prev) < 0 and f(curr) >= 0.
        """
        f0 = self.f(x_prev)
        f1 = self.f(x_curr)
        if oriented:
            return (f0 < 0.0) and (f1 >= 0.0)
        return (f0 <= 0.0 and f1 >= 0.0) or (f0 >= 0.0 and f1 <= 0.0)

    def hit_fraction(self, x_prev: np.ndarray, x_curr: np.ndarray) -> float:
        """
        Linear interpolation of f along the segment to find t* in [0,1].
        """
        f0 = self.f(x_prev)
        f1 = self.f(x_curr)
        denom = f0 - f1
        if denom == 0:
            return 0.5
        t = f0 / denom
        return float(np.clip(t, 0.0, 1.0))

    def hit_point(self, x_prev: np.ndarray, x_curr: np.ndarray) -> np.ndarray:
        t = self.hit_fraction(x_prev, x_curr)
        return (1.0 - t) * x_prev + t * x_curr

    @lru_cache(maxsize=8)
    def basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orthonormal (u, v) spanning the plane in R^3+ using Gram–Schmidt.
        """
        d = self.n.shape[0]
        # Pick a vector not parallel to n
        a = np.zeros(d)
        a[0] = 1.0
        if np.allclose(np.abs(self.n[0]), 1.0, atol=1e-7):
            a = np.eye(d)[1]
        u = a - np.dot(a, self.n) * self.n
        u /= max(np.linalg.norm(u), 1e-12)
        v = np.cross(self.n, u) if d == 3 else _orth_perp(self.n, u)
        v /= max(np.linalg.norm(v), 1e-12)
        return u, v

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project point(s) onto the 2D coordinates in the plane basis (u,v).
        """
        u, v = self.basis()
        x = np.atleast_2d(x)
        # subtract any point on the plane along n: p0 = d * n
        p0 = self.d * self.n
        y = x - p0
        U = y @ u
        V = y @ v
        return np.stack([U, V], axis=-1)

    def intersect_trajectory(self, X: np.ndarray, oriented: bool = False):
        """
        Given X=(T,d), return arrays:
          idx: indices i where segment [X[i], X[i+1]] hits
          tstar: fractions in [0,1]
          pts: hit points (m,d)
          uv: projected 2D coords (m,2)
        """
        X = np.asarray(X, dtype=float)
        s = self.signed_values(X)
        s0 = s[:-1]
        s1 = s[1:]
        if oriented:
            mask = (s0 < 0.0) & (s1 >= 0.0)
        else:
            mask = ((s0 <= 0.0) & (s1 >= 0.0)) | ((s0 >= 0.0) & (s1 <= 0.0))
        idx = np.nonzero(mask)[0]
        tstars = []
        pts = []
        for i in idx:
            t = (s0[i]) / (s0[i] - s1[i]) if (s0[i] - s1[i]) != 0 else 0.5
            t = float(np.clip(t, 0.0, 1.0))
            xh = (1.0 - t) * X[i] + t * X[i + 1]
            tstars.append(t)
            pts.append(xh)
        if len(idx) == 0:
            return idx, np.array([]), np.empty((0, X.shape[1])), np.empty((0, 2))
        pts = np.stack(pts, axis=0)
        uv = self.project(pts)
        return idx, np.array(tstars, dtype=float), pts, uv


def plane_from_point_normal(point: np.ndarray, normal: np.ndarray) -> PoincarePlane:
    n = np.asarray(normal, dtype=float).reshape(-1)
    p = np.asarray(point, dtype=float).reshape(-1)
    d = float(np.dot(n, p))  # since f(x) = n·x - d
    return PoincarePlane(n=n, d=d)


def _orth_perp(n: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Fallback to construct a second vector orthogonal to both n and u in R^d (d>=2).
    """
    d = n.shape[0]
    # Solve for v in span{e_i} s.t. [n; u]·v = 0 (least squares)
    A = np.stack([n, u], axis=0)  # (2,d)
    # Any vector orthogonal to rows of A lives in nullspace(A)
    # Use SVD and take smallest singular vector:
    _, _, Vt = np.linalg.svd(A)
    v = Vt[-1]
    return v
