from __future__ import annotations

import numpy as np


def polar_coords_on_plane(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert plane coordinates (u,v) to polar (r, theta) with atan2.
    """
    u = uv[:, 0]
    v = uv[:, 1]
    r = np.sqrt(u * u + v * v)
    theta = np.arctan2(v, u)
    return r, theta


def return_map_pairs(uv: np.ndarray, mode: str = "angle") -> tuple[np.ndarray, np.ndarray]:
    """
    Build (x_k, x_{k+1}) return map pairs from section intersections.
    mode: 'angle' -> theta_k vs theta_{k+1}; 'radius' -> r_k vs r_{k+1}
    """
    r, th = polar_coords_on_plane(uv)
    if mode == "angle":
        x = th
    elif mode == "radius":
        x = r
    else:
        raise ValueError("mode must be 'angle' or 'radius'")
    if len(x) < 2:
        return np.empty((0,)), np.empty((0,))
    return x[:-1], x[1:]
