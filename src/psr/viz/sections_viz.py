from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_section_points(uv: np.ndarray, s: int = 10):
    uv = np.asarray(uv)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(uv[:, 0], uv[:, 1], s=s, alpha=0.7)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title("Section hits (u, v)")
    ax.grid(True, alpha=0.2)
    return fig


def plot_section_bins(
    uv: np.ndarray,
    bins: tuple[int, int] = (64, 64),
    ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
):
    """
    Draw rectangular bin grid over the (u,v) scatter.
    """
    uv = np.asarray(uv)
    u, v = uv[:, 0], uv[:, 1]
    if ranges is None:
        umin, umax = float(np.min(u)), float(np.max(u))
        vmin, vmax = float(np.min(v)), float(np.max(v))
    else:
        (umin, umax), (vmin, vmax) = ranges

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(u, v, s=5, alpha=0.6)
    nu, nv = bins
    for i in range(1, nu):
        x = umin + (umax - umin) * i / nu
        ax.plot([x, x], [vmin, vmax], linewidth=0.5)
    for j in range(1, nv):
        y = vmin + (vmax - vmin) * j / nv
        ax.plot([umin, umax], [y, y], linewidth=0.5)
    ax.set_xlim(umin, umax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title("Bins on section")
    ax.grid(True, alpha=0.2)
    return fig


def plot_section_transitions(bin_ids: np.ndarray):
    """
    Simple plot of bin index over hit order (diagnostic).
    """
    b = np.asarray(bin_ids, dtype=int)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(len(b)), b, linewidth=0.8)
    ax.set_xlabel("hit index")
    ax.set_ylabel("bin id")
    ax.set_title("Transitions across bins")
    ax.grid(True, alpha=0.3)
    return fig
