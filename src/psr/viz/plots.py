from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_overlay(y: np.ndarray, yhat: np.ndarray, dims=(0, 1, 2), T: int | None = None):
    """
    Overlay true vs predicted coordinates over time.
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    if T is not None:
        y = y[:T]
        yhat = yhat[:T]

    t = np.arange(len(y))
    D = len(dims)
    fig, axes = plt.subplots(D, 1, figsize=(8, 2.6 * D), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, dims):
        ax.plot(t, y[:, d], label=f"y[{d}]")
        ax.plot(t, yhat[:, d], label=f"yhat[{d}]", linestyle="--")
        ax.set_ylabel(f"dim {d}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t")
    axes[0].legend(loc="upper right")
    return fig


def plot_phase3d(traj: np.ndarray, elev: float = 15.0, azim: float = 45.0):
    """
    3D phase portrait for R^3 trajectories.
    """
    x = np.asarray(traj)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x[:, 0], x[:, 1], x[:, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    return fig


def plot_eigenspectrum(W: np.ndarray):
    """
    Plot eigenvalues in the complex plane with unit circle.
    """
    import numpy.linalg as npl

    eig = npl.eigvals(np.asarray(W))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(eig.real, eig.imag, s=12)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), linewidth=1.0)
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Eigenspectrum")
    return fig
