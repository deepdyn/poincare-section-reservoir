from __future__ import annotations

from typing import Callable, Optional

import numpy as np


# -----------------------
# Activations
# -----------------------

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _identity(x: np.ndarray) -> np.ndarray:
    return x


_ACTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "tanh": _tanh,
    "relu": _relu,
    "id": _identity,
    "identity": _identity,
    "linear": _identity,
}


def get_activation(name: str | Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    if callable(name):
        return name  # type: ignore[return-value]
    key = str(name).lower()
    if key not in _ACTS:
        raise ValueError(f"Unknown activation '{name}'. Available: {sorted(_ACTS)}")
    return _ACTS[key]


# -----------------------
# Cell primitives
# -----------------------

def apply_mask(W: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Elementwise mask of the recurrent matrix (e.g., block sparsity).
    mask must be broadcastable to W's shape.
    """
    if mask is None:
        return W
    return W * mask


def leaky_update(
    x_prev: np.ndarray,
    preact: np.ndarray,
    *,
    leak: float = 0.3,
    activation: str | Callable[[np.ndarray], np.ndarray] = "tanh",
) -> np.ndarray:
    """
    One leaky-integrator step:
        x <- (1 - λ) x + λ φ(preact)
    """
    act = get_activation(activation)
    return (1.0 - leak) * x_prev + leak * act(preact)
