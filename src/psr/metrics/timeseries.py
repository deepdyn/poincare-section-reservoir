from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import numpy as np


def _slice_horizon(y: np.ndarray, yhat: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    H = int(H)
    T = min(len(y), len(yhat))
    h = min(H, T)
    return y[:h], yhat[:h]


def evaluate_horizons(
    y: np.ndarray,
    yhat: np.ndarray,
    horizons: Iterable[int],
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> Dict[int, float]:
    """
    Compute metric(y[:H], yhat[:H]) for each H in horizons.
    """
    out: Dict[int, float] = {}
    for H in horizons:
        yt, yh = _slice_horizon(y, yhat, H)
        out[int(H)] = float(metric(yt, yh))
    return out
