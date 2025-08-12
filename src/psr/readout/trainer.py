from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from ..reservoir.esn import ESN
from .linear import RidgeReadout


def collect_states_targets(
    esn: ESN,
    series: np.ndarray,
    *,
    washout: int = 500,
    teacher_forcing: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a multivariate time series Y (T, d), run the ESN and build (S, Y_target)
    pairs for one-step prediction:
        target at t is Y[t+1]; states are S[t] (post-washout).
    If teacher_forcing=True, inputs fed to ESN are ground-truth Y.
    """
    Y = np.asarray(series, dtype=float)
    if teacher_forcing:
        U = Y
    else:
        # If not using teacher forcing, still need inputs; use Y (common baseline).
        U = Y

    X = esn.forward(U)  # (T, N)
    # Align states with next-step targets
    S = X[washout:-1]
    T_eff = S.shape[0]
    Y_target = Y[washout + 1 : washout + 1 + T_eff]
    return S, Y_target


def fit_open_loop(
    esn: ESN,
    series: np.ndarray,
    *,
    washout: int = 500,
    alpha: float = 1e-3,
) -> Tuple[RidgeReadout, Dict]:
    """
    Fit a ridge readout for one-step forecasting in open-loop (teacher forcing).
    """
    S, Yt = collect_states_targets(esn, series, washout=washout, teacher_forcing=True)
    readout = RidgeReadout(alpha=alpha, fit_intercept=True).fit(S, Yt)
    info = {"T_train": S.shape[0], "washout": washout}
    return readout, info


def predict_open_loop(
    esn: ESN,
    readout: RidgeReadout,
    series: np.ndarray,
    *,
    washout: int = 500,
) -> np.ndarray:
    """
    One-step predictions under teacher forcing for the evaluation segment.
    Returns predictions aligned with Y[washout+1:].
    """
    S, _Y = collect_states_targets(esn, series, washout=washout, teacher_forcing=True)
    return readout.predict(S)


def predict_closed_loop(
    esn: ESN,
    readout: RidgeReadout,
    y0: np.ndarray,
    steps: int,
) -> np.ndarray:
    """
    Closed-loop rollout: feed predictions back as inputs.
    Requires input_dim == output_dim.
    """
    y = np.asarray(y0, dtype=float).reshape(1, -1)
    if y.shape[1] != esn.m:
        raise ValueError(f"y0 must have dimension {esn.m}, got {y.shape[1]}")
    preds = []
    x = None
    esn.reset(x0=x)
    u_t = y[0]
    # Prime one step from y0
    x_t = esn.step(u_t)
    for _ in range(steps):
        yhat = readout.predict(x_t.reshape(1, -1))[0]
        preds.append(yhat)
        # feed back
        u_t = yhat
        x_t = esn.step(u_t)
    return np.stack(preds, axis=0)
