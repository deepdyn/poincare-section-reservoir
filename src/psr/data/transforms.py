from __future__ import annotations

from typing import Tuple

import numpy as np


def zscore_ts(x: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


def minmax_ts(x: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    xmin = np.min(x, axis=axis, keepdims=True)
    xmax = np.max(x, axis=axis, keepdims=True)
    return (x - xmin) / np.maximum(xmax - xmin, eps)


def window_time_series(
    x: np.ndarray,
    input_len: int,
    pred_len: int = 1,
    stride: int = 1,
    flatten_inputs: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (X, Y) windows for supervised forecasting.
    x: (T, d)
    X: (N, input_len * d) if flatten_inputs else (N, input_len, d)
    Y: (N, pred_len, d)
    """
    x = np.asarray(x)
    T, d = x.shape
    n = (T - input_len - pred_len) // stride + 1
    if n <= 0:
        raise ValueError("Not enough timesteps for the requested windows")

    X = []
    Y = []
    for i in range(n):
        s = i * stride
        e = s + input_len
        X_i = x[s:e]
        Y_i = x[e : e + pred_len]
        if flatten_inputs:
            X.append(X_i.reshape(-1))
        else:
            X.append(X_i)
        Y.append(Y_i)
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return X, Y


def train_test_split_time_series(
    x: np.ndarray,
    train_frac: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split along time (no shuffling).
    """
    T = len(x)
    t = int(round(T * train_frac))
    return x[:t], x[t:]
