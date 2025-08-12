from __future__ import annotations

import numpy as np


def rmse(y: np.ndarray, yhat: np.ndarray, axis: None | int | tuple[int, ...] = None) -> float | np.ndarray:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.sqrt(np.mean((y - yhat) ** 2, axis=axis))


def nrmse(y: np.ndarray, yhat: np.ndarray, axis: None | int | tuple[int, ...] = None) -> float | np.ndarray:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    num = np.sqrt(np.mean((y - yhat) ** 2, axis=axis))
    denom = np.std(y, axis=axis)
    denom = np.where(denom == 0, 1.0, denom)
    return num / denom


def mape(y: np.ndarray, yhat: np.ndarray, axis: None | int | tuple[int, ...] = None, eps: float = 1e-8):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.mean(np.abs((y - yhat) / np.maximum(np.abs(y), eps)), axis=axis)
