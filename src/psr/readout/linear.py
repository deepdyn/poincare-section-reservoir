from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import MultiTaskLasso


def _center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    return X - mu, mu.ravel()


@dataclass
class RidgeReadout:
    alpha: float = 1e-3
    fit_intercept: bool = True

    # Learned parameters
    W: Optional[np.ndarray] = None  # (N, d_out)
    b: Optional[np.ndarray] = None  # (d_out,)

    def fit(self, S: np.ndarray, Y: np.ndarray) -> "RidgeReadout":
        """
        Closed-form ridge: W = (S^T S + α I)^{-1} S^T Y.
        S: (T, N) states, Y: (T, d_out)
        """
        S = np.asarray(S, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if self.fit_intercept:
            S, muS = _center(S)
            Y, muY = _center(Y)
        else:
            muS = np.zeros(S.shape[1])
            muY = np.zeros(Y.shape[1])

        N = S.shape[1]
        A = S.T @ S
        A.flat[:: N + 1] += self.alpha  # add αI
        B = S.T @ Y
        W = np.linalg.solve(A, B)
        b = muY - muS @ W if self.fit_intercept else np.zeros(Y.shape[1])
        self.W, self.b = W, b
        return self

    def predict(self, S: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None:
            raise RuntimeError("Call fit before predict")
        S = np.asarray(S, dtype=float)
        return S @ self.W + self.b


@dataclass
class LassoReadout:
    alpha: float = 1e-3
    fit_intercept: bool = True
    max_iter: int = 2000

    _model: Optional[MultiTaskLasso] = None

    def fit(self, S: np.ndarray, Y: np.ndarray) -> "LassoReadout":
        S = np.asarray(S, dtype=float)
        Y = np.asarray(Y, dtype=float)
        self._model = MultiTaskLasso(alpha=self.alpha, fit_intercept=self.fit_intercept, max_iter=self.max_iter)
        self._model.fit(S, Y)
        return self

    def predict(self, S: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit before predict")
        return self._model.predict(np.asarray(S, dtype=float))


@dataclass
class RLSReadout:
    """
    Recursive Least Squares for online readout learning.

    Parameters:
        lam: forgetting factor (λ=1.0 => no forgetting)
        delta: initial P = (1/δ) I
    """
    lam: float = 1.0
    delta: float = 1e3

    W: Optional[np.ndarray] = None  # (N, d_out)
    P: Optional[np.ndarray] = None  # (N, N)
    b: Optional[np.ndarray] = None  # (d_out,)

    def init(self, N: int, d_out: int) -> None:
        self.W = np.zeros((N, d_out), dtype=float)
        self.P = (1.0 / self.delta) * np.eye(N, dtype=float)
        self.b = np.zeros(d_out, dtype=float)

    def partial_fit(self, s_t: np.ndarray, y_t: np.ndarray) -> None:
        """
        One RLS update with feature vector s_t (N,) and target y_t (d_out,).
        """
        if self.W is None or self.P is None or self.b is None:
            self.init(len(s_t), len(y_t))

        s = s_t.reshape(-1, 1)  # (N,1)
        # Gain
        denom = self.lam + float(s.T @ self.P @ s)
        K = (self.P @ s) / denom  # (N,1)
        # Error
        e = (y_t - (self.W.T @ s).ravel() - self.b)  # (d_out,)

        # Update
        self.W += (K @ e.reshape(1, -1))
        self.P = (self.P - K @ s.T @ self.P) / self.lam
        # Intercept via mean tracking (simple)
        self.b += 0.0 * e  # keep as 0 by default; adjust policy if needed

    def fit(self, S: np.ndarray, Y: np.ndarray) -> "RLSReadout":
        for t in range(S.shape[0]):
            self.partial_fit(S[t], Y[t])
        return self

    def predict(self, S: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None:
            raise RuntimeError("Call fit before predict")
        return np.asarray(S) @ self.W + self.b
