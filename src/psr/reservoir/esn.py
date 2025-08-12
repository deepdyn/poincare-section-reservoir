from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .cells import apply_mask, leaky_update, get_activation


@dataclass
class ESN:
    """
    Basic leaky-integrator ESN.

    Shapes:
        W:    (N, N)
        Win:  (N, m)
        bias: (N,)
    """
    W: np.ndarray
    Win: np.ndarray
    bias: Optional[np.ndarray] = None
    leak: float = 0.3
    activation: str = "tanh"
    mask: Optional[np.ndarray] = None
    input_scale: float = 1.0

    def __post_init__(self):
        self.W = np.asarray(self.W, dtype=float)
        self.Win = np.asarray(self.Win, dtype=float)
        if self.bias is None:
            self.bias = np.zeros(self.W.shape[0], dtype=float)
        else:
            self.bias = np.asarray(self.bias, dtype=float)
        if self.W.shape[0] != self.W.shape[1]:
            raise ValueError("W must be square")
        if self.Win.shape[0] != self.W.shape[0]:
            raise ValueError("Win must have same number of rows as W")
        if self.bias.shape[0] != self.W.shape[0]:
            raise ValueError("bias must have length N")

        # cache activation fn
        self._act = get_activation(self.activation)
        # current state
        self._x = np.zeros(self.W.shape[0], dtype=float)

    @property
    def N(self) -> int:
        return self.W.shape[0]

    @property
    def m(self) -> int:
        return self.Win.shape[1]

    def reset(self, x0: Optional[np.ndarray] = None) -> None:
        self._x = np.zeros(self.N, dtype=float) if x0 is None else np.asarray(x0, dtype=float)

    def step(self, u_t: np.ndarray) -> np.ndarray:
        """
        Advance one step with input u_t (shape: (m,)).
        Returns the new state x_t (shape: (N,)).
        """
        u_t = np.asarray(u_t, dtype=float).reshape(-1)
        if u_t.shape[0] != self.m:
            raise ValueError(f"u_t must have shape ({self.m},), got {u_t.shape}")

        W_eff = apply_mask(self.W, self.mask)
        pre = W_eff @ self._x + self.Win @ (self.input_scale * u_t) + self.bias
        self._x = leaky_update(self._x, pre, leak=self.leak, activation=self._act)
        return self._x

    def forward(self, U: np.ndarray, *, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run the ESN over a sequence of inputs U with shape (T, m).
        Returns states with shape (T, N).
        """
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[1] != self.m:
            raise ValueError(f"U must have shape (T, {self.m})")
        self.reset(x0=x0)
        T = U.shape[0]
        X = np.zeros((T, self.N), dtype=float)
        for t in range(T):
            X[t] = self.step(U[t])
        return X
