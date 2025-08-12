from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ODEResult:
    t: np.ndarray          # (T,)
    x: np.ndarray          # (T, d)
    params: Dict[str, float]


def simulate_ode(
    f: Callable[[float, np.ndarray, Dict[str, float]], np.ndarray],
    x0: np.ndarray,
    T: int = 20_000,
    dt: float = 0.01,
    params: Dict[str, float] | None = None,
    method: str = "RK45",
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> ODEResult:
    """
    Integrate an ODE x' = f(t, x, params) on a fixed grid.

    Args:
        f: RHS function f(t, x, params)
        x0: initial state (d,)
        T: number of time steps
        dt: step size
        params: dict of parameters passed to f
    """
    params = {} if params is None else dict(params)
    t = np.arange(T, dtype=float) * dt
    t_span = (t[0], t[-1])

    def rhs(ti: float, xi: np.ndarray) -> np.ndarray:
        return f(ti, xi, params)

    sol = solve_ivp(rhs, t_span=t_span, y0=np.asarray(x0, dtype=float), t_eval=t, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    x = sol.y.T  # (T, d)
    return ODEResult(t=t, x=x, params=params)


# -----------------------------
# Canonical chaotic systems
# -----------------------------

def _lorenz63_rhs(_t: float, x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    sigma = p.get("sigma", 10.0)
    rho = p.get("rho", 28.0)
    beta = p.get("beta", 8.0 / 3.0)
    X, Y, Z = x
    return np.array([sigma * (Y - X), X * (rho - Z) - Y, X * Y - beta * Z], dtype=float)


def lorenz63(
    T: int = 20_000,
    dt: float = 0.01,
    x0: Tuple[float, float, float] | None = None,
    seed: int | None = 0,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> ODEResult:
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = tuple(rng.uniform(-1.0, 1.0, size=3))
    params = {"sigma": sigma, "rho": rho, "beta": beta}
    return simulate_ode(_lorenz63_rhs, np.array(x0, dtype=float), T=T, dt=dt, params=params)


def _rossler_rhs(_t: float, x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    a = p.get("a", 0.2)
    b = p.get("b", 0.2)
    c = p.get("c", 5.7)
    X, Y, Z = x
    return np.array([-Y - Z, X + a * Y, b + Z * (X - c)], dtype=float)


def rossler(
    T: int = 20_000,
    dt: float = 0.05,
    x0: Tuple[float, float, float] | None = None,
    seed: int | None = 0,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
) -> ODEResult:
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = tuple(rng.uniform(-1.0, 1.0, size=3))
    params = {"a": a, "b": b, "c": c}
    return simulate_ode(_rossler_rhs, np.array(x0, dtype=float), T=T, dt=dt, params=params)


def _chen_ueta_rhs(_t: float, x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    # Chen–Ueta (a.k.a. Chen) system
    a = p.get("a", 35.0)
    b = p.get("b", 3.0)
    c = p.get("c", 28.0)
    X, Y, Z = x
    return np.array([a * (Y - X), (c - a) * X - X * Z + c * Y, X * Y - b * Z], dtype=float)


def chen_ueta(
    T: int = 20_000,
    dt: float = 0.01,
    x0: Tuple[float, float, float] | None = None,
    seed: int | None = 0,
    a: float = 35.0,
    b: float = 3.0,
    c: float = 28.0,
) -> ODEResult:
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = tuple(rng.uniform(-1.0, 1.0, size=3))
    params = {"a": a, "b": b, "c": c}
    return simulate_ode(_chen_ueta_rhs, np.array(x0, dtype=float), T=T, dt=dt, params=params)
