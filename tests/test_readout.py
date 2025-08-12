import numpy as np
from psr.readout.linear import RidgeReadout

def test_ridge_fit_predict_linear_map():
    rng = np.random.default_rng(0)
    T, N, d = 200, 15, 4
    S = rng.normal(size=(T, N))
    W_true = rng.normal(size=(N, d))
    b_true = rng.normal(size=(d,))
    Y = S @ W_true + b_true + 0.01 * rng.normal(size=(T, d))
    rd = RidgeReadout(alpha=1e-3, fit_intercept=True).fit(S, Y)
    Yhat = rd.predict(S)
    mse = np.mean((Y - Yhat) ** 2)
    assert mse < 1e-2
