import numpy as np
from psr.metrics import rmse, nrmse, vpt

def test_basic_errors_and_vpt():
    y = np.array([[1.0, 0.0],[0.0, 1.0]])
    yhat = y.copy()
    assert rmse(y, yhat) == 0.0
    assert nrmse(y, yhat) == 0.0
    # Now add an error after first step
    yhat2 = y.copy()
    yhat2[1] = yhat2[1] + 2.0
    assert vpt(y, yhat2, tol=0.1) == 1
