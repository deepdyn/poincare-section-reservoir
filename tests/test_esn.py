import numpy as np
from psr.reservoir.esn import ESN

def test_esn_forward_shapes_and_identity_activation():
    N, m, T = 10, 3, 20
    W = np.zeros((N, N))
    Win = np.ones((N, m))
    esn = ESN(W=W, Win=Win, leak=1.0, activation="identity")
    U = np.random.randn(T, m)
    X = esn.forward(U)
    assert X.shape == (T, N)
    # With W=0, bias=0, leak=1, identity, state equals phi(Win @ u) each step
    np.testing.assert_allclose(X[0], Win @ U[0], rtol=1e-6, atol=1e-6)
