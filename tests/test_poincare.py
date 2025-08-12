import numpy as np
from psr.geometry.poincare import PoincarePlane

def test_plane_hit_and_projection():
    pl = PoincarePlane(n=np.array([0.0, 0.0, 1.0]), d=0.0)  # z = 0
    a = np.array([1.0, 2.0, -1.0])
    b = np.array([1.0, 2.0,  1.0])
    assert pl.hits(a, b)
    t = pl.hit_fraction(a, b)
    xh = pl.hit_point(a, b)
    assert 0.0 <= t <= 1.0
    assert np.isclose(xh[2], 0.0)
    uv = pl.project(xh)
    assert uv.shape == (1, 2)
