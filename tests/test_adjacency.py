import numpy as np
import scipy.sparse as sp
from psr.graphs.adjacency import bins_from_points_rect, build_transition_adjacency, normalize_adjacency

def test_bins_and_transitions():
    # Points along a diagonal in (u,v)
    uv = np.stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)], axis=1)
    bin_ids, ranges = bins_from_points_rect(uv, bins=(8, 8))
    assert bin_ids.shape[0] == uv.shape[0]
    A = build_transition_adjacency(bin_ids, n_bins=64, directed=True, normalize="row", sparse=True)
    assert A.shape == (64, 64)
    assert sp.issparse(A)
    # Row-stochastic check on non-empty rows
    rowsum = np.array(A.sum(axis=1)).ravel()
    if rowsum.max() > 0:
        assert np.allclose(rowsum[rowsum > 0], 1.0)
