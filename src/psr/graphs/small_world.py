from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp


def watts_strogatz_adjacency(
    n: int,
    k: int,
    p: float,
    *,
    directed: bool = False,
    weight_scale: float = 1.0,
    seed: Optional[int] = 0,
    sparse: bool = True,
) -> sp.csr_matrix | np.ndarray:
    """
    Watts–Strogatz small-world graph.
    k: each node is joined with its k nearest neighbors in ring topology (use even k)
    p: rewiring probability
    """
    rng = np.random.default_rng(seed)
    if directed:
        G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
        # Orient edges randomly
        edges = []
        for u, v in G.edges():
            if rng.random() < 0.5:
                edges.append((u, v))
            else:
                edges.append((v, u))
        row = [u for u, _ in edges]
        col = [v for _, v in edges]
    else:
        G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
        row = [u for u, v in G.edges()] + [v for u, v in G.edges()]
        col = [v for u, v in G.edges()] + [u for u, v in G.edges()]

    data = np.ones(len(row), dtype=float) * float(weight_scale)
    if sparse:
        A = sp.csr_matrix((data, (row, col)), shape=(n, n))
    else:
        A = np.zeros((n, n), dtype=float)
        A[row, col] = data
    # remove self-loops if any
    if sparse:
        A.setdiag(0.0)
        A.eliminate_zeros()
    else:
        np.fill_diagonal(A, 0.0)
    return A
