"""Random walk over the multilayer graph and personalized PageRank.

Each layer ``l`` has a symmetric weighted adjacency matrix ``A_l`` and a
row-stochastic transition matrix ``P_l``. With layer weights ``w_l`` the walker
at node ``i`` picks layer ``l`` with probability proportional to ``w_l``, but
only among the layers in which ``i`` has at least one edge:

    P[i, :] = sum_l w_l * [deg_l(i) > 0] * P_l[i, :] / sum_l w_l * [deg_l(i) > 0]

This keeps ``P`` stochastic when a node is isolated in some layer (a file
without imports, a file never changed together with another one). A node with
no edges in any layer teleports back to the seed distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .model import LAYERS, SpletGraph

DEFAULT_DAMPING = 0.85


@dataclass
class LayerMatrices:
    ids: list[str]
    index: dict[str, int]
    adjacency: dict[str, sp.csr_matrix]
    is_file: np.ndarray

    @classmethod
    def from_graph(cls, graph: SpletGraph) -> LayerMatrices:
        ids = sorted(graph.nodes)
        index = {n: i for i, n in enumerate(ids)}
        adjacency = {}
        for name in LAYERS:
            edges = graph.layers.get(name, {})
            rows = [index[e.a] for e in edges.values()] + [index[e.b] for e in edges.values()]
            cols = [index[e.b] for e in edges.values()] + [index[e.a] for e in edges.values()]
            vals = [e.weight for e in edges.values()] * 2
            adjacency[name] = sp.csr_matrix((vals, (rows, cols)), shape=(len(ids), len(ids)))
        is_file = np.array([graph.nodes[n].kind == "file" for n in ids])
        return cls(ids, index, adjacency, is_file)

    def transition(self, weights: dict[str, float]) -> sp.csr_matrix:
        n = len(self.ids)
        total = sp.csr_matrix((n, n))
        norm = np.zeros(n)
        for name, w in weights.items():
            if w <= 0:
                continue
            a = self.adjacency[name]
            deg = np.asarray(a.sum(axis=1)).ravel()
            has = deg > 0
            inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=has)
            total = total + w * sp.diags(inv) @ a
            norm += w * has
        inv_norm = np.divide(1.0, norm, out=np.zeros_like(norm), where=norm > 0)
        return (sp.diags(inv_norm) @ total).tocsr()


def personalized_pagerank(
    p: sp.csr_matrix,
    seeds: np.ndarray,
    damping: float = DEFAULT_DAMPING,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> np.ndarray:
    """Scores for a batch of seed distributions.

    ``seeds`` has shape (n, q): every column is a probability distribution to
    restart from. Returns an (n, q) matrix whose columns are the stationary
    distributions.
    """
    pt = p.T.tocsr()
    dangling = np.asarray(p.sum(axis=1)).ravel() < 1e-12
    r = seeds.copy()
    for _ in range(max_iter):
        leaked = r[dangling].sum(axis=0)
        nxt = damping * (pt @ r) + (1 - damping + damping * leaked) * seeds
        if np.abs(nxt - r).sum(axis=0).max() < tol:
            return np.asarray(nxt)
        r = nxt
    return np.asarray(r)


class Ranker:
    """Ranks project files by relevance to one or more seed files."""

    def __init__(
        self,
        graph: SpletGraph,
        weights: dict[str, float],
        damping: float = DEFAULT_DAMPING,
        matrices: LayerMatrices | None = None,
    ):
        self.m = matrices or LayerMatrices.from_graph(graph)
        self.weights = weights
        self.damping = damping
        self.p = self.m.transition(weights)

    def scores(self, seed_sets: list[list[str]]) -> np.ndarray:
        n = len(self.m.ids)
        seeds = np.zeros((n, len(seed_sets)))
        for j, group in enumerate(seed_sets):
            idx = [self.m.index[s] for s in group if s in self.m.index]
            if idx:
                seeds[idx, j] = 1.0 / len(idx)
        return personalized_pagerank(self.p, seeds, self.damping)

    def rank(self, seeds: list[str], k: int = 10) -> list[tuple[str, float]]:
        return self.rank_many([seeds], k)[0]

    def rank_many(self, seed_sets: list[list[str]], k: int = 10) -> list[list[tuple[str, float]]]:
        scores = self.scores(seed_sets)
        out = []
        for j, group in enumerate(seed_sets):
            col = scores[:, j].copy()
            col[~self.m.is_file] = -1.0
            for s in group:
                if s in self.m.index:
                    col[self.m.index[s]] = -1.0
            top = np.argsort(-col, kind="stable")[:k]
            out.append([(self.m.ids[i], float(col[i])) for i in top if col[i] > 0])
        return out
