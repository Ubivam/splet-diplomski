"""Time-split evaluation: can a graph built at revision T predict which files
change together in the commits that follow T?

For every later commit ``c`` and every file ``s`` in ``c`` (the seed), the
other files of ``c`` are the relevant answers. A method ranks all candidate
files for the seed; the ranking is scored with reciprocal rank and recall@k.
Only commits made after ``T`` are used for testing, and the graph sees only
the tree and the history of ``T``, so no information from the future leaks
into the ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .fusion import LayerMatrices, personalized_pagerank
from .history import NormalizedCommit, run_git
from .model import EVOLUTION, LAYERS, SpletGraph

KS = (1, 5, 10)
MAX_QUERY_COMMIT_FILES = 30


@dataclass(frozen=True)
class Query:
    seed: str
    truth: frozenset[str]


def make_queries(
    commits: list[NormalizedCommit], scope: set[str], max_files: int = MAX_QUERY_COMMIT_FILES
) -> list[Query]:
    queries: list[Query] = []
    for c in commits:
        files = c.files & scope
        if 2 <= len(files) <= max_files:
            queries.extend(Query(s, frozenset(files - {s})) for s in sorted(files))
    return queries


def first_parent_commits(repo: Path) -> list[tuple[str, int]]:
    """(sha, commit time) along the first-parent chain of HEAD, oldest first."""
    out = run_git(repo, "log", "--first-parent", "--reverse", "--format=%H %ct", "HEAD")
    return [(sha, int(ts)) for sha, ts in (line.split() for line in out.splitlines())]


def time_quantiles(repo: Path, fractions: list[float]) -> list[int]:
    """Commit times below which the given fractions of all non-merge commits lie."""
    times = sorted(
        int(t) for t in run_git(repo, "log", "--no-merges", "--format=%ct", "HEAD").split()
    )
    return [times[min(int(f * len(times)), len(times) - 1)] for f in fractions]


def revision_at(chain: list[tuple[str, int]], moment: int) -> str:
    """Last first-parent commit made at or before ``moment``."""
    chosen = chain[0][0]
    for sha, ts in chain:
        if ts > moment:
            break
        chosen = sha
    return chosen


@dataclass
class Scores:
    """Per-query metrics of one method."""

    rr: np.ndarray  # reciprocal rank of the first relevant file
    recall: dict[int, np.ndarray]

    def summary(self) -> dict[str, float]:
        out = {"MRR": float(self.rr.mean())}
        out.update({f"R@{k}": float(v.mean()) for k, v in self.recall.items()})
        return out


class Evaluator:
    """Scores rankings of a fixed set of queries against one graph."""

    def __init__(self, graph: SpletGraph, queries: list[Query], scope: set[str], seed: int = 0):
        self.graph = graph
        self.m = LayerMatrices.from_graph(graph)
        self.queries = queries
        self.candidates = np.array([n in scope for n in self.m.ids])
        seeds = sorted({q.seed for q in queries})
        self.seed_col = {s: j for j, s in enumerate(seeds)}
        self.seed_matrix = np.zeros((len(self.m.ids), len(seeds)))
        for s, j in self.seed_col.items():
            self.seed_matrix[self.m.index[s], j] = 1.0
        # Fixed random tie-breaking: files with equal scores (typically zero,
        # i.e. unreachable) are ordered randomly but identically for all methods.
        self.tiebreak = np.random.default_rng(seed).random(len(self.m.ids)) * 1e-12

    # --- scoring -----------------------------------------------------------------
    def evaluate(self, score_matrix: np.ndarray) -> Scores:
        """``score_matrix`` has one column per distinct seed (see ``seed_col``)."""
        rr = np.zeros(len(self.queries))
        recall = {k: np.zeros(len(self.queries)) for k in KS}
        positions = {}
        for s, j in self.seed_col.items():
            col = np.where(self.candidates, score_matrix[:, j] + self.tiebreak, -np.inf)
            col[self.m.index[s]] = -np.inf
            pos = np.empty(len(col), dtype=np.int64)
            pos[np.argsort(-col, kind="stable")] = np.arange(1, len(col) + 1)
            positions[s] = pos
        for i, q in enumerate(self.queries):
            ranks = positions[q.seed][[self.m.index[t] for t in q.truth]]
            rr[i] = 1.0 / ranks.min()
            for k in KS:
                recall[k][i] = (ranks <= k).sum() / len(ranks)
        return Scores(rr, recall)

    def pagerank(self, weights: dict[str, float], damping: float = 0.85) -> np.ndarray:
        return personalized_pagerank(self.m.transition(weights), self.seed_matrix, damping)

    def evaluate_weights(self, weights: dict[str, float]) -> Scores:
        return self.evaluate(self.pagerank(weights))

    # --- baselines ---------------------------------------------------------------
    def random(self) -> Scores:
        return self.evaluate(np.zeros_like(self.seed_matrix))

    def rose(self) -> Scores:
        """Association-rule baseline after Zimmermann et al. (2005): rank by the
        confidence count(s, x) / changes(s), i.e. by the number of past commits
        shared with the seed. Only direct co-change neighbours score above zero."""
        counts = sp.lil_matrix((len(self.m.ids), len(self.m.ids)))
        for (a, b), e in self.graph.layers[EVOLUTION].items():
            n = e.detail.get("count", 0)
            counts[self.m.index[a], self.m.index[b]] = n
            counts[self.m.index[b], self.m.index[a]] = n
        return self.evaluate(counts.tocsr() @ self.seed_matrix)


def simplex_grid(step: float = 0.1, layers: tuple[str, ...] = LAYERS) -> list[dict[str, float]]:
    """All weight vectors on the simplex with the given step (286 for 4 layers, 0.1)."""
    n = round(1 / step)
    grid = []
    for combo in product(range(n + 1), repeat=len(layers) - 1):
        if sum(combo) <= n:
            parts = [*combo, n - sum(combo)]
            grid.append({name: p / n for name, p in zip(layers, parts, strict=True)})
    return grid


def tune_weights(
    evaluator: Evaluator, grid: list[dict[str, float]], metric: str = "MRR"
) -> tuple[dict[str, float], list[tuple[dict[str, float], float]]]:
    """Best weights on a validation evaluator, and the score of every grid point."""
    results = [(w, evaluator.evaluate_weights(w).summary()[metric]) for w in grid]
    best = max(results, key=lambda wr: wr[1])[0]
    return best, results


def bootstrap_ci(
    values: np.ndarray, n: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)])
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
