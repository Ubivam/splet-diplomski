"""The time-split evaluation used in chapter 6 of the thesis."""

from __future__ import annotations

import numpy as np
import pytest

from splet.build import build_graph
from splet.evaluation import (
    Evaluator,
    Query,
    bootstrap_ci,
    make_queries,
    revision_at,
    simplex_grid,
    tune_weights,
)
from splet.history import NormalizedCommit

CORE, UTIL, INIT, TEST = (
    "src/pkg/core.py",
    "src/pkg/util.py",
    "src/pkg/__init__.py",
    "tests/test_core.py",
)


def commit(files: set[str]) -> NormalizedCommit:
    return NormalizedCommit("x", 0, frozenset(files))


def test_make_queries_one_per_file_within_limits():
    scope = {"a", "b", "c"}
    queries = make_queries(
        [commit({"a"}), commit({"a", "b", "z"}), commit({"a", "b", "c"})], scope, max_files=2
    )
    assert queries == [Query("a", frozenset({"b"})), Query("b", frozenset({"a"}))]


def test_simplex_grid_covers_the_simplex():
    grid = simplex_grid(0.1)
    assert len(grid) == 286
    assert all(abs(sum(w.values()) - 1) < 1e-9 for w in grid)
    assert len({tuple(sorted(w.items())) for w in grid}) == 286


def test_revision_at_picks_last_commit_before_moment():
    chain = [("a", 10), ("b", 20), ("c", 30)]
    assert revision_at(chain, 25) == "b"
    assert revision_at(chain, 5) == "a"


def test_bootstrap_interval_contains_the_mean():
    values = np.random.default_rng(1).random(200)
    lo, hi = bootstrap_ci(values, n=500)
    assert lo < values.mean() < hi


@pytest.fixture
def evaluator(repo):
    graph = build_graph(repo.root)
    scope = {f for f in graph.files() if f.endswith(".py")}
    queries = [Query(CORE, frozenset({UTIL})), Query(UTIL, frozenset({CORE, TEST}))]
    return Evaluator(graph, queries, scope)


def test_perfect_scores_give_reciprocal_rank_one(evaluator):
    m = evaluator.m
    scores = np.zeros_like(evaluator.seed_matrix)
    scores[m.index[UTIL], evaluator.seed_col[CORE]] = 1.0
    scores[m.index[CORE], evaluator.seed_col[UTIL]] = 1.0
    scores[m.index[TEST], evaluator.seed_col[UTIL]] = 0.5
    result = evaluator.evaluate(scores)
    assert result.rr.tolist() == [1.0, 1.0]
    assert result.recall[1].tolist() == [1.0, 0.5]
    assert result.summary()["R@5"] == 1.0


def test_history_beats_random_and_rose_is_scored(evaluator):
    history = evaluator.evaluate_weights({"evolution": 1.0}).summary()["MRR"]
    assert history >= evaluator.random().summary()["MRR"]
    assert 0 < evaluator.rose().summary()["MRR"] <= 1


def test_tune_weights_returns_the_best_grid_point(evaluator):
    grid = [{"evolution": 1.0}, {"docs": 1.0}]
    best, results = tune_weights(evaluator, grid)
    assert best == max(results, key=lambda r: r[1])[0]
