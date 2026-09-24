"""The time-split evaluation used in chapter 6 of the thesis."""

from __future__ import annotations

import numpy as np
import pytest

from splet.association import TransactionIndex
from splet.build import build_graph
from splet.evaluation import (
    Evaluator,
    Query,
    SetEvaluator,
    SetQuery,
    bootstrap_ci,
    make_queries,
    make_set_queries,
    revision_at,
    simplex_grid,
    tune_weights,
)
from splet.fusion import LayerMatrices, Ranker, personalized_pagerank
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


def test_set_queries_split_each_commit_once_per_size():
    c = NormalizedCommit("ab12cd34ef56", 0, frozenset({"a", "b", "c", "d"}))
    queries = make_set_queries([c], {"a", "b", "c", "d"}, sizes=(1, 2, 3, 4))
    assert [len(q.seeds) for q in queries] == [1, 2, 3]  # the answer is never empty
    assert all(q.seeds | q.truth == c.files and not q.seeds & q.truth for q in queries)
    assert make_set_queries([c], set(c.files), sizes=(1, 2, 3)) == queries[:3]


def test_set_evaluator_ranks_like_the_tool(repo):
    graph = build_graph(repo.root)
    scope = {f for f in graph.files() if f.endswith(".py")}
    weights = {"evolution": 0.7, "hierarchy": 0.3}
    ev = SetEvaluator(graph, [SetQuery(frozenset({CORE, UTIL}), frozenset({TEST}))], scope)
    tool = Ranker(graph, weights).scores([[CORE, UTIL]])[:, 0]
    ours = personalized_pagerank(ev.m.transition(weights), ev.restart, 0.85)[:, 0]
    assert np.allclose(tool, ours)


def test_pagerank_is_linear_in_the_restart_without_dangling_nodes(repo):
    """With a positive hierarchy weight every node has an edge (proposition in
    chapter 4), so the ranking for a set of files is the mean of single-file ones."""
    graph = build_graph(repo.root)
    m = LayerMatrices.from_graph(graph)
    p = m.transition({"evolution": 0.7, "hierarchy": 0.3})
    assert np.allclose(np.asarray(p.sum(axis=1)).ravel(), 1.0)
    single = np.zeros((len(m.ids), 2))
    single[m.index[CORE], 0] = single[m.index[UTIL], 1] = 1.0
    joint = personalized_pagerank(p, single.mean(axis=1, keepdims=True), 0.85)[:, 0]
    mean = personalized_pagerank(p, single, 0.85).mean(axis=1)
    assert np.allclose(joint, mean, atol=1e-7)


def test_set_evaluator_scores_graph_and_rules(repo):
    graph = build_graph(repo.root)
    scope = {f for f in graph.files() if f.endswith(".py")}
    ev = SetEvaluator(graph, [SetQuery(frozenset({CORE, UTIL}), frozenset({TEST}))], scope)
    history = [NormalizedCommit("x", 0, frozenset({CORE, UTIL, TEST}))]
    index = TransactionIndex(history, set(graph.files()))
    assert ev.rules(index, "rose").rr.tolist() == [1.0]
    assert ev.rules(index, "tarmaq").rr.tolist() == [1.0]
    assert 0 < ev.pagerank({"evolution": 1.0}).rr[0] <= 1
