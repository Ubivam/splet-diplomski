"""ROSE and TARMAQ on the worked example of Rolfsnes et al. (SANER 2016, sec. IV)."""

from __future__ import annotations

import pytest

from splet.association import RuleScore, TransactionIndex
from splet.history import NormalizedCommit

FILES = {"a", "b", "c", "d", "e"}


@pytest.fixture
def index() -> TransactionIndex:
    history = [{"a", "b", "c"}, {"a", "d"}, {"c", "d"}]
    return TransactionIndex([NormalizedCommit("x", 0, frozenset(t)) for t in history], FILES)


def test_seen_and_unseen_queries(index):
    assert index.is_seen({"a"})
    assert index.is_seen({"a", "b"})
    assert not index.is_seen({"a", "c", "d"})  # all changed before, never together
    assert not index.is_seen({"e", "d"})  # e never changed


def test_rose_answers_only_seen_queries(index):
    assert index.rose({"a", "c", "d"}) == {}
    assert index.rose({"a", "b"}) == {"c": RuleScore(1, 1.0)}


def test_tarmaq_uses_the_largest_overlap(index):
    # q2: every transaction shares two files with the query; only {a, b, c}
    # has a file outside it, so the only rule is {a, c} -> b.
    assert index.tarmaq({"a", "c", "d"}) == {"b": RuleScore(1, 1.0)}
    # q3: {c, d} is filtered out; three rules {a} -> b, c, d.
    assert index.tarmaq({"a"}) == {
        "b": RuleScore(1, 0.5),
        "c": RuleScore(1, 0.5),
        "d": RuleScore(1, 0.5),
    }
    # q1: the new file e is ignored and d alone is the antecedent.
    assert set(index.tarmaq({"e", "d"})) == {"a", "c"}


def test_rule_order_is_support_then_confidence():
    assert RuleScore(2, 0.1).value() > RuleScore(1, 1.0).value()
    assert RuleScore(1, 0.9).value() > RuleScore(1, 0.2).value()


def test_large_commits_are_not_transactions():
    big = NormalizedCommit("x", 0, frozenset(f"f{i}" for i in range(40)))
    index = TransactionIndex([big], {f"f{i}" for i in range(40)})
    assert index.transactions == []
