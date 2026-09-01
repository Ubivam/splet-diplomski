"""Evolutionary layer: files that change together in the version history.

The weight of an edge sums the contributions of the commits that touched both
files. A commit that touches ``n`` files contributes ``1/(n-1)`` to each of its
pairs, so that large commits do not dominate, and commits older than the
reference time are discounted exponentially with the given half-life.
Commits with more than ``max_commit_files`` files (mass reformatting, licence
headers, vendoring) carry no information about coupling and are skipped.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from ..history import NormalizedCommit
from ..model import EVOLUTION, SpletGraph

SECONDS_PER_DAY = 86_400
DEFAULT_MAX_COMMIT_FILES = 30
DEFAULT_HALF_LIFE_DAYS = 365.0


def decay(age_seconds: float, half_life_days: float | None) -> float:
    if not half_life_days:
        return 1.0
    return float(0.5 ** (max(age_seconds, 0.0) / (half_life_days * SECONDS_PER_DAY)))


def build_evolution(
    graph: SpletGraph,
    commits: list[NormalizedCommit],
    reference_time: int,
    max_commit_files: int = DEFAULT_MAX_COMMIT_FILES,
    half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS,
) -> None:
    files = {n for n, node in graph.nodes.items() if node.kind == "file"}
    changes: Counter[str] = Counter()
    for commit in commits:
        touched = sorted(commit.files & files)
        changes.update(touched)
        if not 2 <= len(touched) <= max_commit_files:
            continue
        w = decay(reference_time - commit.timestamp, half_life_days) / (len(touched) - 1)
        for a, b in combinations(touched, 2):
            graph.add_edge(EVOLUTION, a, b, w, "co-change", count=1, last_change=commit.timestamp)
    for name, n in changes.items():
        graph.nodes[name].changes = n
    graph.params.update(
        commits=len(commits), max_commit_files=max_commit_files, half_life_days=half_life_days
    )
