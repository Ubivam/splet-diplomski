"""Association-rule change recommendation over the commit history.

Two baselines from the literature, used by the evaluation to compare the
multilayer graph with rule mining on queries of one or more files:

* ROSE (Zimmermann et al., 2005) keeps only the transactions that contain the
  whole query ``Q`` and creates the rules ``Q -> x``.
* TARMAQ (Rolfsnes et al., 2016) keeps the transactions with the largest
  overlap ``k`` with the query and creates the rules ``Q' -> x`` with
  ``Q' = Q ∩ T``, ``|Q'| = k``. It therefore answers queries whose files never
  changed together, where ROSE has no rule at all.

Both rank the consequents by support and then by confidence. A transaction is
the set of files of one commit, with the same size limit as the evolutionary
layer (commits with more files carry no information about coupling).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .history import NormalizedCommit
from .layers.evolution import DEFAULT_MAX_COMMIT_FILES


@dataclass(frozen=True, order=True)
class RuleScore:
    """Support (number of transactions) and confidence of the best rule for a file."""

    support: int
    confidence: float

    def value(self) -> float:
        """One number with the same order: support first, confidence breaks ties."""
        return self.support + 0.5 * self.confidence


class TransactionIndex:
    """Commits as transactions of files, with an inverted index file -> commits."""

    def __init__(
        self,
        commits: Iterable[NormalizedCommit],
        files: set[str],
        max_commit_files: int = DEFAULT_MAX_COMMIT_FILES,
    ):
        self.transactions: list[frozenset[str]] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for commit in commits:
            touched = commit.files & files
            if not 2 <= len(touched) <= max_commit_files:
                continue
            for f in touched:
                self.postings[f].append(len(self.transactions))
            self.transactions.append(frozenset(touched))

    def containing(self, items: Iterable[str]) -> set[int]:
        """Transactions that contain every one of ``items``."""
        lists = sorted((self.postings.get(f, []) for f in items), key=len)
        if not lists:
            return set()
        result = set(lists[0])
        for other in lists[1:]:
            result.intersection_update(other)
        return result

    def is_seen(self, query: Iterable[str]) -> bool:
        """True when some past commit changed all files of the query together."""
        return bool(self.containing(query))

    def rose(self, query: Iterable[str]) -> dict[str, RuleScore]:
        q = frozenset(query)
        matching = self.containing(q)
        support = Counter(x for t in matching for x in self.transactions[t] - q)
        return {x: RuleScore(n, n / len(matching)) for x, n in support.items()}

    def tarmaq(self, query: Iterable[str]) -> dict[str, RuleScore]:
        q = frozenset(query)
        overlap = Counter(t for f in q for t in self.postings.get(f, []))
        if not overlap:
            return {}
        k = max(overlap.values())
        filtered = [t for t, n in overlap.items() if n == k]
        # Every transaction that contains an antecedent Q' (|Q'| = k) overlaps the
        # query in exactly Q', so frequencies can be counted on the filtered set.
        antecedents = Counter(self.transactions[t] & q for t in filtered)
        rules = Counter(
            (self.transactions[t] & q, x) for t in filtered for x in self.transactions[t] - q
        )
        best: dict[str, RuleScore] = {}
        for (antecedent, x), n in rules.items():
            score = RuleScore(n, n / antecedents[antecedent])
            if x not in best or score > best[x]:
                best[x] = score
        return best
