"""Questions answered from a built graph: related files, hidden dependencies,
central files, explanations and paths."""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import networkx as nx

from .fusion import Ranker
from .model import DOCS, EVOLUTION, HIERARCHY, LAYERS, STRUCTURE, Edge, SpletGraph
from .scan import DOC_LANGUAGES

HUB_SHARE = 0.10
HUB_MIN_COMMITS = 50  # below this, shares of commits say nothing about hubs
CODE_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "c",
    "cpp",
    "java",
    "go",
    "rust",
    "ruby",
    "shell",
    "sql",
}


class Adjacency:
    """Per-layer neighbour lists, built once per graph."""

    def __init__(self, graph: SpletGraph):
        self.graph = graph
        self.by_layer: dict[str, dict[str, dict[str, Edge]]] = {}
        for name in LAYERS:
            adj: dict[str, dict[str, Edge]] = defaultdict(dict)
            for (a, b), e in graph.layers[name].items():
                adj[a][b] = e
                adj[b][a] = e
            self.by_layer[name] = adj

    def neighbors(self, node: str, layer: str) -> dict[str, Edge]:
        return self.by_layer[layer].get(node, {})


def describe_link(graph: SpletGraph, adj: Adjacency, a: str, b: str) -> list[str]:
    """Human-readable reasons why two files are directly connected."""
    reasons = []
    if e := adj.neighbors(a, STRUCTURE).get(b):
        reasons.extend(e.detail.get("imports", []))
    if e := adj.neighbors(a, EVOLUTION).get(b):
        reasons.append(f"changed together in {e.detail.get('count', 0)} commits")
    if (e := adj.neighbors(a, HIERARCHY).get(b)) and "tests" in e.kinds:
        reasons.append("test named after the module")
    if e := adj.neighbors(a, DOCS).get(b):
        reasons.append(f"documentation reference ({', '.join(e.kinds)})")
    if (
        not reasons
        and graph.nodes[a].kind == graph.nodes[b].kind == "file"
        and a.rpartition("/")[0] == b.rpartition("/")[0]
    ):
        reasons.append("same directory")
    return reasons


@dataclass
class Related:
    path: str
    score: float
    reasons: list[str] = field(default_factory=list)


def related_files(
    graph: SpletGraph,
    seeds: list[str],
    *,
    k: int = 10,
    weights: dict[str, float] | None = None,
    code_only: bool = False,
    ranker: Ranker | None = None,
) -> list[Related]:
    ranker = ranker or Ranker(graph, weights or graph.params["weights"])
    adj = Adjacency(graph)
    ranked = ranker.rank(seeds, k=len(graph.nodes))
    out = []
    for path, score in ranked:
        if code_only and graph.nodes[path].lang not in CODE_LANGUAGES:
            continue
        reasons = [r for s in seeds for r in describe_link(graph, adj, s, path)]
        out.append(Related(path, round(score, 4), reasons or ["indirect (through neighbours)"]))
        if len(out) == k:
            break
    return out


CI_NAMES = {
    ".pre-commit-config.yaml",
    "tox.ini",
    ".readthedocs.yaml",
    ".readthedocs.yml",
    "noxfile.py",
    ".coveragerc",
    "codecov.yml",
    ".gitlab-ci.yml",
}
CONFIG_LANGUAGES = {"config", "build"}
# Categories of hidden dependencies, most useful to a developer first.
CATEGORIES = (
    "cross_language",
    "code_code",
    "config_code",
    "test_code",
    "test_test",
    "config_config",
    "ci",
    "other",
)


def is_ci(path: str) -> bool:
    p = PurePosixPath(path)
    return ".github" in p.parts or p.name in CI_NAMES


def classify_pair(graph: SpletGraph, a: str, b: str) -> str:  # noqa: PLR0911 (flat decision list)
    """Kind of a file pair, from the languages and roles of the two files."""
    na, nb = graph.nodes[a], graph.nodes[b]
    code = CODE_LANGUAGES | {"template", "style"}
    if is_ci(a) and is_ci(b):
        return "ci"
    if na.lang in CONFIG_LANGUAGES and nb.lang in CONFIG_LANGUAGES:
        return "config_config"
    if (na.lang in CONFIG_LANGUAGES) != (nb.lang in CONFIG_LANGUAGES):
        return "config_code" if (na.lang in code or nb.lang in code) else "other"
    if na.lang in code and nb.lang in code:
        if na.is_test and nb.is_test:
            return "test_test"
        if na.is_test or nb.is_test:
            return "test_code"
        return "code_code" if na.lang == nb.lang else "cross_language"
    return "ci" if is_ci(a) or is_ci(b) else "other"


@dataclass
class HiddenDependency:
    a: str
    b: str
    commits: int
    weight: float
    confidence: float  # max over directions of P(b changes | a changes)
    structural_distance: int | None
    category: str = "other"


def hub_files(graph: SpletGraph, share: float = HUB_SHARE) -> set[str]:
    """Files touched by more than ``share`` of all commits (changelogs, version files)."""
    commits = graph.params.get("commits", 0)
    if commits < HUB_MIN_COMMITS:
        return set()
    return {f for f in graph.files() if graph.nodes[f].changes > share * commits}


def hidden_dependencies(
    graph: SpletGraph, min_commits: int = 3, limit: int = 20, min_distance: int = 3
) -> list[HiddenDependency]:
    """Pairs that change together but are far apart (or unconnected) in the code.

    ``structural_distance`` is the number of import hops between the files;
    ``None`` means no import path exists at all. Prose documents are left out
    (the documentation layer covers them), and so are hub files that change
    in a large share of all commits and would pair with everything.
    """
    excluded = hub_files(graph) | {f for f in graph.files() if graph.nodes[f].lang in DOC_LANGUAGES}
    s = nx.Graph()
    s.add_nodes_from(graph.files())
    s.add_edges_from(graph.layers[STRUCTURE])
    tests = {k for k, e in graph.layers[HIERARCHY].items() if "tests" in e.kinds}
    near: dict[str, set[str]] = {}  # files within min_distance-1 import hops

    def is_near(a: str, b: str) -> bool:
        if a not in near:
            near[a] = set(nx.single_source_shortest_path_length(s, a, cutoff=min_distance - 1))
        return b in near[a]

    found = []
    for (a, b), e in graph.layers[EVOLUTION].items():
        count = e.detail.get("count", 0)
        if count < min_commits or (a, b) in tests or a in excluded or b in excluded:
            continue
        if is_near(a, b):
            continue
        conf = max(count / max(graph.nodes[a].changes, 1), count / max(graph.nodes[b].changes, 1))
        found.append(
            HiddenDependency(a, b, count, e.weight, conf, None, classify_pair(graph, a, b))
        )
    # Pairs involving code first; CI and configuration-only pairs are real but
    # rarely what a developer is looking for.
    rank = {c: i for i, c in enumerate(CATEGORIES)}
    found.sort(
        key=lambda h: (
            rank[h.category] >= rank["config_config"],
            -h.commits * h.confidence,
            h.a,
            h.b,
        )
    )
    top = found[:limit]
    # Exact distances only for the reported pairs; ``None`` means no import path.
    for h in top:
        try:
            h.structural_distance = nx.shortest_path_length(s, h.a, h.b)
        except nx.NetworkXNoPath:
            h.structural_distance = None
    return top


def central_files(graph: SpletGraph, limit: int = 10) -> list[tuple[str, int, int]]:
    """Files most imported by others: (path, importers, commits)."""
    adj = Adjacency(graph)
    rows = [(f, len(adj.neighbors(f, STRUCTURE)), graph.nodes[f].changes) for f in graph.files()]
    rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    return rows[:limit]


def explain_file(graph: SpletGraph, path: str, per_layer: int = 8) -> dict[str, Any]:
    if path not in graph.nodes:
        raise KeyError(path)
    adj = Adjacency(graph)
    node = graph.nodes[path]
    layers = {}
    for name in LAYERS:
        nbrs = sorted(adj.neighbors(path, name).items(), key=lambda kv: -kv[1].weight)
        layers[name] = [
            {
                "path": p,
                "weight": round(e.weight, 4),
                "kinds": e.kinds,
                **({"commits": e.detail["count"]} if "count" in e.detail else {}),
            }
            for p, e in nbrs[:per_layer]
            if not p.startswith("dir:")
        ]
    return {
        "path": path,
        "language": node.lang,
        "is_test": node.is_test,
        "lines": node.lines,
        "commits": node.changes,
        "community": graph.communities.get(path),
        "layers": layers,
    }


def connection_path(graph: SpletGraph, a: str, b: str) -> list[dict[str, Any]]:
    """Strongest chain of links between two files, with the layer of each hop."""
    g = nx.Graph()
    for name in LAYERS:
        total = sum(e.weight for e in graph.layers[name].values()) or 1.0
        for (u, v), e in graph.layers[name].items():
            cost = total / (e.weight * len(graph.layers[name]))
            if not g.has_edge(u, v) or g[u][v]["cost"] > cost:
                g.add_edge(u, v, cost=cost, layer=name)
    if a not in g or b not in g:
        raise ValueError(f"{a if a not in g else b!r} has no links in any layer")
    try:
        nodes = nx.shortest_path(g, a, b, weight="cost")
    except nx.NetworkXNoPath:
        raise ValueError(f"no chain of links connects {a!r} and {b!r}") from None
    return [{"from": u, "to": v, "layer": g[u][v]["layer"]} for u, v in itertools.pairwise(nodes)]


def community_summary(graph: SpletGraph, limit_members: int = 6) -> list[dict[str, Any]]:
    adj = Adjacency(graph)
    groups: dict[int, list[str]] = defaultdict(list)
    for f, c in graph.communities.items():
        groups[c].append(f)
    out = []
    for c in sorted(groups):
        members = sorted(
            groups[c], key=lambda f: -(len(adj.neighbors(f, STRUCTURE)) + graph.nodes[f].changes)
        )
        dirs: dict[str, int] = defaultdict(int)
        for f in groups[c]:
            dirs[f.rpartition("/")[0] or "."] += 1
        main_dir = max(dirs.items(), key=lambda kv: kv[1])[0]
        out.append(
            {
                "id": c,
                "size": len(groups[c]),
                "main_directory": main_dir,
                "top_members": members[:limit_members],
            }
        )
    return out
