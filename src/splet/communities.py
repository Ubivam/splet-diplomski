"""Leiden community detection on the file-level projection of the graph."""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

import igraph as ig
import leidenalg

from .model import HIERARCHY, LAYERS, SpletGraph


def file_projection(graph: SpletGraph, weights: dict[str, float]) -> dict[tuple[str, str], float]:
    """Weighted undirected file–file edges combining all layers.

    Every layer is scaled to unit total weight before mixing, so that the layer
    weights mean the same thing as in the random walk. Directory containment is
    projected to file pairs: files in a directory with ``k`` files are linked
    with weight ``1/(k-1)``.
    """
    per_layer: dict[str, dict[tuple[str, str], float]] = {}
    files = set(graph.files())
    for name in LAYERS:
        pairs: dict[tuple[str, str], float] = {}
        for (a, b), e in graph.layers[name].items():
            if a in files and b in files:
                pairs[(a, b)] = e.weight
        if name == HIERARCHY:
            siblings: dict[str, list[str]] = defaultdict(list)
            for f in files:
                siblings[str(PurePosixPath(f).parent)].append(f)
            for group in siblings.values():
                group.sort()
                for i, a in enumerate(group):
                    for b in group[i + 1 :]:
                        pairs[(a, b)] = pairs.get((a, b), 0.0) + 1.0 / (len(group) - 1)
        per_layer[name] = pairs
    combined: dict[tuple[str, str], float] = defaultdict(float)
    for name, pairs in per_layer.items():
        total = sum(pairs.values())
        if total == 0 or weights.get(name, 0) <= 0:
            continue
        for key, w in pairs.items():
            combined[key] += weights[name] * w / total
    return dict(combined)


def detect_communities(
    graph: SpletGraph, weights: dict[str, float], resolution: float = 1.0, seed: int = 42
) -> dict[str, int]:
    """Community id per file; communities are numbered by size, largest first."""
    files = graph.files()
    index = {f: i for i, f in enumerate(files)}
    edges = file_projection(graph, weights)
    g = ig.Graph(n=len(files), edges=[(index[a], index[b]) for a, b in edges])
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=list(edges.values()) or None,
        resolution_parameter=resolution,
        seed=seed,
    )
    order = sorted(range(len(part)), key=lambda c: -len(part[c]))
    rank = {c: r for r, c in enumerate(order)}
    return {files[v]: rank[c] for c, members in enumerate(part) for v in members}
