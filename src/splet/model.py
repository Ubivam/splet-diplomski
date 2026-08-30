"""Multilayer graph: one set of nodes, several independent edge layers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HIERARCHY = "hierarchy"
STRUCTURE = "structure"
EVOLUTION = "evolution"
DOCS = "docs"
LAYERS = (HIERARCHY, STRUCTURE, EVOLUTION, DOCS)

FORMAT_VERSION = 1

# How repeated additions to the same edge combine their details: numbers are
# summed, except the fields listed here, which keep the largest value.
DETAIL_MAX_FIELDS = frozenset({"last_change"})


@dataclass
class Node:
    id: str
    kind: str  # "file" or "dir"
    lang: str = ""
    is_test: bool = False
    lines: int = 0
    changes: int = 0  # commits touching the file in the analysed history


@dataclass
class Edge:
    a: str
    b: str
    weight: float
    kinds: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


def edge_key(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


@dataclass
class SpletGraph:
    root: str
    revision: str = ""
    built_at: int = 0
    nodes: dict[str, Node] = field(default_factory=dict)
    layers: dict[str, dict[tuple[str, str], Edge]] = field(
        default_factory=lambda: {name: {} for name in LAYERS}
    )
    communities: dict[str, int] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        self.nodes.setdefault(node.id, node)

    def add_edge(self, layer: str, u: str, v: str, weight: float, kind: str, **detail: Any) -> Edge:
        """Add weight to the undirected edge ``u``–``v`` in ``layer``.

        Details are merged with those already on the edge: numbers are summed
        (or maximised for ``DETAIL_MAX_FIELDS``), lists are united keeping
        order, anything else is overwritten.
        """
        if u == v:
            raise ValueError(f"self-loop on {u!r} in layer {layer!r}")
        key = edge_key(u, v)
        edge = self.layers[layer].get(key)
        if edge is None:
            edge = self.layers[layer][key] = Edge(key[0], key[1], 0.0)
        edge.weight += weight
        if kind not in edge.kinds:
            edge.kinds.append(kind)
        for name, value in detail.items():
            old = edge.detail.get(name)
            if isinstance(value, (int, float)) and isinstance(old, (int, float)):
                edge.detail[name] = max(old, value) if name in DETAIL_MAX_FIELDS else old + value
            elif isinstance(value, list):
                edge.detail.setdefault(name, [])
                edge.detail[name] += [x for x in value if x not in edge.detail[name]]
            else:
                edge.detail[name] = value
        return edge

    def files(self) -> list[str]:
        return sorted(n.id for n in self.nodes.values() if n.kind == "file")

    def edge(self, layer: str, u: str, v: str) -> Edge | None:
        return self.layers[layer].get(edge_key(u, v))

    def neighbors(self, node: str, layer: str) -> dict[str, Edge]:
        out = {}
        for (a, b), e in self.layers[layer].items():
            if a == node:
                out[b] = e
            elif b == node:
                out[a] = e
        return out

    # --- serialization -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT_VERSION,
            "root": self.root,
            "revision": self.revision,
            "built_at": self.built_at,
            "params": self.params,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "layers": {
                name: [asdict(e) for e in edges.values()] for name, edges in self.layers.items()
            },
            "communities": self.communities,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpletGraph:
        if data.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported graph format {data.get('format')!r}")
        g = cls(
            root=data["root"],
            revision=data["revision"],
            built_at=data["built_at"],
            params=data.get("params", {}),
            communities=data.get("communities", {}),
        )
        for n in data["nodes"]:
            g.nodes[n["id"]] = Node(**n)
        for name, edges in data["layers"].items():
            g.layers[name] = {edge_key(e["a"], e["b"]): Edge(**e) for e in edges}
        return g

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SpletGraph:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
