"""Hierarchy layer: the directory tree and the test-naming convention.

Every directory becomes a node (``dir:<path>``) linked to its parent, and
every file is linked to its directory. A test file is additionally linked to
the module it is named after (``test_app.py`` → ``app.py``), which is how most
Python and JavaScript projects pair tests with code. Leading underscores of
private modules are ignored (``test_decoders.py`` → ``_decoders.py``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import PurePosixPath

from ..model import HIERARCHY, Node, SpletGraph
from ..scan import SourceFile

ROOT_DIR = "dir:."
TEST_AFFIX = re.compile(r"^(?:test_)?(.+?)(?:_test|\.test|\.spec)?$")


def dir_id(path: PurePosixPath) -> str:
    return ROOT_DIR if str(path) in ("", ".") else f"dir:{path}"


def tested_stem(file: SourceFile) -> str | None:
    stem = PurePosixPath(file.path).stem
    match = TEST_AFFIX.match(stem)
    name = match.group(1) if match else stem
    return name if name != stem or stem.startswith("test") else None


def _shared_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(PurePosixPath(a).parts, PurePosixPath(b).parts, strict=False):
        if x != y:
            break
        n += 1
    return n


def build_hierarchy(graph: SpletGraph, files: list[SourceFile]) -> None:
    graph.add_node(Node(ROOT_DIR, "dir"))
    for f in files:
        child, parent = f.path, PurePosixPath(f.path).parent
        while True:
            pid = dir_id(parent)
            graph.add_node(Node(pid, "dir"))
            graph.add_edge(HIERARCHY, child, pid, 1.0, "contains")
            if pid == ROOT_DIR or graph.edge(HIERARCHY, pid, dir_id(parent.parent)):
                break
            child, parent = pid, parent.parent

    by_stem: dict[str, list[SourceFile]] = defaultdict(list)
    for f in files:
        if not f.is_test:
            by_stem[PurePosixPath(f.path).stem.lstrip("_")].append(f)
    for f in files:
        if not f.is_test:
            continue
        stem = tested_stem(f)
        candidates = [c for c in by_stem.get((stem or "").lstrip("_"), []) if c.lang == f.lang]
        if not candidates:
            continue
        # Several modules with the same name: pair with the closest one.
        best = max(candidates, key=lambda c: _shared_prefix(c.path, f.path))
        graph.add_edge(HIERARCHY, f.path, best.path, 1.0, "tests")
