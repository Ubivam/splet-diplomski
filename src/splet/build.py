"""Building the multilayer graph of a project directory or of a git revision."""

from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from .communities import detect_communities
from .history import (
    NormalizedCommit,
    commit_time,
    has_commits,
    history_until,
    is_git_repo,
    resolve_rev,
    run_git,
)
from .layers.docs import build_docs
from .layers.evolution import DEFAULT_HALF_LIFE_DAYS, DEFAULT_MAX_COMMIT_FILES, build_evolution
from .layers.hierarchy import build_hierarchy
from .layers.structure import build_structure
from .model import DOCS, EVOLUTION, HIERARCHY, STRUCTURE, Node, SpletGraph
from .pyindex import PythonModuleIndex
from .scan import SourceFile, scan

CACHE_DIR = ".splet"
GRAPH_FILE = "graph.json"

# Layer weights of the random walk. They were chosen on the experiment of the
# thesis (chapter 6) by leave-one-project-out selection over a 0.1 grid on the
# simplex: the vector with the best mean MRR over ten Python projects.
# Structure gets no weight in the walk because directory/test pairing and
# history already carry what imports tell about co-changing files; imports
# still explain results and define hidden dependencies.
DEFAULT_WEIGHTS = {HIERARCHY: 0.3, STRUCTURE: 0.0, EVOLUTION: 0.7, DOCS: 0.0}
# Best weights without history, for directories that are not git repositories.
STATIC_WEIGHTS = {HIERARCHY: 0.7, STRUCTURE: 0.3, EVOLUTION: 0.0, DOCS: 0.0}


def default_weights(graph: SpletGraph) -> dict[str, float]:
    """Layer weights for a graph; per-node normalisation in the walk already
    handles individual files without history, so only a graph with no history
    at all needs different weights."""
    return dict(DEFAULT_WEIGHTS if graph.layers[EVOLUTION] else STATIC_WEIGHTS)


def build_graph(
    root: Path,
    revision: str | None = None,
    *,
    history_root: Path | None = None,
    max_commit_files: int = DEFAULT_MAX_COMMIT_FILES,
    half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS,
    follow_renames: bool = True,
    with_communities: bool = True,
) -> SpletGraph:
    """Graph of the files under ``root``.

    ``history_root`` is the git repository whose history feeds the evolutionary
    layer (defaults to ``root``); ``revision`` is the commit whose ancestors are
    used and whose file names are the reference (defaults to ``HEAD``).
    """
    root = root.resolve()
    history_root = (history_root or root).resolve()
    files = scan(root)
    graph = SpletGraph(root=str(root), built_at=int(time.time()))
    graph.params["fingerprint"] = fingerprint(root, files)
    for f in files:
        lines = (root / f.path).read_bytes().count(b"\n")
        graph.add_node(Node(f.path, "file", f.lang, f.is_test, lines))

    index = PythonModuleIndex([f.path for f in files if f.lang == "python"])
    build_hierarchy(graph, files)
    build_structure(graph, root, files, index)
    build_docs(graph, root, files, index)

    if is_git_repo(history_root) and has_commits(history_root):
        rev = resolve_rev(history_root, revision or "HEAD")
        graph.revision = rev
        # Git reports paths relative to the repository top level; the graph
        # uses paths relative to ``root``, which may be a subdirectory.
        prefix = run_git(root, "rev-parse", "--show-prefix").strip() if history_root == root else ""
        alive = {prefix + f.path for f in files}
        commits = [
            NormalizedCommit(c.sha, c.timestamp, frozenset(p[len(prefix) :] for p in c.files))
            for c in history_until(history_root, rev, alive, follow_renames)
        ]
        build_evolution(
            graph, commits, commit_time(history_root, rev), max_commit_files, half_life_days
        )

    weights = default_weights(graph)
    graph.params["weights"] = weights
    if with_communities and graph.files():
        graph.communities = detect_communities(graph, weights)
    return graph


def export_revision(repo: Path, revision: str, target: Path) -> None:
    """Write the tree of ``revision`` into ``target`` (no working-tree changes)."""
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", revision],
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")


def build_graph_at(repo: Path, revision: str, **kwargs: Any) -> SpletGraph:
    """Graph of a past revision, as if it were built on that day."""
    with tempfile.TemporaryDirectory(prefix="splet-") as tmp:
        export_revision(repo, revision, Path(tmp))
        graph = build_graph(Path(tmp), revision, history_root=repo, **kwargs)
    graph.root = str(repo.resolve())
    return graph


def fingerprint(root: Path, files: list[SourceFile]) -> str:
    """Digest of the scanned files' paths, sizes and modification times.

    Together with the revision it tells whether a cached graph still describes
    the working tree, including edits that are not committed yet.
    """
    digest = hashlib.sha1(usedforsecurity=False)
    for f in files:
        st = (root / f.path).stat()
        digest.update(f"{f.path}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def cache_path(root: Path) -> Path:
    return root / CACHE_DIR / GRAPH_FILE


def current_revision(root: Path) -> str:
    return resolve_rev(root) if is_git_repo(root) and has_commits(root) else ""


def is_current(graph: SpletGraph, root: Path) -> bool:
    """Whether ``graph`` was built from the present state of ``root``."""
    return graph.revision == current_revision(root) and graph.params.get(
        "fingerprint"
    ) == fingerprint(root.resolve(), scan(root))


def load_or_build(root: Path, rebuild: bool = False) -> SpletGraph:
    """Cached graph if it still matches the project, otherwise a fresh build."""
    path = cache_path(root)
    if path.exists() and not rebuild:
        try:
            graph = SpletGraph.load(path)
        except (ValueError, KeyError, TypeError):
            graph = None  # unreadable or older format: rebuild below
        if graph is not None and is_current(graph, root):
            return graph
    graph = build_graph(root)
    graph.save(path)
    return graph
