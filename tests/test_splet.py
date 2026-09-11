from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from splet.analysis import classify_pair, hidden_dependencies, related_files
from splet.build import build_graph, build_graph_at
from splet.cli import main
from splet.fusion import LayerMatrices, Ranker, personalized_pagerank
from splet.history import files_at, history_after, history_until, resolve_rev
from splet.layers import hierarchy
from splet.layers.evolution import decay
from splet.model import DOCS, EVOLUTION, HIERARCHY, STRUCTURE, SpletGraph
from splet.obsidian import export_vault
from splet.pyindex import PythonModuleIndex
from splet.scan import SourceFile
from splet.server import create_server

CORE, UTIL, INIT = "src/pkg/core.py", "src/pkg/util.py", "src/pkg/__init__.py"
TEST, CFG = "tests/test_core.py", "config/settings.toml"


# --- history -------------------------------------------------------------------


def test_history_follows_renames_backward(repo):
    head = resolve_rev(repo.root)
    commits = history_until(repo.root, head, files_at(repo.root, head))
    together = [c for c in commits if {CORE, UTIL} <= c.files]
    # Commit 2 used the old paths pkg/core.py and pkg/util.py.
    assert len(together) == 4  # initial, core+util, move, core+util again


def test_history_forward_maps_to_reference_names(repo):
    revs = repo.git("rev-list", "--reverse", "HEAD").split()
    before_move = revs[1]
    alive = files_at(repo.root, before_move)
    later = history_after(repo.root, before_move, revs[-1], alive)
    touched = set().union(*(c.files for c in later))
    assert "pkg/core.py" in touched and "src/pkg/core.py" not in touched
    assert TEST not in touched  # added after the reference revision


# --- layers --------------------------------------------------------------------


def test_python_index_resolves_src_layout_and_relative_imports():
    idx = PythonModuleIndex([INIT, CORE, UTIL, TEST])
    assert idx.resolve("pkg.core") == CORE
    assert idx.resolve("pkg.core.run") == CORE
    assert idx.absolute(None, 1, CORE) == "pkg"
    assert idx.absolute("util", 1, CORE) == "pkg.util"
    assert idx.absolute("core", 1, INIT) == "pkg.core"


def test_tested_stem():
    assert hierarchy.tested_stem(SourceFile("tests/test_core.py", "python", True)) == "core"
    assert hierarchy.tested_stem(SourceFile("web/app.spec.ts", "typescript", True)) == "app"
    assert hierarchy.tested_stem(SourceFile("tests/conftest.py", "python", True)) is None


@pytest.fixture
def graph(repo) -> SpletGraph:
    return build_graph(repo.root)


def test_structure_layer(graph):
    assert graph.edge(STRUCTURE, CORE, UTIL) is not None
    assert graph.edge(STRUCTURE, INIT, CORE) is not None
    assert graph.edge(STRUCTURE, TEST, CORE) is not None
    assert graph.edge(STRUCTURE, CORE, CFG) is None


def test_test_pairing_ignores_private_underscore():
    g = SpletGraph(root=".")
    files = [
        SourceFile("pkg/_decoders.py", "python", False),
        SourceFile("tests/test_decoders.py", "python", True),
    ]
    hierarchy.build_hierarchy(g, files)
    assert "tests" in g.edge(HIERARCHY, "tests/test_decoders.py", "pkg/_decoders.py").kinds


def test_hierarchy_layer_pairs_test_with_module(graph):
    edge = graph.edge(HIERARCHY, TEST, CORE)
    assert edge is not None and "tests" in edge.kinds
    assert graph.edge(HIERARCHY, CORE, "dir:src/pkg") is not None


def test_docs_layer(graph):
    assert "link" in graph.edge(DOCS, "README.md", CORE).kinds
    assert graph.edge(DOCS, "docs/api.md", CORE) is not None  # `pkg.core`
    assert graph.edge(DOCS, "docs/api.md", UTIL) is not None  # `src/pkg/util.py`


def test_evolution_layer_counts_and_hidden_dependency(graph):
    assert graph.edge(EVOLUTION, CORE, UTIL).detail["count"] == 4
    assert graph.edge(EVOLUTION, CORE, CFG).detail["count"] == 2
    hidden = hidden_dependencies(graph, min_commits=2)
    assert any({h.a, h.b} == {CORE, CFG} and h.structural_distance is None for h in hidden)


def test_decay_half_life():
    assert decay(0, 365) == 1.0
    assert decay(365 * 86_400, 365) == pytest.approx(0.5)
    assert decay(10**9, None) == 1.0


def test_graph_at_past_revision_uses_old_names(repo):
    revs = repo.git("rev-list", "--reverse", "HEAD").split()
    old = build_graph_at(repo.root, revs[1])
    assert "pkg/core.py" in old.nodes and CORE not in old.nodes
    assert old.edge(EVOLUTION, "pkg/core.py", "pkg/util.py").detail["count"] == 2


def test_serialization_roundtrip(graph, tmp_path):
    graph.save(tmp_path / "g.json")
    loaded = SpletGraph.load(tmp_path / "g.json")
    assert loaded.to_dict() == json.loads(json.dumps(graph.to_dict()))


# --- ranking -------------------------------------------------------------------


def test_transition_matrix_is_stochastic(graph):
    m = LayerMatrices.from_graph(graph)
    p = m.transition(graph.params["weights"])
    sums = np.asarray(p.sum(axis=1)).ravel()
    assert np.all((np.abs(sums - 1) < 1e-9) | (sums == 0))


def test_pagerank_conserves_mass(graph):
    m = LayerMatrices.from_graph(graph)
    seeds = np.zeros((len(m.ids), 2))
    seeds[m.index[CORE], 0] = 1
    seeds[m.index[CFG], 1] = 1
    r = personalized_pagerank(m.transition(graph.params["weights"]), seeds)
    assert r.sum(axis=0) == pytest.approx([1.0, 1.0])


def test_history_only_ranking_finds_co_changed_config(graph):
    ranker = Ranker(graph, {EVOLUTION: 1.0})
    top = [p for p, _ in ranker.rank([CFG], k=2)]
    assert top[0] == CORE


def test_related_files_explains_reasons(graph):
    rows = related_files(graph, [CORE], k=5)
    util = next(r for r in rows if r.path == UTIL)
    assert any("->" in reason for reason in util.reasons)
    assert any("commits" in reason for reason in util.reasons)


# --- outputs -------------------------------------------------------------------


def test_obsidian_vault(graph, tmp_path):
    out = export_vault(graph, tmp_path / "vault", "sr")
    note = (out / f"{CORE}.md").read_text(encoding="utf-8")
    assert "[[src/pkg/util.py|util.py]]" in note and "splet/c" in note
    settings = json.loads((out / ".obsidian" / "graph.json").read_text())
    assert settings["colorGroups"][0]["query"].startswith("tag:#splet/c")
    export_vault(graph, out, "en")  # re-export over its own vault is allowed


def test_obsidian_refuses_foreign_directory(graph, tmp_path):
    (tmp_path / "notes").mkdir()
    with pytest.raises(FileExistsError):
        export_vault(graph, tmp_path / "notes")


def test_mcp_server_tools(repo):

    server = create_server(repo.root)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {
        "related_files",
        "explain_file",
        "hidden_dependencies",
        "communities",
        "connection_path",
        "project_report",
        "export_obsidian",
    } <= names


def test_cli_build_and_related(repo, capsys):
    assert main(["build", "--repo", str(repo.root)]) == 0
    assert main(["related", CORE, "--repo", str(repo.root), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out.split("\n", 2)[2])
    assert rows[0]["path"] in {UTIL, INIT, TEST}
    with pytest.raises(SystemExit, match="not project files"):
        main(["related", "nope.py", "--repo", str(repo.root)])


def test_hidden_dependency_categories(graph):

    assert classify_pair(graph, CORE, CFG) == "config_code"
    assert classify_pair(graph, TEST, CORE) == "test_code"
    assert classify_pair(graph, CORE, UTIL) == "code_code"
    hidden = hidden_dependencies(graph, min_commits=2)
    assert next(h for h in hidden if {h.a, h.b} == {CORE, CFG}).category == "config_code"
