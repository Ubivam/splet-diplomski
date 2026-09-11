"""Graph construction edge cases, caching, ranking properties and analyses."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from splet.analysis import (
    central_files,
    community_summary,
    connection_path,
    explain_file,
    hidden_dependencies,
    hub_files,
)
from splet.build import (
    DEFAULT_WEIGHTS,
    STATIC_WEIGHTS,
    build_graph,
    cache_path,
    is_current,
    load_or_build,
)
from splet.fusion import Ranker
from splet.model import EVOLUTION, STRUCTURE, SpletGraph
from splet.report import layer_overlap, render_report

CORE, UTIL, CFG = "src/pkg/core.py", "src/pkg/util.py", "config/settings.toml"


def test_directory_without_git_uses_static_weights(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("")
    graph = build_graph(tmp_path)
    assert graph.revision == ""
    assert not graph.layers[EVOLUTION]
    assert graph.params["weights"] == STATIC_WEIGHTS
    assert graph.edge(STRUCTURE, "a.py", "b.py") is not None


def test_repository_without_commits_builds(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("")
    graph = build_graph(tmp_path)
    assert graph.files() == ["a.py"]
    assert graph.revision == ""


def test_subdirectory_of_repository_gets_history(repo):
    graph = build_graph(repo.root / "src")
    assert "pkg/core.py" in graph.nodes
    assert graph.edge(EVOLUTION, "pkg/core.py", "pkg/util.py").detail["count"] == 4
    assert graph.params["weights"] == DEFAULT_WEIGHTS


def test_cache_follows_working_tree_edits(repo):
    first = load_or_build(repo.root)
    assert cache_path(repo.root).exists()
    assert is_current(first, repo.root)
    target = repo.root / CORE
    target.write_text(target.read_text() + "import pkg.util\n")
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert not is_current(first, repo.root)
    assert is_current(load_or_build(repo.root), repo.root)


def test_unreadable_cache_is_rebuilt(repo):
    path = cache_path(repo.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": 0}))
    assert load_or_build(repo.root).files()


# --- ranking properties -------------------------------------------------------------------


def test_file_without_history_still_ranks_through_other_layers(repo):
    repo.write("src/pkg/new.py", "from pkg import util\n")
    graph = build_graph(repo.root)  # new.py is untracked: no history at all
    assert not graph.neighbors("src/pkg/new.py", EVOLUTION)
    ranked = [p for p, _ in Ranker(graph, DEFAULT_WEIGHTS).rank(["src/pkg/new.py"], k=3)]
    assert UTIL in ranked


def test_rank_many_matches_single_rankings(repo):
    graph = build_graph(repo.root)
    ranker = Ranker(graph, DEFAULT_WEIGHTS)
    many = ranker.rank_many([[CORE], [CFG], [CORE, CFG]], k=4)
    for batch, seed in ((many[0], CORE), (many[1], CFG)):
        single = ranker.rank([seed], k=4)
        # A batch stops when its slowest column converges, so scores agree
        # to the iteration tolerance rather than bit for bit.
        assert [p for p, _ in batch] == [p for p, _ in single]
        assert [s for _, s in batch] == pytest.approx([s for _, s in single], abs=1e-8)
    assert CORE not in {p for p, _ in many[2]} and CFG not in {p for p, _ in many[2]}


def test_scores_are_probability_distributions(repo):
    graph = build_graph(repo.root)
    scores = Ranker(graph, {"hierarchy": 0.5, "evolution": 0.5}).scores([[CORE], [UTIL]])
    assert np.allclose(scores.sum(axis=0), 1.0)
    assert (scores >= 0).all()


# --- analyses -----------------------------------------------------------------------------


def test_explain_file_lists_each_layer(repo):
    info = explain_file(build_graph(repo.root), CORE)
    assert set(info["layers"]) == {"hierarchy", "structure", "evolution", "docs"}
    assert info["commits"] >= 5
    with pytest.raises(KeyError):
        explain_file(build_graph(repo.root), "missing.py")


def test_connection_path_reports_layers_and_missing_links(repo):
    graph = build_graph(repo.root)
    hops = connection_path(graph, CFG, UTIL)
    assert hops[0]["from"] == CFG and hops[-1]["to"] == UTIL
    assert {h["layer"] for h in hops} <= {"hierarchy", "structure", "evolution", "docs"}
    graph.nodes["lonely.py"] = graph.nodes[CORE].__class__("lonely.py", "file")
    with pytest.raises(ValueError, match="no links"):
        connection_path(graph, CORE, "lonely.py")


def test_hub_files_need_enough_history():
    graph = SpletGraph(root=".")
    assert hub_files(graph) == set()


def test_hidden_dependency_distance_is_exact_for_reported_pairs(repo):
    hidden = hidden_dependencies(build_graph(repo.root), min_commits=2)
    pair = next(h for h in hidden if {h.a, h.b} == {CORE, CFG})
    assert pair.structural_distance is None  # a config file is never imported


def test_central_files_and_communities(repo):
    graph = build_graph(repo.root)
    assert central_files(graph)[0][0] in {CORE, UTIL}
    summary = community_summary(graph)
    assert sum(c["size"] for c in summary) == len(graph.files())


def test_report_mentions_every_layer(repo):
    graph = build_graph(repo.root)
    text = render_report(graph)
    for heading in ("## Layers", "## Communities", "## Central files", "## Hidden dependencies"):
        assert heading in text
    assert 0.0 <= layer_overlap(graph, STRUCTURE, EVOLUTION) <= 1.0


def test_graph_file_is_valid_json_with_format_version(repo):
    load_or_build(repo.root)
    data = json.loads(Path(cache_path(repo.root)).read_text())
    assert data["format"] == 1 and data["revision"]
