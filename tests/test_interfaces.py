"""Command line, MCP server and Obsidian export as a user would meet them."""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from mcp import Client

from splet import __version__
from splet.build import build_graph
from splet.cli import main
from splet.obsidian import count_word, export_vault
from splet.server import GraphStore, create_server

CORE = "src/pkg/core.py"


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "1 комит"),
        (2, "2 комита"),
        (4, "4 комита"),
        (5, "5 комитова"),
        (11, "11 комитова"),
        (12, "12 комитова"),
        (21, "21 комит"),
        (22, "22 комита"),
        (111, "111 комитова"),
        (0, "0 комитова"),
    ],
)
def test_serbian_plural_forms(n, expected):
    assert count_word(n, "commits", "sr") == expected


def test_english_plural_forms():
    assert count_word(1, "lines", "en") == "1 line"
    assert count_word(3, "lines", "en") == "3 lines"


def test_cli_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize("command", ["hidden", "report", "explain"])
def test_cli_read_only_commands(repo, capsys, command):
    args = [command, "--repo", str(repo.root)] + ([CORE] if command == "explain" else [])
    assert main(args) == 0
    out = capsys.readouterr().out
    if command == "explain":
        assert json.loads(out)["path"] == CORE
    else:
        assert out.strip()


def test_cli_obsidian_and_report_files(repo, tmp_path, capsys):
    vault, report = tmp_path / "vault", tmp_path / "report.md"
    assert main(["obsidian", "--repo", str(repo.root), "--out", str(vault), "--lang", "sr"]) == 0
    assert main(["report", "--repo", str(repo.root), "--out", str(report)]) == 0
    assert (vault / f"{CORE}.md").exists() and report.read_text().startswith("# splet report")


def test_cli_reports_errors_without_traceback(repo, tmp_path, capsys):
    (tmp_path / "notes").mkdir()
    code = main(["obsidian", "--repo", str(repo.root), "--out", str(tmp_path / "notes")])
    assert code == 1 and "refusing to overwrite" in capsys.readouterr().err


def test_vault_links_only_to_existing_notes(repo, tmp_path):
    vault = export_vault(build_graph(repo.root), tmp_path / "v")
    for note in vault.rglob("*.md"):
        for target in re.findall(r"\[\[([^|\]]+)", note.read_text()):
            assert (vault / f"{target}.md").exists(), (note, target)


# --- MCP server ---------------------------------------------------------------------------


def call(server, name, **arguments):
    return asyncio.run(server.call_tool(name, arguments))


def test_mcp_related_files_tool(repo):
    server = create_server(repo.root)
    result = call(server, "related_files", paths=[CORE], limit=3)
    rows = result.structured_content["result"] if hasattr(result, "structured_content") else result
    assert rows and all({"path", "score", "reasons"} <= set(r) for r in rows)


def test_mcp_every_read_only_tool_answers(repo):
    server = create_server(repo.root)
    for name, args in [
        ("explain_file", {"path": CORE}),
        ("hidden_dependencies", {}),
        ("communities", {}),
        ("connection_path", {"source": CORE, "target": "src/pkg/util.py"}),
        ("project_report", {}),
    ]:
        call(server, name, **args)


def test_mcp_client_receives_request_errors(repo, tmp_path):
    """A wrong request comes back as a readable tool error, not a server crash."""
    foreign = tmp_path / "notes"
    foreign.mkdir()

    async def errors() -> list[str]:
        async with Client(create_server(repo.root)) as client:
            texts = []
            for name, args in [
                ("related_files", {"paths": ["src/pkg/cor.py"]}),
                ("connection_path", {"source": CORE, "target": "/etc/hosts"}),
                ("export_obsidian", {"target_directory": str(foreign)}),
                ("export_obsidian", {"target_directory": str(foreign), "language": "de"}),
            ]:
                result = await client.call_tool(name, args)
                assert result.is_error
                texts.append(result.content[0].text)
            return texts

    wrong_path, outside, occupied, language = asyncio.run(errors())
    assert f"Did you mean: {CORE}" in wrong_path
    assert "outside the project" in outside
    assert "not a splet vault" in occupied
    assert "language must be one of" in language


def test_graph_store_normalises_paths(repo):
    store = GraphStore(repo.root)
    assert store.normalize(str(repo.root / CORE)) == CORE
    assert store.normalize(f"./{CORE}") == CORE
    with pytest.raises(ValueError, match="Did you mean"):
        store.normalize("src/pkg/cor.py")
    with pytest.raises(ValueError, match="outside the project"):
        store.normalize("/etc/hosts")


def test_graph_store_reuses_ranker_until_project_changes(repo):
    store = GraphStore(repo.root)
    first = store.ranker()
    assert store.ranker() is first
    store.get(rebuild=True)
    assert store.ranker() is not first
