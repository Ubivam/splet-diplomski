"""MCP server that exposes the multilayer graph of one project to AI tools.

Start with ``splet serve --repo PATH`` (stdio transport). The graph is built on
the first request, cached in ``PATH/.splet/graph.json`` and rebuilt when the
git ``HEAD`` moves.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from .analysis import (
    community_summary,
    connection_path,
    explain_file,
    hidden_dependencies,
    related_files,
)
from .build import is_current, load_or_build
from .fusion import Ranker
from .model import SpletGraph
from .obsidian import HEADINGS, export_vault
from .report import render_report

INSTRUCTIONS = """\
splet holds a multilayer graph of the current project: imports (structure),
files that historically change together (evolution), documentation references
and the directory tree. Before editing a file, call `related_files` with the
files you plan to change: it lists the other files that usually need to be read
or updated together, each with the reason (import, N common commits, test,
documentation). `hidden_dependencies` lists couplings that imports do not show.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


class RequestError(ToolError, ValueError):
    """A request the tool cannot answer (unknown path, no connection, occupied
    directory). Unlike an unexpected exception, its message reaches the client."""


class GraphStore:
    """The graph of one project, rebuilt when the project changes."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._graph: SpletGraph | None = None
        self._ranker: Ranker | None = None

    def get(self, rebuild: bool = False) -> SpletGraph:
        if rebuild or self._graph is None or not is_current(self._graph, self.root):
            self._graph = load_or_build(self.root, rebuild=rebuild)
            self._ranker = None
        return self._graph

    def ranker(self) -> Ranker:
        graph = self.get()
        if self._ranker is None:
            self._ranker = Ranker(graph, graph.params["weights"])
        return self._ranker

    def normalize(self, path: str) -> str:
        """Project-relative path for an absolute or relative file name."""
        graph = self.get()
        p = Path(path)
        if p.is_absolute():
            try:
                rel = p.resolve().relative_to(self.root).as_posix()
            except ValueError:
                raise RequestError(f"{path!r} is outside the project {self.root}") from None
        else:
            rel = p.as_posix()
        rel = rel.removeprefix("./")
        if rel not in graph.nodes:
            close = difflib.get_close_matches(rel, graph.files(), n=3, cutoff=0.5)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise RequestError(f"{path!r} is not a file of the project.{hint}")
        return rel


def create_server(root: Path) -> MCPServer:
    store = GraphStore(root)
    server = MCPServer(name="splet", instructions=INSTRUCTIONS)

    @server.tool(name="related_files", annotations=READ_ONLY)
    def related_files_tool(
        paths: list[str], limit: int = 10, code_only: bool = False
    ) -> list[dict[str, Any]]:
        """Files most related to the given ones across all layers, each with its reason."""
        seeds = [store.normalize(p) for p in paths]
        return [
            asdict(r)
            for r in related_files(
                store.get(), seeds, k=limit, code_only=code_only, ranker=store.ranker()
            )
        ]

    @server.tool(name="explain_file", annotations=READ_ONLY)
    def explain_file_tool(path: str) -> dict[str, Any]:
        """Neighbours of one file in every layer, its community and history."""
        return explain_file(store.get(), store.normalize(path))

    @server.tool(name="hidden_dependencies", annotations=READ_ONLY)
    def hidden_dependencies_tool(limit: int = 15, min_commits: int = 3) -> list[dict[str, Any]]:
        """File pairs that change together although no short import path links them."""
        return [asdict(h) for h in hidden_dependencies(store.get(), min_commits, limit)]

    @server.tool(name="communities", annotations=READ_ONLY)
    def communities_tool() -> list[dict[str, Any]]:
        """Groups of closely connected files (Leiden communities) with their main members."""
        return community_summary(store.get())

    @server.tool(name="connection_path", annotations=READ_ONLY)
    def connection_path_tool(source: str, target: str) -> list[dict[str, Any]]:
        """Strongest chain of links between two files and the layer of every hop."""
        a, b = store.normalize(source), store.normalize(target)
        try:
            return connection_path(store.get(), a, b)
        except ValueError as exc:
            raise RequestError(str(exc)) from exc

    @server.tool(annotations=READ_ONLY)
    def project_report() -> str:
        """Markdown overview: layers, communities, central files, hidden dependencies."""
        return render_report(store.get())

    @server.tool()
    def rebuild_graph() -> dict[str, Any]:
        """Rebuild the graph from the working tree and git history."""
        g = store.get(rebuild=True)
        return {
            "files": len(g.files()),
            "revision": g.revision,
            "edges": {name: len(e) for name, e in g.layers.items()},
        }

    @server.tool()
    def export_obsidian(target_directory: str, language: str = "en") -> str:
        """Write the graph as an Obsidian vault (notes coloured by community)."""
        if language not in HEADINGS:
            raise RequestError(f"language must be one of: {', '.join(HEADINGS)}")
        try:
            out = export_vault(store.get(), Path(target_directory).expanduser(), language)
        except FileExistsError as exc:
            raise RequestError(str(exc)) from exc
        return f"Vault written to {out}. Open it in Obsidian and use the graph view."

    return server


def serve(root: Path | None = None) -> None:
    root = root or Path(os.environ.get("SPLET_REPO", os.getcwd()))
    create_server(root).run("stdio")
