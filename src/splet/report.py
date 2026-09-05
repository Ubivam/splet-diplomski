"""Markdown report about a built graph."""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Callable

from .analysis import central_files, community_summary, hidden_dependencies
from .model import EVOLUTION, LAYERS, STRUCTURE, SpletGraph


def layer_overlap(graph: SpletGraph, a: str, b: str) -> float:
    """Jaccard similarity of the file–file edge sets of two layers."""
    files = set(graph.files())
    ea = {k for k in graph.layers[a] if k[0] in files and k[1] in files}
    eb = {k for k in graph.layers[b] if k[0] in files and k[1] in files}
    return len(ea & eb) / len(ea | eb) if ea | eb else 0.0


def _code(path: str) -> str:
    return f"`{path}`"


def render_report(graph: SpletGraph, link: Callable[[str], str] = _code) -> str:
    """Report text; ``link`` formats a file reference (plain code or wiki link)."""
    files = graph.files()
    langs = Counter(graph.nodes[f].lang for f in files)
    built = dt.datetime.fromtimestamp(graph.built_at).strftime("%Y-%m-%d %H:%M")
    lines = [
        "# splet report",
        "",
        f"Project `{graph.root}` at revision `{graph.revision[:10] or 'working tree'}`, "
        f"built {built}.",
        "",
        f"{len(files)} files ({', '.join(f'{n} {lang}' for lang, n in langs.most_common(5))}), "
        f"{graph.params.get('commits', 0)} commits analysed.",
        "",
        "## Layers",
        "",
        "| Layer | Edges | Weight in walk |",
        "|---|---:|---:|",
    ]
    weights = graph.params.get("weights", {})
    for name in LAYERS:
        lines.append(f"| {name} | {len(graph.layers[name])} | {weights.get(name, 0):.2f} |")
    lines += [
        "",
        f"Overlap of structural and evolutionary edges (Jaccard): "
        f"{layer_overlap(graph, STRUCTURE, EVOLUTION):.3f}. "
        "A low value means the history reveals couplings the imports do not show.",
        "",
    ]

    lines += ["## Communities", ""]
    for c in community_summary(graph):
        members = ", ".join(link(m) for m in c["top_members"][:4])
        lines.append(
            f"- **{c['id']}** — {c['size']} files, mostly in `{c['main_directory']}`: {members}"
        )

    lines += [
        "",
        "## Central files (most imported)",
        "",
        "| File | Importers | Commits |",
        "|---|---:|---:|",
    ]
    for path, importers, commits in central_files(graph):
        lines.append(f"| {link(path)} | {importers} | {commits} |")

    hidden = hidden_dependencies(graph)
    lines += [
        "",
        "## Hidden dependencies",
        "",
        "Files that change together although no short import path connects them.",
        "",
        "| File A | File B | Kind | Commits | Confidence | Import hops |",
        "|---|---|---|---:|---:|---:|",
    ]
    for h in hidden:
        hops = "none" if h.structural_distance is None else str(h.structural_distance)
        lines.append(
            f"| {link(h.a)} | {link(h.b)} | {h.category.replace('_', ' ')} | "
            f"{h.commits} | {h.confidence:.2f} | {hops} |"
        )
    if not hidden:
        lines.append("| — | — | — | — | — | — |")
    return "\n".join(lines) + "\n"
