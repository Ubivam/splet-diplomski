"""Export of the graph as an Obsidian vault.

Every project file becomes a note at the same relative path (``src/app.py`` →
``src/app.py.md``) whose wiki links are the strongest neighbours in each layer.
Notes carry a community tag, and ``.obsidian/graph.json`` assigns a colour to
each community, so Obsidian's graph view shows the modules of the project in
colour. An index note holds the report and the hidden dependencies.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .analysis import Adjacency, hidden_dependencies, hub_files
from .model import DOCS, EVOLUTION, HIERARCHY, STRUCTURE, SpletGraph
from .report import render_report
from .scan import DOC_LANGUAGES

# Colour-blind-friendly qualitative palette (Okabe–Ito extended).
PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#8C564B",
    "#17BECF",
    "#7F7F7F",
    "#BCBD22",
    "#9467BD",
]
MAX_LINKS_PER_LAYER = 8
MARKER = ".splet-vault"

HEADINGS = {
    "en": {
        STRUCTURE: "Code dependencies (imports)",
        EVOLUTION: "Changes together with",
        DOCS: "Documentation",
        HIERARCHY: "Tests",
        "hidden": "Hidden dependencies",
        "community": "community",
        "index": "_splet",
        "hidden_note": "_hidden dependencies",
    },
    "sr": {
        STRUCTURE: "Зависности у коду (импорти)",
        EVOLUTION: "Мења се заједно са",
        DOCS: "Документација",
        HIERARCHY: "Тестови",
        "hidden": "Скривене зависности",
        "community": "заједница",
        "index": "_splet",
        "hidden_note": "_скривене зависности",
    },
}
UNITS: dict[str, dict[str, tuple[str, ...]]] = {
    "en": {"commits": ("commit", "commits"), "lines": ("line", "lines")},
    "sr": {"commits": ("комит", "комита", "комитова"), "lines": ("линија", "линије", "линија")},
}


def count_word(n: int, unit: str, language: str) -> str:
    """``n`` with the unit in the grammatical number the language requires."""
    forms = UNITS[language][unit]
    if language == "sr":
        if n % 10 == 1 and n % 100 != 11:
            form = forms[0]
        elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
            form = forms[1]
        else:
            form = forms[2]
    else:
        form = forms[0] if n == 1 else forms[1]
    return f"{n} {form}"


def community_tag(c: int | None) -> str:
    return "splet/c-none" if c is None else f"splet/c{c:02d}"


def wiki(path: str) -> str:
    return f"[[{path}|{path.rpartition('/')[2]}]]"


def _note(
    graph: SpletGraph,
    adj: Adjacency,
    path: str,
    *,
    hidden_for: dict[str, list[tuple[str, int]]],
    noise: set[str],
    language: str,
) -> str:
    h = HEADINGS[language]
    node = graph.nodes[path]
    tags = [community_tag(graph.communities.get(path)), f"splet/{node.lang}"]
    if node.is_test:
        tags.append("splet/test")
    if path in hidden_for:
        tags.append("splet/hidden")
    out = [
        "---",
        f'path: "{path}"',
        f"language: {node.lang}",
        f"lines: {node.lines}",
        f"commits: {node.changes}",
        f"community: {graph.communities.get(path)}",
        "tags:",
        *(f"  - {t}" for t in tags),
        "---",
        "",
        f"`{path}` · {node.lang} · {count_word(node.lines, 'lines', language)}"
        f" · {count_word(node.changes, 'commits', language)}"
        f" · {h['community']} {graph.communities.get(path)}",
        "",
    ]
    for layer in (STRUCTURE, EVOLUTION, DOCS, HIERARCHY):
        nbrs = [(p, e) for p, e in adj.neighbors(path, layer).items() if not p.startswith("dir:")]
        if layer == HIERARCHY:
            nbrs = [(p, e) for p, e in nbrs if "tests" in e.kinds]
        if layer == EVOLUTION:
            # Changelogs and other hubs pair with everything; documents have their own section.
            nbrs = [(p, e) for p, e in nbrs if e.detail.get("count", 0) >= 2 and p not in noise]
        if not nbrs:
            continue
        nbrs.sort(key=lambda pe: (-pe[1].detail.get("count", 0), -pe[1].weight, pe[0]))
        out += [f"## {h[layer]}", ""]
        for p, e in nbrs[:MAX_LINKS_PER_LAYER]:
            suffix = (
                f" — {count_word(e.detail['count'], 'commits', language)}"
                if layer == EVOLUTION
                else ""
            )
            out.append(f"- {wiki(p)}{suffix}")
        out.append("")
    if path in hidden_for:
        out += [f"## {h['hidden']}", ""]
        out += [
            f"- {wiki(other)} — {count_word(n, 'commits', language)}"
            for other, n in hidden_for[path]
        ]
        out.append("")
    return "\n".join(out)


def _graph_settings(graph: SpletGraph) -> dict[str, Any]:
    groups = []
    for c in sorted(set(graph.communities.values()))[: len(PALETTE)]:
        rgb = int(PALETTE[c].lstrip("#"), 16)
        groups.append({"query": f"tag:#{community_tag(c)}", "color": {"a": 1, "rgb": rgb}})
    return {
        "colorGroups": groups,
        "showTags": False,
        "showAttachments": False,
        "hideUnresolved": True,
        "showOrphans": True,
        "showArrow": False,
        "nodeSizeMultiplier": 1.1,
        "lineSizeMultiplier": 0.6,
        "centerStrength": 0.45,
        "repelStrength": 12,
        "linkStrength": 1,
        "linkDistance": 180,
    }


def export_vault(graph: SpletGraph, target: Path, language: str = "en") -> Path:
    """Write the vault into ``target``; an earlier export there is replaced."""
    h = HEADINGS[language]
    if target.exists():
        if not (target / MARKER).exists():
            raise FileExistsError(
                f"{target} exists and is not a splet vault; refusing to overwrite"
            )
        shutil.rmtree(target)
    (target / ".obsidian").mkdir(parents=True)
    (target / MARKER).write_text(graph.revision + "\n", encoding="utf-8")

    adj = Adjacency(graph)
    noise = hub_files(graph) | {f for f in graph.files() if graph.nodes[f].lang in DOC_LANGUAGES}
    hidden = hidden_dependencies(graph, limit=50)
    hidden_for: dict[str, list[tuple[str, int]]] = {}
    for d in hidden:
        hidden_for.setdefault(d.a, []).append((d.b, d.commits))
        hidden_for.setdefault(d.b, []).append((d.a, d.commits))

    for path in graph.files():
        note = target / f"{path}.md"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(
            _note(graph, adj, path, hidden_for=hidden_for, noise=noise, language=language),
            encoding="utf-8",
        )

    (target / f"{h['index']}.md").write_text(render_report(graph, link=wiki), encoding="utf-8")
    hidden_lines = [f"# {h['hidden']}", ""] + [
        f"- {wiki(d.a)} ↔ {wiki(d.b)} — {count_word(d.commits, 'commits', language)}"
        for d in hidden
    ]
    (target / f"{h['hidden_note']}.md").write_text("\n".join(hidden_lines) + "\n", encoding="utf-8")
    (target / ".obsidian" / "graph.json").write_text(
        json.dumps(_graph_settings(graph), indent=2), encoding="utf-8"
    )
    return target
