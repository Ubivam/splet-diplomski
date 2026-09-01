"""Documentation layer: links from prose documents to project files.

Recognised references:

* Markdown links and images, ``[text](relative/path)``;
* reStructuredText/Sphinx roles and directives, ``:class:`pkg.mod.Name```,
  ``.. automodule:: pkg.mod``, ``.. literalinclude:: ../src/x.py``;
* inline code that names a file path or a dotted Python module,
  ```src/pkg/mod.py``` or ```pkg.mod```.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from ..model import DOCS, SpletGraph
from ..pyindex import PythonModuleIndex
from ..scan import DOC_LANGUAGES, SourceFile, read_text
from .structure import _normalize

MD_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
RST_ROLE = re.compile(
    r":(?:py:)?(?:mod|class|func|meth|attr|exc|data|obj|doc):`~?!?([^`<]+?)(?:\s*<([^>]+)>)?`"
)
RST_DIRECTIVE = re.compile(
    r"^\s*\.\.\s+(automodule|autoclass|autofunction|currentmodule|module|literalinclude|include)::\s*(\S+)",
    re.MULTILINE,
)
INLINE_CODE = re.compile(r"``?([A-Za-z_][\w./-]*[\w])``?")
DOTTED = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")


class DocResolver:
    def __init__(self, files: list[SourceFile], index: PythonModuleIndex):
        self.known = {f.path for f in files}
        self.index = index
        self.docs_by_stem: dict[str, str] = {}
        for f in files:
            if f.lang in DOC_LANGUAGES:
                self.docs_by_stem.setdefault(str(PurePosixPath(f.path).with_suffix("")), f.path)

    def path(self, target: str, doc: str) -> str | None:
        target = target.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            return None
        base = PurePosixPath(doc).parent
        rel = (
            _normalize(PurePosixPath(target.lstrip("/")))
            if target.startswith("/")
            else _normalize(base / target)
        )
        if rel in self.known:
            return rel
        return self.docs_by_stem.get(rel)

    def dotted(self, name: str) -> str | None:
        name = name.strip().lstrip("~!.")
        if not DOTTED.match(name) and name not in self.index.by_name:
            return None
        return self.index.resolve(name)

    def references(self, text: str, doc: str) -> list[tuple[str, str]]:
        """(target file, kind) pairs found in one document."""
        refs: list[tuple[str, str]] = []
        for target in MD_LINK.findall(text):
            if hit := self.path(target, doc):
                refs.append((hit, "link"))
        for title, explicit in RST_ROLE.findall(text):
            target = explicit or title
            hit = self.path(target, doc) or self.dotted(target)
            if hit:
                refs.append((hit, "role"))
        for directive, target in RST_DIRECTIVE.findall(text):
            hit = (
                self.path(target, doc)
                if directive in ("literalinclude", "include")
                else self.dotted(target)
            )
            if hit:
                refs.append((hit, "directive"))
        for token in INLINE_CODE.findall(text):
            hit = (
                self.path(token, doc)
                or (token if token in self.known else None)
                or self.dotted(token)
            )
            if hit:
                refs.append((hit, "mention"))
        return [(t, k) for t, k in refs if t != doc]


def build_docs(
    graph: SpletGraph, root: Path, files: list[SourceFile], index: PythonModuleIndex
) -> None:
    resolver = DocResolver(files, index)
    for f in files:
        if f.lang not in DOC_LANGUAGES:
            continue
        counts: dict[str, dict[str, int]] = {}
        for target, kind in resolver.references(read_text(root, f.path), f.path):
            counts.setdefault(target, {}).setdefault(kind, 0)
            counts[target][kind] += 1
        for target, kinds in counts.items():
            for kind, n in kinds.items():
                graph.add_edge(
                    DOCS, f.path, target, 1.0 if kind == "link" else 0.5, kind, mentions=n
                )
