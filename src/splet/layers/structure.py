"""Structural layer: dependencies that are visible in the source code.

Python imports are read from the AST; JavaScript/TypeScript imports from the
module specifiers of ``import``/``export``/``require`` statements. Only
dependencies between files of the project are kept.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path, PurePosixPath

from ..model import STRUCTURE, SpletGraph
from ..pyindex import PythonModuleIndex
from ..scan import SourceFile, read_text

MAX_SYMBOLS = 12
JS_SPECIFIER = re.compile(
    r"""(?:\bimport\s+(?:[\w*{}\s,$]+\s+from\s+)?|\bexport\s+[\w*{}\s,$]+\s+from\s+|"""
    r"""\brequire\s*\(\s*|\bimport\s*\(\s*)['"]([^'"]+)['"]"""
)
JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


def python_imports(source: str, importer: str, index: PythonModuleIndex) -> dict[str, list[str]]:
    """Map of imported project file → imported names, for one Python file."""
    try:
        with warnings.catch_warnings():
            # Old sources contain escapes such as "\s" that Python now warns about.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    found: dict[str, list[str]] = {}

    def record(target: str | None, symbol: str) -> None:
        if target and target != importer:
            names = found.setdefault(target, [])
            if symbol and symbol not in names and len(names) < MAX_SYMBOLS:
                names.append(symbol)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                record(index.resolve(alias.name, importer), alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = index.absolute(node.module, node.level, importer)
            for alias in node.names:
                # ``from pkg import sub`` may name a submodule rather than an attribute.
                sub = index.by_name.get(f"{base}.{alias.name}") if base else None
                record(
                    sub or index.resolve(base, importer if node.level == 0 else None), alias.name
                )
    return found


def js_imports(source: str, importer: str, known: set[str]) -> list[str]:
    targets = []
    base = PurePosixPath(importer).parent
    for spec in JS_SPECIFIER.findall(source):
        if not spec.startswith("."):
            continue
        stem = _normalize(base / spec)
        for candidate in [
            stem,
            *(stem + ext for ext in JS_EXTENSIONS),
            *(f"{stem}/index{ext}" for ext in JS_EXTENSIONS),
        ]:
            if candidate in known and candidate != importer:
                targets.append(candidate)
                break
    return targets


def _normalize(path: PurePosixPath) -> str:
    parts: list[str] = []
    for part in path.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


def build_structure(
    graph: SpletGraph, root: Path, files: list[SourceFile], index: PythonModuleIndex
) -> None:
    known = {f.path for f in files}
    for f in files:
        if f.lang == "python":
            imports = python_imports(read_text(root, f.path), f.path, index)
            for target, symbols in imports.items():
                graph.add_edge(
                    STRUCTURE,
                    f.path,
                    target,
                    1.0,
                    "import",
                    imports=[f"{f.path} -> {target}"],
                    symbols=symbols,
                )
        elif f.lang in ("javascript", "typescript"):
            for target in js_imports(read_text(root, f.path), f.path, known):
                graph.add_edge(
                    STRUCTURE, f.path, target, 1.0, "import", imports=[f"{f.path} -> {target}"]
                )
