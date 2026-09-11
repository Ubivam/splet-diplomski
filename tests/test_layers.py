"""Layer builders and the pieces they rely on, tested on small in-memory inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from splet.history import Change, Commit, _parse_log, normalize_backward, normalize_forward
from splet.layers.docs import DocResolver, build_docs
from splet.layers.structure import js_imports, python_imports
from splet.model import DOCS, SpletGraph
from splet.pyindex import PythonModuleIndex
from splet.scan import SourceFile, is_test_path, language_of, scan


def write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


# --- history parsing and path normalisation -----------------------------------------------


def test_parse_log_reads_renames_copies_and_plain_changes():
    text = "\x1eabc\x1f100\n\nM\ta.py\nR087\told.py\tnew.py\nC100\tx.py\ty.py\nA\tz.py\n"
    (commit,) = _parse_log(text)
    assert commit.sha == "abc" and commit.timestamp == 100
    assert commit.changes == (
        Change("M", "a.py"),
        Change("R", "new.py", "old.py"),
        Change("C", "y.py", "x.py"),
        Change("A", "z.py"),
    )


def test_backward_normalisation_drops_an_earlier_file_with_the_same_name():
    # Newest first: b.py was re-created after an older b.py had been deleted.
    commits = [
        Commit("3", 3, (Change("A", "b.py"), Change("M", "a.py"))),
        Commit("2", 2, (Change("D", "b.py"),)),
        Commit("1", 1, (Change("M", "b.py"), Change("M", "a.py"))),
    ]
    out = normalize_backward(commits, {"a.py", "b.py"})
    assert [set(c.files) for c in out] == [{"a.py", "b.py"}, {"a.py"}]


def test_backward_normalisation_without_rename_tracking_loses_history():
    commits = [
        Commit("2", 2, (Change("R", "new.py", "old.py"),)),
        Commit("1", 1, (Change("M", "old.py"), Change("M", "other.py"))),
    ]
    followed = normalize_backward(commits, {"new.py", "other.py"})
    ignored = normalize_backward(commits, {"new.py", "other.py"}, follow_renames=False)
    assert followed[-1].files == {"new.py", "other.py"}
    assert ignored[-1].files == {"other.py"}


def test_forward_normalisation_maps_renamed_files_back():
    commits = [
        Commit("1", 1, (Change("R", "b.py", "a.py"),)),
        Commit("2", 2, (Change("M", "b.py"), Change("A", "c.py"))),
    ]
    out = normalize_forward(commits, {"a.py"})
    assert [set(c.files) for c in out] == [{"a.py"}, {"a.py"}]


# --- scanning -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_app.py", True),
        ("pkg/app_test.py", True),
        ("web/button.spec.ts", True),
        ("src/testing.py", False),
        ("src/app.py", False),
    ],
)
def test_is_test_path(path, expected):
    assert is_test_path(path) is expected


def test_language_of_known_and_unknown_files():
    assert language_of("a/b.PY") == "python"
    assert language_of("Makefile") == "build"
    assert language_of("image.png") is None


def test_plain_directory_walk_skips_tool_directories(tmp_path):
    write(
        tmp_path,
        {
            "app.py": "",
            "node_modules/lib/index.js": "",
            ".venv/lib/site.py": "",
            "build/out.py": "",
            "docs/index.md": "",
        },
    )
    assert [f.path for f in scan(tmp_path)] == ["app.py", "docs/index.md"]


def test_git_listing_keeps_tracked_build_directory(repo):
    repo.write("build/tool.py", "")
    repo.commit("tracked build helper")
    assert "build/tool.py" in {f.path for f in scan(repo.root)}


# --- Python module index ------------------------------------------------------------------


def test_python_index_sibling_suffix_and_parent_package():
    idx = PythonModuleIndex(
        ["scripts/run.py", "scripts/helpers.py", "lib/pkg/__init__.py", "lib/pkg/sub/mod.py"]
    )
    assert idx.resolve("helpers", importer="scripts/run.py") == "scripts/helpers.py"
    assert idx.resolve("sub.mod") == "lib/pkg/sub/mod.py"  # unique suffix
    assert idx.resolve("pkg.sub") == "lib/pkg/__init__.py"  # closest existing parent
    assert idx.resolve("unknown.module") is None
    assert idx.absolute(None, 2, "lib/pkg/sub/mod.py") == "pkg"


def test_python_imports_ignore_syntax_errors_and_external_modules():
    idx = PythonModuleIndex(["pkg/__init__.py", "pkg/a.py", "pkg/b.py"])
    assert python_imports("def broken(:\n", "pkg/a.py", idx) == {}
    found = python_imports("import os\nfrom pkg import b\nfrom .b import thing\n", "pkg/a.py", idx)
    assert found == {"pkg/b.py": ["b", "thing"]}


# --- JavaScript and TypeScript ------------------------------------------------------------


def test_js_imports_resolve_extensions_index_files_and_parents():
    known = {"src/app.ts", "src/util.ts", "src/lib/index.js", "shared/x.jsx"}
    source = (
        "import { a } from './util'\n"
        "import lib from './lib'\n"
        "const x = require('../shared/x')\n"
        "export * from 'react'\n"
    )
    assert js_imports(source, "src/app.ts", known) == [
        "src/util.ts",
        "src/lib/index.js",
        "shared/x.jsx",
    ]


# --- documentation ------------------------------------------------------------------------


def test_doc_resolver_understands_sphinx_roles_directives_and_links():
    files = [
        SourceFile("pkg/__init__.py", "python", False),
        SourceFile("pkg/core.py", "python", False),
        SourceFile("examples/demo.py", "python", False),
        SourceFile("docs/api.rst", "rst", False),
        SourceFile("docs/guide.rst", "rst", False),
    ]
    resolver = DocResolver(files, PythonModuleIndex([f.path for f in files if f.lang == "python"]))
    text = (
        "See :class:`~pkg.core.Engine` and :doc:`guide`.\n"
        ".. automodule:: pkg.core\n"
        ".. literalinclude:: ../examples/demo.py\n"
        "External `link <https://example.org>`_.\n"
    )
    refs = set(resolver.references(text, "docs/api.rst"))
    assert ("pkg/core.py", "role") in refs
    assert ("docs/guide.rst", "role") in refs
    assert ("pkg/core.py", "directive") in refs
    assert ("examples/demo.py", "directive") in refs
    assert all("example.org" not in target for target, _ in refs)


def test_docs_layer_weights_links_above_mentions(tmp_path):
    write(
        tmp_path,
        {
            "README.md": "[core](pkg/core.py) and `pkg.util`\n",
            "pkg/__init__.py": "",
            "pkg/core.py": "",
            "pkg/util.py": "",
        },
    )
    files = scan(tmp_path)
    graph = SpletGraph(root=str(tmp_path))
    index = PythonModuleIndex([f.path for f in files if f.lang == "python"])
    build_docs(graph, tmp_path, files, index)
    assert graph.edge(DOCS, "README.md", "pkg/core.py").weight == 1.0
    assert graph.edge(DOCS, "README.md", "pkg/util.py").weight == 0.5
