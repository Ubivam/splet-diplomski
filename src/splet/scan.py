"""Discovering the files of a project and classifying them."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .history import GitError, is_git_repo, run_git

MAX_FILE_BYTES = 1_000_000

LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".toml": "config",
    ".yaml": "config",
    ".yml": "config",
    ".json": "config",
    ".cfg": "config",
    ".ini": "config",
    ".html": "template",
    ".jinja": "template",
    ".j2": "template",
    ".css": "style",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".sh": "shell",
    ".sql": "sql",
    ".tex": "latex",
}
DOC_LANGUAGES = {"markdown", "rst", "text"}
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    ".splet",
    ".idea",
    ".vscode",
    "site-packages",
}
# Never part of the project, even when a repository tracks them by mistake.
ALWAYS_SKIP = {".git", ".splet"}
EXTRA_NAMES = {
    "Makefile": "build",
    "Dockerfile": "build",
    "LICENSE": "text",
    "CHANGES": "text",
    "AUTHORS": "text",
}


@dataclass(frozen=True)
class SourceFile:
    path: str  # POSIX path relative to the project root
    lang: str
    is_test: bool


def language_of(path: str) -> str | None:
    p = PurePosixPath(path)
    return LANGUAGES.get(p.suffix.lower()) or EXTRA_NAMES.get(p.name)


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name.lower()
    return (
        any(part in ("test", "tests", "testing", "__tests__", "spec") for part in p.parts[:-1])
        or name.startswith("test_")
        or name.endswith(("_test.py", ".test.ts", ".test.js", ".spec.ts", ".spec.js", "_test.go"))
    )


def _walk(root: Path) -> list[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            found.append((Path(dirpath) / name).relative_to(root).as_posix())
    return found


def list_candidates(root: Path) -> list[str]:
    """Tracked files for git repositories, a filtered walk otherwise."""
    if is_git_repo(root):
        try:
            prefix = run_git(root, "rev-parse", "--show-prefix").strip()
            listed = run_git(root, "ls-files", "--cached", "--others", "--exclude-standard")
            return [
                p[len(prefix) :] if prefix and p.startswith(prefix) else p
                for p in listed.splitlines()
                if p
            ]
        except GitError:
            pass
    return _walk(root)


def scan(root: Path, candidates: list[str] | None = None) -> list[SourceFile]:
    """Text files of known types under ``root``, sorted by path."""
    files = []
    for rel in candidates if candidates is not None else list_candidates(root):
        # Git's own list already honours .gitignore, so SKIP_DIRS is applied
        # only while walking a plain directory (see _walk).
        if any(part in ALWAYS_SKIP for part in PurePosixPath(rel).parts):
            continue
        lang = language_of(rel)
        full = root / rel
        if lang is None or not full.is_file() or full.stat().st_size > MAX_FILE_BYTES:
            continue
        files.append(SourceFile(rel, lang, is_test_path(rel)))
    return sorted(files, key=lambda f: f.path)


def read_text(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8", errors="replace")
