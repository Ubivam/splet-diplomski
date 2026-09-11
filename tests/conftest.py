"""A small git repository with a known history, built fresh for each test."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


class Repo:
    def __init__(self, root: Path):
        self.root = root
        self.clock = 1_600_000_000
        root.mkdir(parents=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True,
            text=True,
            check=True,
            env={
                **ENV,
                "GIT_AUTHOR_DATE": f"@{self.clock}",
                "GIT_COMMITTER_DATE": f"@{self.clock}",
            },
        ).stdout

    def write(self, path: str, text: str) -> None:
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def commit(self, message: str, days: int = 1) -> str:
        self.clock += days * 86_400
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    """History:

    1. pkg/core.py, pkg/util.py, pkg/__init__.py, README.md
    2. core+util changed together
    3. pkg/ moved to src/pkg/ (rename)
    4. core+util changed together again, plus tests/test_core.py
    5. core + config/settings.toml changed together (no import between them)
    6. core + config/settings.toml again
    7. docs/api.md added
    """
    r = Repo(tmp_path / "proj")
    r.write("pkg/__init__.py", "from .core import run\n")
    r.write(
        "pkg/core.py",
        "from . import util\nfrom pkg.util import helper\n\ndef run():\n    return helper()\n",
    )
    r.write("pkg/util.py", "def helper():\n    return 1\n")
    r.write("README.md", "See [core](pkg/core.py).\n")
    r.commit("initial")
    r.write("pkg/core.py", r.root.joinpath("pkg/core.py").read_text() + "# v2\n")
    r.write("pkg/util.py", r.root.joinpath("pkg/util.py").read_text() + "# v2\n")
    r.commit("core and util")
    r.git("mv", "pkg", "src")
    (r.root / "src" / "pkg").mkdir()
    for name in ("__init__.py", "core.py", "util.py"):
        r.git("mv", f"src/{name}", f"src/pkg/{name}")
    r.write("README.md", "See [core](src/pkg/core.py).\n")
    r.commit("move to src layout")
    r.write("src/pkg/core.py", r.root.joinpath("src/pkg/core.py").read_text() + "# v3\n")
    r.write("src/pkg/util.py", r.root.joinpath("src/pkg/util.py").read_text() + "# v3\n")
    r.write(
        "tests/test_core.py", "from pkg.core import run\n\ndef test_run():\n    assert run() == 1\n"
    )
    r.commit("core, util and a test")
    for i in (1, 2):
        r.write("src/pkg/core.py", r.root.joinpath("src/pkg/core.py").read_text() + f"# cfg {i}\n")
        r.write("config/settings.toml", f"level = {i}\n")
        r.commit(f"config {i}")
    r.write("docs/api.md", "The entry point is `pkg.core`; helpers live in `src/pkg/util.py`.\n")
    r.commit("docs")
    return r
