"""Mapping Python module names to files of a project.

Handles regular packages (``__init__.py``), implicit namespace packages nested
in a regular package, the ``src/`` layout, scripts and tests that live in
directories without ``__init__.py``, and relative imports. A top-level
namespace package (no ``__init__.py`` anywhere above it) is indistinguishable
from a directory of scripts and is treated as one.
"""

from __future__ import annotations

from pathlib import PurePosixPath


def _package_dirs(py_files: list[str]) -> set[PurePosixPath]:
    """Directories that Python imports as packages.

    A directory with ``__init__.py`` is a regular package. A directory without
    it, but inside a package and holding Python code, is a namespace package
    (PEP 420), so ``import pkg.sub.mod`` works although ``pkg/sub`` has no
    ``__init__.py``.
    """
    packages = {PurePosixPath(f).parent for f in py_files if PurePosixPath(f).name == "__init__.py"}
    code_dirs: set[PurePosixPath] = set()
    for f in py_files:
        parent = PurePosixPath(f).parent
        while parent != PurePosixPath(".") and parent not in code_dirs:
            code_dirs.add(parent)
            parent = parent.parent
    changed = True
    while changed:
        nested = {d for d in code_dirs - packages if d.parent in packages}
        changed = bool(nested)
        packages |= nested
    return packages


class PythonModuleIndex:
    def __init__(self, py_files: list[str]):
        self.files = set(py_files)
        self.packages = _package_dirs(py_files)
        self.by_name: dict[str, str] = {}
        self.name_of: dict[str, str] = {}
        self._suffix: dict[str, set[str]] = {}
        for f in sorted(py_files):
            name = self._module_name(PurePosixPath(f))
            self.name_of[f] = name
            self.by_name.setdefault(name, f)
            parts = name.split(".")
            for i in range(len(parts)):
                self._suffix.setdefault(".".join(parts[i:]), set()).add(f)

    def _module_name(self, path: PurePosixPath) -> str:
        """Dotted name relative to the directory above the top-level package."""
        parts = [path.stem] if path.name != "__init__.py" else []
        parent = path.parent
        while parent in self.packages:
            parts.insert(0, parent.name)
            parent = parent.parent
        return ".".join(parts) if parts else path.parent.name

    def is_package(self, file: str) -> bool:
        return PurePosixPath(file).name == "__init__.py"

    def resolve(self, name: str, importer: str | None = None) -> str | None:
        """File that defines module ``name`` (or its closest existing parent)."""
        if importer is not None:
            local = self._sibling(name, importer)
            if local:
                return local
        probe = name
        while probe:
            if probe in self.by_name:
                return self.by_name[probe]
            candidates = self._suffix.get(probe, set())
            if len(candidates) == 1:
                return next(iter(candidates))
            probe = probe.rpartition(".")[0]
        return None

    def _sibling(self, name: str, importer: str) -> str | None:
        """Script-style import of a module that sits next to the importer."""
        directory = PurePosixPath(importer).parent
        if directory in self.packages:
            return None
        first = name.split(".", maxsplit=1)[0]
        for candidate in (directory / f"{first}.py", directory / first / "__init__.py"):
            if str(candidate) in self.files:
                return str(candidate)
        return None

    def absolute(self, module: str | None, level: int, importer: str) -> str:
        """Absolute dotted name for a (possibly relative) ``from`` import."""
        if level == 0:
            return module or ""
        base = self.name_of.get(importer, "").split(".")
        if not self.is_package(importer):
            base = base[:-1]
        base = base[: len(base) - (level - 1)] if level > 1 else base
        return ".".join([*base, module] if module else base)
