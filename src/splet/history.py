"""Reading git history with file paths normalized to one reference revision.

Paths change over a project's life (renames, moves into ``src/``). Co-change
statistics are only useful if every historical path is mapped to the name the
file has at the revision the graph is built for. Two directions are needed:

* backward: commits that are ancestors of the reference revision (training
  history), walked newest-first;
* forward: commits after the reference revision (evaluation), walked
  oldest-first.

A path that cannot be mapped (file deleted before the reference revision, or
added after it) resolves to ``None`` and is dropped.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

RECORD_SEP = "\x1e"
FIELD_SEP = "\x1f"


class GitError(RuntimeError):
    """A git command failed or the directory is not a git repository."""


@dataclass(frozen=True)
class Change:
    status: str  # A, M, D, R, C, T
    path: str  # path after the commit (new path for renames)
    old_path: str | None = None


@dataclass(frozen=True)
class Commit:
    sha: str
    timestamp: int
    changes: tuple[Change, ...]


@dataclass(frozen=True)
class NormalizedCommit:
    sha: str
    timestamp: int
    files: frozenset[str]


def run_git(repo: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise GitError("git executable not found") from exc
    except subprocess.CalledProcessError as exc:
        raise GitError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    return proc.stdout


def is_git_repo(repo: Path) -> bool:
    try:
        return run_git(repo, "rev-parse", "--is-inside-work-tree").strip() == "true"
    except GitError:
        return False


def has_commits(repo: Path) -> bool:
    """False for a freshly initialised repository without any commit."""
    try:
        run_git(repo, "rev-parse", "--verify", "--quiet", "HEAD")
    except GitError:
        return False
    return True


def resolve_rev(repo: Path, rev: str = "HEAD") -> str:
    return run_git(repo, "rev-parse", rev).strip()


def files_at(repo: Path, rev: str) -> set[str]:
    return set(run_git(repo, "ls-tree", "-r", "--name-only", rev).splitlines())


def commit_time(repo: Path, rev: str) -> int:
    return int(run_git(repo, "show", "-s", "--format=%ct", rev).strip())


def _parse_log(text: str) -> list[Commit]:
    commits = []
    for record in text.split(RECORD_SEP):
        lines = [ln for ln in record.splitlines() if ln.strip()]
        if not lines:
            continue
        sha, ts = lines[0].split(FIELD_SEP)
        changes = []
        for line in lines[1:]:
            parts = line.split("\t")
            status = parts[0][0]
            if status in ("R", "C") and len(parts) == 3:
                changes.append(Change(status, parts[2], parts[1]))
            elif len(parts) == 2:
                changes.append(Change(status, parts[1]))
        commits.append(Commit(sha, int(ts), tuple(changes)))
    return commits


def read_log(repo: Path, rev_range: str, reverse: bool = False) -> list[Commit]:
    """Non-merge commits in ``rev_range`` with rename detection enabled."""
    args = [
        "log",
        "--no-merges",
        "-M",
        "--name-status",
        f"--format={RECORD_SEP}%H{FIELD_SEP}%ct",
        rev_range,
    ]
    if reverse:
        args.insert(1, "--reverse")
    return _parse_log(run_git(repo, *args))


def normalize_backward(
    commits_newest_first: list[Commit], alive: set[str], follow_renames: bool = True
) -> list[NormalizedCommit]:
    """Map paths of ancestor commits to their names in ``alive`` (reference revision).

    With ``follow_renames=False`` a rename is treated as a deletion followed by
    an addition, which is what a tool that ignores renames effectively does.
    """
    alias: dict[str, str | None] = {}

    def resolve(path: str) -> str | None:
        if path in alias:
            return alias[path]
        return path if path in alive else None

    result = []
    for commit in commits_newest_first:
        files = set()
        for ch in commit.changes:
            target = resolve(ch.path)
            if target is not None and ch.status != "D":
                files.add(target)
            # Walking back in time: before this commit the file had another
            # name (rename) or did not exist (add).
            if ch.status == "R" and ch.old_path is not None:
                alias[ch.old_path] = target if follow_renames else None
                alias[ch.path] = None
            elif ch.status in ("A", "C") or ch.status == "D":
                alias[ch.path] = None
        if files:
            result.append(NormalizedCommit(commit.sha, commit.timestamp, frozenset(files)))
    return result


def normalize_forward(
    commits_oldest_first: list[Commit], alive: set[str]
) -> list[NormalizedCommit]:
    """Map paths of commits made after the reference revision back to its names."""
    current: dict[str, str | None] = {p: p for p in alive}
    result = []
    for commit in commits_oldest_first:
        files = set()
        for ch in commit.changes:
            if ch.status == "R" and ch.old_path is not None:
                current[ch.path] = current.pop(ch.old_path, None)
            elif ch.status in ("A", "C"):
                current[ch.path] = None
            target = current.get(ch.path)
            if target is not None:
                files.add(target)
            if ch.status == "D":
                current.pop(ch.path, None)
        if files:
            result.append(NormalizedCommit(commit.sha, commit.timestamp, frozenset(files)))
    return result


def history_until(
    repo: Path, rev: str, alive: set[str], follow_renames: bool = True
) -> list[NormalizedCommit]:
    """Ancestors of ``rev`` (inclusive), newest first, paths as of ``rev``."""
    return normalize_backward(read_log(repo, rev), alive, follow_renames)


def history_after(repo: Path, rev: str, head: str, alive: set[str]) -> list[NormalizedCommit]:
    """Commits reachable from ``head`` but not from ``rev``, oldest first, paths as of ``rev``."""
    return normalize_forward(read_log(repo, f"{rev}..{head}", reverse=True), alive)
