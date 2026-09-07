"""Command line interface: ``splet <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .analysis import explain_file, hidden_dependencies, related_files
from .build import cache_path, load_or_build
from .history import GitError
from .obsidian import export_vault
from .report import render_report


def _root(args: argparse.Namespace) -> Path:
    root = Path(args.repo).resolve()
    if not root.is_dir():
        raise SystemExit(f"splet: {root} is not a directory")
    return root


def _rel(root: Path, path: str) -> str:
    p = Path(path)
    return (p.resolve().relative_to(root) if p.is_absolute() or p.exists() else p).as_posix()


def cmd_build(args: argparse.Namespace) -> None:
    root = _root(args)
    g = load_or_build(root, rebuild=True)
    edges = ", ".join(f"{name} {len(e)}" for name, e in g.layers.items())
    print(f"{len(g.files())} files, {g.params.get('commits', 0)} commits; edges: {edges}")
    print(f"communities: {len(set(g.communities.values()))}; graph saved to {cache_path(root)}")


def cmd_related(args: argparse.Namespace) -> None:
    root = _root(args)
    g = load_or_build(root)
    seeds = [_rel(root, p) for p in args.files]
    missing = [s for s in seeds if s not in g.nodes]
    if missing:
        raise SystemExit(f"splet: not project files: {', '.join(missing)}")
    rows = related_files(g, seeds, k=args.limit, code_only=args.code_only)
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=False))
        return
    for r in rows:
        print(f"{r.score:8.4f}  {r.path}\n          {'; '.join(r.reasons)}")


def cmd_explain(args: argparse.Namespace) -> None:
    root = _root(args)
    print(
        json.dumps(
            explain_file(load_or_build(root), _rel(root, args.file)), indent=2, ensure_ascii=False
        )
    )


def cmd_hidden(args: argparse.Namespace) -> None:
    found = hidden_dependencies(load_or_build(_root(args)), args.min_commits, args.limit)
    if not found:
        print(f"no hidden dependencies (pairs changed together in >= {args.min_commits} commits)")
    for h in found:
        hops = (
            "no import path" if h.structural_distance is None else f"{h.structural_distance} hops"
        )
        print(
            f"{h.commits:4d} commits  conf {h.confidence:.2f}  {hops:15s} "
            f"{h.category:15s} {h.a}  <->  {h.b}"
        )


def cmd_report(args: argparse.Namespace) -> None:
    text = render_report(load_or_build(_root(args)))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"report written to {args.out}")
    else:
        print(text)


def cmd_obsidian(args: argparse.Namespace) -> None:
    root = _root(args)
    out = Path(args.out) if args.out else root.parent / f"{root.name}-splet-vault"
    export_vault(load_or_build(root), out, args.lang)
    print(f"vault written to {out}")


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import serve  # noqa: PLC0415 (MCP stack loads only for this command)

    serve(_root(args))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="splet", description="Multilayer knowledge graph of a project."
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add(
        name: str, fn: Callable[[argparse.Namespace], None], help_: str
    ) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--repo", default=".", help="project directory (default: current)")
        sp.set_defaults(func=fn)
        return sp

    add("build", cmd_build, "build (or rebuild) the graph")
    sp = add("related", cmd_related, "files related to the given files")
    sp.add_argument("files", nargs="+")
    sp.add_argument("-k", "--limit", type=int, default=10)
    sp.add_argument("--code-only", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp = add("explain", cmd_explain, "neighbours of a file in every layer")
    sp.add_argument("file")
    sp = add("hidden", cmd_hidden, "couplings not visible in imports")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--min-commits", type=int, default=3)
    sp = add("report", cmd_report, "markdown report")
    sp.add_argument("--out")
    sp = add("obsidian", cmd_obsidian, "export an Obsidian vault")
    sp.add_argument("--out")
    sp.add_argument("--lang", choices=("en", "sr"), default="en")
    add("serve", cmd_serve, "run the MCP server (stdio)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.func(args)
    except (GitError, FileExistsError, KeyError, ValueError) as exc:
        print(f"splet: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
