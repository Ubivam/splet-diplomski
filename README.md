# splet

Multilayer knowledge graph of a software project. One set of nodes (files and
directories), four edge layers:

| Layer | Source |
|---|---|
| hierarchy | directory tree, test ↔ module pairing by name |
| structure | Python imports from the AST, JS/TS relative imports |
| evolution | files changed together in git history, with rename tracking and a one-year half-life |
| docs | Markdown links, Sphinx roles/directives, inline code mentions |

Files are ranked by personalized PageRank over a random walk that mixes the
layers. Default layer weights (hierarchy 0.3, structure 0.0, evolution 0.7,
docs 0.0) were chosen by leave-one-project-out selection on ten Python
projects, as part of the bachelor thesis *A Multilayer Knowledge Graph Based
on Dependencies* (ETF, University of Belgrade, 2026).

## Use

```sh
uv tool install --editable .
splet build   --repo ~/project
splet related src/pkg/module.py --repo ~/project
splet hidden  --repo ~/project          # co-change without a short import path
splet obsidian --repo ~/project --out ~/project-vault --lang sr
splet serve   --repo ~/project          # MCP server (stdio)
```

## Connect to an AI tool

```sh
scripts/povezi.sh claude-code    ~/project
scripts/povezi.sh claude-desktop ~/project
scripts/povezi.sh cursor         ~/project
scripts/povezi.sh json           ~/project   # config for any MCP client
```

MCP tools: `related_files`, `explain_file`, `hidden_dependencies`,
`communities`, `connection_path`, `project_report`, `rebuild_graph`,
`export_obsidian`.

## How it works

```
files ──► scan, pyindex ──► hierarchy, structure, docs ─┐
git log -M ──► history (renames, decay) ──► evolution ──┴─► graph.json ──► PageRank ──► CLI / MCP / Obsidian
```

The graph is cached in `.splet/graph.json` and rebuilt when `HEAD` moves or
any scanned file changes (size or modification time).

## Limitations

- Nodes are files. Ranked files are often large, so an agent that reads them
  whole spends many tokens; function-level nodes are future work.
- Import resolution covers Python (including `src/` layouts and PEP 420
  namespace packages inside packages) and relative JS/TS imports only.
- History is only as good as the commits: tangled commits create spurious
  links, and new files have no history (the walk then uses the other layers).

## Development

```sh
uv sync
uv run pytest --cov=splet        # 92 tests
uv run ruff check src tests      # lint
uv run ruff format --check src tests
uv run mypy                      # strict type checking
uv run python scripts/proba_mcp.py ~/project   # real MCP client against the server
```

Continuous integration (`.github/workflows/ci.yml`) runs the same checks on
Python 3.12 and 3.13.

## License

MIT, see `LICENSE`.
