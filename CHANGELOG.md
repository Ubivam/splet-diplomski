# Changelog

## 0.1.0 — 2026-09-24

First release, accompanying the bachelor thesis *Вишеслојни граф знања
софтверског пројекта заснован на структурним и еволуционим зависностима*
(ETF, University of Belgrade).

### Graph
- Four layers over the files of a project: directory hierarchy with
  test-to-module pairing, Python and JS/TS imports, git co-change history,
  documentation links and mentions.
- History follows renames (`git log -M`) and discounts old commits with a
  one-year half-life; commits touching more than 30 files are ignored.
- Python module resolution handles the `src/` layout, relative imports,
  submodule imports and PEP 420 namespace packages nested in packages.
- Leiden communities on the combined file graph.

### Ranking
- Personalized PageRank over a multiplex random walk that renormalises layer
  weights per node, so files without history fall back to the other layers.
- Default weights (hierarchy 0.3, history 0.7) chosen by leave-one-project-out
  selection on ten Python projects; static weights when there is no history.

### Interfaces
- CLI: `build`, `related`, `explain`, `hidden`, `report`, `obsidian`, `serve`.
- MCP server (stdio) with eight tools; read-only tools are annotated. A wrong
  request (unknown path, no connection, occupied directory) returns a tool
  error with a readable message instead of a generic server failure.
- Obsidian vault export with community colours and Serbian or English notes.
- `scripts/povezi.sh` registers the server with Claude Code, Claude Desktop
  and Cursor without overwriting existing configuration.

### Quality
- 83 tests (94 % line coverage), ruff, strict mypy, CI on Python 3.12 and 3.13.
- Cache invalidated by revision and by a fingerprint of the working tree.
