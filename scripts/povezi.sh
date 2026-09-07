#!/usr/bin/env bash
# Instalira splet i povezuje njegov MCP server sa AI alatom.
#
#   scripts/povezi.sh claude-code    <projekat>   # Claude Code (CLI)
#   scripts/povezi.sh claude-desktop <projekat>   # Claude Desktop (macOS/Windows/Linux)
#   scripts/povezi.sh cursor         <projekat>   # Cursor (.cursor/mcp.json u projektu)
#   scripts/povezi.sh json           <projekat>   # ispisuje konfiguraciju za bilo koji MCP klijent
#
# Postojeće konfiguracije se ne brišu: dodaje se samo unos "splet", a pre
# izmene se pravi rezervna kopija (*.bak).
set -euo pipefail

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[[ $# -eq 2 ]] || usage

CLIENT=$1
PROJECT=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")/.." && pwd)

if ! command -v uv >/dev/null; then
  echo "Potreban je uv: https://docs.astral.sh/uv/ (brew install uv)" >&2
  exit 1
fi

echo "→ instaliram splet iz $HERE"
uv tool install --quiet --force --editable "$HERE"
SPLET=$(uv tool dir --bin)/splet
[[ -x $SPLET ]] || { echo "splet nije pronađen posle instalacije ($SPLET)" >&2; exit 1; }

echo "→ gradim graf za $PROJECT"
"$SPLET" build --repo "$PROJECT"

# merge_json <datoteka> <ključ-sa-serverima>: dodaje unos "splet" u JSON konfiguraciju.
merge_json() {
  local file=$1 key=$2
  mkdir -p "$(dirname "$file")"
  [[ -f $file ]] && cp "$file" "$file.bak"
  SPLET="$SPLET" PROJECT="$PROJECT" python3 - "$file" "$key" <<'PY'
import json, os, sys
path, key = sys.argv[1], sys.argv[2]
try:
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
except FileNotFoundError:
    config = {}
config.setdefault(key, {})["splet"] = {
    "command": os.environ["SPLET"],
    "args": ["serve", "--repo", os.environ["PROJECT"]],
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(config, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
PY
  echo "✓ upisano u $file"
}

case $CLIENT in
  claude-code)
    command -v claude >/dev/null || { echo "Claude Code (claude) nije instaliran" >&2; exit 1; }
    claude mcp remove splet --scope user >/dev/null 2>&1 || true
    claude mcp add splet --scope user -- "$SPLET" serve --repo "$PROJECT"
    echo "✓ Claude Code: server 'splet' dodat (provera: claude mcp list)"
    ;;
  claude-desktop)
    case "$(uname -s)" in
      Darwin) CFG="$HOME/Library/Application Support/Claude/claude_desktop_config.json" ;;
      Linux)  CFG="${XDG_CONFIG_HOME:-$HOME/.config}/Claude/claude_desktop_config.json" ;;
      *)      CFG="$APPDATA/Claude/claude_desktop_config.json" ;;
    esac
    merge_json "$CFG" mcpServers
    echo "  Ponovo pokrenite Claude Desktop."
    ;;
  cursor)
    merge_json "$PROJECT/.cursor/mcp.json" mcpServers
    ;;
  json)
    printf '{\n  "mcpServers": {\n    "splet": {\n      "command": "%s",\n      "args": ["serve", "--repo", "%s"]\n    }\n  }\n}\n' "$SPLET" "$PROJECT"
    ;;
  *) usage ;;
esac
