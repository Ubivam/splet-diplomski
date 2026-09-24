"""Proba MCP servera pravim klijentom: pokreće `splet serve` kao zaseban proces.

uv run python scripts/proba_mcp.py <putanja-projekta>
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SPLET = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "splet"
SEED = "src/flask/sessions.py"


def first_text(content: list) -> str:
    return next((c.text for c in content if getattr(c, "type", "") == "text"), "")


async def main(repo: str) -> None:
    params = StdioServerParameters(command=str(SPLET), args=["serve", "--repo", repo])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        print("server:", init.server_info.name)
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])
        result = await session.call_tool(
            "related_files", {"paths": [SEED], "limit": 4, "code_only": True}
        )
        print(first_text(result.content)[:600])
        # Pogrešno napisana putanja mora da vrati grešku sa predlogom, a ne da obori server.
        result = await session.call_tool("related_files", {"paths": ["src/flask/sesions.py"]})
        print("error case:", result.is_error, first_text(result.content)[:150])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
