"""Proba MCP servera pravim klijentom: pokreće `splet serve` kao zaseban proces.

    uv run python scripts/proba_mcp.py <putanja-projekta>
"""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main(repo):
    params = StdioServerParameters(command=str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "splet"), args=["serve", "--repo", repo])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            print("server:", init.server_info.name, "| instructions:", (init.instructions or "")[:60])
            tools = await s.list_tools()
            print("tools:", [t.name for t in tools.tools])
            res = await s.call_tool("related_files", {"paths": ["src/flask/sessions.py"], "limit": 4, "code_only": True})
            for c in res.content: print(c.text[:600])
            res = await s.call_tool("related_files", {"paths": ["src/flask/sesions.py"]})
            print("error case:", res.is_error, res.content[0].text[:150])
asyncio.run(main(sys.argv[1]))
