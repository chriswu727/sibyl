"""Smoke-check the built MCP server over a real stdio client session."""
from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


QUESTION = "In what year was the company that created CUDA founded?"
EXPECTED_KEYLESS_TOOLS = {
    "gather_evidence",
    "gather_bundle",
    "gather_sources",
    "quick_search",
    "read_url",
}


async def check(command: str) -> None:
    server = StdioServerParameters(command=command, args=["--profile", "keyless"])
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            if names != EXPECTED_KEYLESS_TOOLS:
                raise RuntimeError(
                    f"unexpected keyless tools: {sorted(names)!r}"
                )
            result = await session.call_tool(
                "gather_evidence",
                {"question": QUESTION, "max_steps": 2},
            )
            structured = result.structuredContent or {}
            if result.isError:
                raise RuntimeError("gather_evidence returned a tool error")
            if structured.get("status") != "active":
                raise RuntimeError(f"unexpected loop status: {structured.get('status')!r}")
            if structured.get("next_action") != "decompose_query":
                raise RuntimeError(
                    f"unexpected loop action: {structured.get('next_action')!r}"
                )
            if structured.get("current_step") is not None:
                raise RuntimeError("dependent question triggered an unnecessary retrieval")
            diagnostics = structured.get("diagnostics", {})
            if diagnostics.get("remaining_steps") != 2:
                raise RuntimeError(f"unexpected loop diagnostics: {diagnostics!r}")


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "sibyl-mcp"
    try:
        asyncio.run(check(command))
    except Exception as exc:
        print(f"MCP stdio smoke check failed: {exc}", file=sys.stderr)
        return 1
    print("MCP stdio smoke check passed with five keyless tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
