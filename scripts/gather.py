#!/usr/bin/env python
"""Thin CLI over the gather_sources MCP tool — keyless retrieval, no synthesis.

    python scripts/gather.py "your search query" [max_sources]

Prints the numbered [Source N] full-text blocks so a caller (you) can reason
over them. No API key required.
"""
import asyncio
import sys

from sibyl.mcp_server import gather_sources


async def main():
    if len(sys.argv) < 2:
        print("usage: gather.py <query> [max_sources]")
        return
    query = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    fn = gather_sources.fn if hasattr(gather_sources, "fn") else gather_sources
    print(await fn(query, max_sources=n))


if __name__ == "__main__":
    asyncio.run(main())
