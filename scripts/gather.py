#!/usr/bin/env python
"""Thin CLI over the sibyl.gather_sources Python API — keyless retrieval, no synthesis.

    python scripts/gather.py "your search query" [max_sources]

Prints the numbered [Source N] full-text blocks so a caller (you) can reason
over them. No API key required.
"""

import asyncio
import sys

from sibyl import gather_sources


async def main():
    if len(sys.argv) < 2:
        print("usage: gather.py <query> [max_sources]")
        return
    query = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(await gather_sources(query, max_sources=n))


if __name__ == "__main__":
    asyncio.run(main())
