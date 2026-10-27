"""Offline MCP fixture process; never installed as a production entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from sibyl.config import Config
from sibyl.mcp_server import create_server
from sibyl.runtime import SibylRuntime
from sibyl.scraper import WebPage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true")
    args = parser.parse_args()
    fixture = (
        Path(__file__).resolve().parents[1] / "docs/examples/sqlite-wal/sources.json"
    )
    sources = {item["url"]: item for item in json.loads(fixture.read_text())["sources"]}

    async def read(url: str, **kwargs) -> WebPage:
        item = sources[url]
        return WebPage(url, item["title"], item["text"])

    runtime = SibylRuntime(config=Config())
    server = create_server(runtime, agent_mode=args.agent)
    with (
        patch("sibyl.research_sources.scrape_url", read),
        patch(
            "httpx.AsyncClient.send",
            AsyncMock(side_effect=AssertionError("Offline replay forbids HTTP")),
        ),
    ):
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
