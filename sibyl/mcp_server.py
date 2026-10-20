"""MCP composition root and command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
from importlib.util import find_spec

from mcp.server.fastmcp import FastMCP

from . import mcp_tools
from .config import Config
from .mcp_tools import _format_report as _format_report
from .mcp_tools import (
    analyze,
    chart,
    cancel_research,
    complete_research,
    continue_research,
    compare,
    fetch_market_data,
    gather_bundle,
    gather_evidence,
    gather_sources,
    get_research,
    quick_search,
    read_url,
    research,
    save_report,
    start_research,
    swot,
    timeline,
    trends,
)
from .runtime import SibylRuntime, default_runtime

__all__ = [
    "create_server",
    "main",
    "mcp",
    "analyze",
    "chart",
    "cancel_research",
    "complete_research",
    "continue_research",
    "compare",
    "fetch_market_data",
    "gather_bundle",
    "gather_evidence",
    "gather_sources",
    "get_research",
    "quick_search",
    "read_url",
    "research",
    "save_report",
    "start_research",
    "swot",
    "timeline",
    "trends",
]


def create_server(runtime: SibylRuntime | None = None) -> FastMCP:
    """Create an isolated server without sharing reports, loops, or cache entries."""
    return mcp_tools._catalog.create_server(
        runtime or SibylRuntime(), mcp_tools._INSTRUCTIONS
    )


mcp = create_server(default_runtime)


def _mcp_profile(config: Config) -> str:
    requested = os.environ.get("SIBYL_MCP_PROFILE", "auto").strip().lower()
    if requested not in {"auto", "keyless", "report", "finance", "full"}:
        raise RuntimeError(
            "SIBYL_MCP_PROFILE must be auto, keyless, report, finance, or full"
        )
    if requested == "auto":
        installed = all(find_spec(name) for name in ("litellm", "fpdf"))
        return "report" if config.has_llm_credentials() and installed else "keyless"
    return requested


async def configure_tool_profile(server: FastMCP, runtime: SibylRuntime) -> str:
    profile = _mcp_profile(runtime.config)
    enabled = set(mcp_tools._KEYLESS_TOOLS)
    for profiles, modules, extra, names in (
        ({"report", "full"}, ("litellm", "fpdf"), "report", mcp_tools._REPORT_TOOLS),
        (
            {"finance", "full"},
            ("matplotlib", "pandas", "pytrends", "yfinance"),
            "finance",
            mcp_tools._FINANCE_TOOLS,
        ),
    ):
        if profile in profiles:
            missing = [name for name in modules if find_spec(name) is None]
            if missing:
                raise RuntimeError(
                    f"Install sibyl-research[{extra}]; missing: {', '.join(missing)}"
                )
            if extra == "report" and not runtime.config.has_llm_credentials():
                raise RuntimeError(
                    "Report tools require an LLM provider key or local/API-base backend."
                )
            enabled.update(names)
    for tool in await server.list_tools():
        if tool.name not in enabled:
            server.remove_tool(tool.name)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Sibyl MCP server")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument(
        "--profile", choices=["auto", "keyless", "report", "finance", "full"]
    )
    args = parser.parse_args()
    if args.version:
        from . import __version__

        print(__version__)
        return
    if args.profile:
        os.environ["SIBYL_MCP_PROFILE"] = args.profile
    runtime = SibylRuntime()
    server = create_server(runtime)
    profile = asyncio.run(configure_tool_profile(server, runtime))
    if args.list_tools:
        print(f"profile: {profile}")
        for tool in asyncio.run(server.list_tools()):
            print(tool.name)
        return
    server.run()


if __name__ == "__main__":
    main()
