"""MCP composition root and command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from mcp.server.fastmcp import FastMCP

from . import mcp_tools
from .config import Config
from .mcp_tools import _format_report as _format_report
from .mcp_tools import (
    assess_research,
    cancel_research,
    complete_research,
    continue_research,
    gather_bundle,
    gather_evidence,
    gather_sources,
    get_research,
    quick_search,
    read_url,
    save_report,
    start_research,
)
from .runtime import RuntimePolicy, SibylRuntime, default_runtime
from .storage import LEGACY_OUTPUT_ROOT

__all__ = [
    "create_server",
    "main",
    "mcp",
    "assess_research",
    "cancel_research",
    "complete_research",
    "continue_research",
    "gather_bundle",
    "gather_evidence",
    "gather_sources",
    "get_research",
    "quick_search",
    "read_url",
    "save_report",
    "start_research",
]


def create_server(
    runtime: SibylRuntime | None = None, *, agent_mode: bool = False
) -> FastMCP:
    """Create an isolated server without sharing reports, loops, or cache entries."""
    server = mcp_tools._catalog.create_server(
        runtime or SibylRuntime(),
        mcp_tools._AGENT_INSTRUCTIONS if agent_mode else mcp_tools._INSTRUCTIONS,
        agent_mode=agent_mode,
    )
    if agent_mode:
        for name in mcp_tools._catalog.names(agent_mode=True):
            if name not in mcp_tools._AGENT_TOOLS:
                server.remove_tool(name)
    return server


mcp = create_server(default_runtime)


def _mcp_profile(config: Config | None = None) -> str:
    requested = os.environ.get("SIBYL_MCP_PROFILE", "agent").strip().lower()
    if requested not in {"auto", "agent", "keyless"}:
        raise RuntimeError(
            "Use the agent (default) or keyless compatibility profile. "
            "Model-backed and finance tools have been removed."
        )
    return "agent" if requested == "auto" else requested


async def configure_tool_profile(server: FastMCP, runtime: SibylRuntime) -> str:
    profile = _mcp_profile()
    enabled = mcp_tools._AGENT_TOOLS if profile == "agent" else mcp_tools._KEYLESS_TOOLS
    for tool in await server.list_tools():
        if tool.name not in enabled:
            server.remove_tool(tool.name)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Sibyl MCP server")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument(
        "--monitor-port",
        type=int,
        help="Authenticated loopback monitoring port (0 selects a free port)",
    )
    parser.add_argument("--log-level", choices=["WARNING", "INFO"], default="WARNING")
    parser.add_argument("--profile", choices=["auto", "agent", "keyless"])
    args = parser.parse_args()
    if args.version:
        from . import __version__

        print(__version__)
        return
    if args.profile:
        os.environ["SIBYL_MCP_PROFILE"] = args.profile
    operational_logger = logging.getLogger("sibyl.runtime")
    operational_logger.setLevel(args.log_level)
    operational_logger.addHandler(logging.StreamHandler())
    operational_logger.propagate = False
    policy = RuntimePolicy(monitor_port=args.monitor_port)
    legacy = LEGACY_OUTPUT_ROOT.absolute()
    if (
        not os.environ.get("SIBYL_OUTPUT_ROOT")
        and (legacy / ".research" / "jobs.sqlite3").is_file()
        and legacy != policy.output_root
    ):
        operational_logger.warning(
            "Sibyl now stores research in %s. Research saved under %s was not "
            "moved; set SIBYL_OUTPUT_ROOT=%s to keep using it.",
            policy.output_root,
            legacy,
            legacy,
        )
    runtime = SibylRuntime(policy)
    server = create_server(runtime, agent_mode=_mcp_profile(runtime.config) == "agent")
    profile = asyncio.run(configure_tool_profile(server, runtime))
    if args.list_tools:
        print(f"profile: {profile}")
        for tool in asyncio.run(server.list_tools()):
            print(tool.name)
        return
    server.run()


if __name__ == "__main__":
    main()
