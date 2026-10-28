"""MCP composition root and command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from mcp.server.fastmcp import FastMCP

from . import mcp_tools
from .mcp_tools import _format_report as _format_report
from .mcp_tools import (
    assess_research,
    cancel_research,
    complete_research,
    continue_research,
    gather_bundle,
    get_research,
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
    "get_research",
    "save_report",
    "start_research",
]


def create_server(runtime: SibylRuntime | None = None) -> FastMCP:
    """Create an isolated server without sharing reports or cache entries."""
    return mcp_tools._catalog.create_server(
        runtime or SibylRuntime(), mcp_tools._INSTRUCTIONS
    )


mcp = create_server(default_runtime)


def _check_profile(requested: str | None) -> None:
    """Reject the removed keyless profile instead of silently changing tools."""
    value = (requested or "agent").strip().lower()
    if value not in {"auto", "agent"}:
        raise SystemExit(
            f"The {value!r} MCP profile has been removed; Sibyl serves one tool set. "
            "Drop --profile/SIBYL_MCP_PROFILE. See docs/MIGRATION.md."
        )


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
    parser.add_argument("--profile", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.version:
        from . import __version__

        print(__version__)
        return
    _check_profile(args.profile or os.environ.get("SIBYL_MCP_PROFILE"))
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
    server = create_server(runtime)
    if args.list_tools:
        for tool in asyncio.run(server.list_tools()):
            print(tool.name)
        return
    server.run()


if __name__ == "__main__":
    main()
