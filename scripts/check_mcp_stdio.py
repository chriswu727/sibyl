"""Smoke-check the built MCP server over a real stdio client session."""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import socket
import subprocess
import sys
import tempfile

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

QUESTION = "In what year was the company that created CUDA founded?"
EXPECTED_TOOLS = {
    "start_research",
    "get_research",
    "continue_research",
    "complete_research",
    "assess_research",
    "cancel_research",
    "save_report",
    "gather_bundle",
}


async def check(command: str, monitor: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix="sibyl-stdio-") as directory:
        check_removed_profile(command, directory)
        await check_in_directory(command, directory, monitor)


def check_removed_profile(command: str, directory: str) -> None:
    """The removed keyless profile must fail loudly, not serve a changed tool set."""
    base = {**os.environ, "SIBYL_OUTPUT_ROOT": directory}
    for args, env in (
        (["--profile", "keyless"], base),
        ([], {**base, "SIBYL_MCP_PROFILE": "keyless"}),
    ):
        result = subprocess.run(
            [command, *args],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 or "removed" not in result.stderr:
            raise RuntimeError("The removed keyless profile did not fail clearly")


async def check_in_directory(
    command: str, directory: str, monitor: bool = False
) -> None:
    args: list[str] = []
    env = {"SIBYL_OUTPUT_ROOT": directory}
    token = secrets.token_urlsafe(32)
    if monitor:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        args += ["--monitor-port", str(port), "--log-level", "INFO"]
        env["SIBYL_MONITOR_TOKEN"] = token
        url = f"http://127.0.0.1:{port}"
    server = StdioServerParameters(
        command=command,
        args=args,
        env=env,
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            if names != EXPECTED_TOOLS:
                raise RuntimeError(f"unexpected MCP tools: {sorted(names)!r}")
            undeclared = await session.call_tool(
                "start_research", {"question": QUESTION, "claim_contract_version": 1}
            )
            error = (undeclared.structuredContent or {}).get("error", {})
            if not undeclared.isError or error.get("code") != "invalid_parameters":
                raise RuntimeError("An undeclared argument was silently accepted")
            created = await session.call_tool("start_research", {"question": QUESTION})
            job = created.structuredContent or {}
            if created.isError or job.get("status") != "awaiting_input":
                raise RuntimeError("Could not create a research job")
            if job.get("view") != "brief" or "evidence" in job:
                raise RuntimeError("start_research did not return a compact brief")
            if job.get("claim_contract_version") != 2:
                raise RuntimeError("New job does not use evidence-linked claims")
            loaded = await session.call_tool(
                "get_research", {"job_id": job["job_id"], "view": "full"}
            )
            full = loaded.structuredContent or {}
            if loaded.isError or full.get("job_id") != job["job_id"]:
                raise RuntimeError("Research job was not readable")
            if "evidence" not in full or "model_calls" in full:
                raise RuntimeError("Full research view has an unexpected shape")
            deleted = await session.call_tool(
                "cancel_research", {"job_id": job["job_id"], "delete": True}
            )
            if (
                deleted.isError
                or (deleted.structuredContent or {}).get("status") != "deleted"
            ):
                raise RuntimeError("Research checkpoint cleanup failed")
            if monitor:
                async with httpx.AsyncClient(trust_env=False, base_url=url) as client:
                    denied = await client.get("/status")
                    if denied.status_code != 401:
                        raise RuntimeError(
                            "Monitoring allowed an unauthenticated request"
                        )
                    headers = {"Authorization": f"Bearer {token}"}
                    status = await client.get("/status", headers=headers)
                    if (
                        status.status_code != 200
                        or QUESTION in status.text
                        or token in status.text
                    ):
                        raise RuntimeError(
                            "Monitoring status failed or disclosed content"
                        )
                    if (
                        status.json()["tools"]["start_research"]["outcomes"][
                            "completed"
                        ]
                        != 1
                    ):
                        raise RuntimeError(
                            "MCP tool invocation did not reach monitoring"
                        )
                    metrics = await client.get("/metrics", headers=headers)
                    if "sibyl_store_healthy 1" not in metrics.text:
                        raise RuntimeError("Monitoring metrics unavailable")
    if monitor:
        async with httpx.AsyncClient(trust_env=False) as client:
            try:
                await client.get(url + "/healthz", timeout=1)
            except httpx.ConnectError:
                pass
            else:
                raise RuntimeError("Monitoring listener survived MCP shutdown")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="sibyl-mcp")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Accepted for compatibility; the eight-tool set is the only MCP surface",
    )
    args = parser.parse_args()
    try:
        asyncio.run(check(args.command, args.monitor))
    except Exception as exc:
        print(f"MCP stdio smoke check failed: {exc}", file=sys.stderr)
        return 1
    print(
        "MCP stdio smoke check passed with eight tools; the removed keyless "
        "profile is rejected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
