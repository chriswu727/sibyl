"""Smoke-check the built MCP server over a real stdio client session."""
from __future__ import annotations

import asyncio
import argparse
import secrets
import socket
import sys
import tempfile

import httpx

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


QUESTION = "In what year was the company that created CUDA founded?"
EXPECTED_KEYLESS_TOOLS = {
    "start_research",
    "get_research",
    "continue_research",
    "complete_research",
    "assess_research",
    "cancel_research",
    "save_report",
    "gather_evidence",
    "gather_bundle",
    "gather_sources",
    "quick_search",
    "read_url",
}


async def check(command: str, monitor: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix="sibyl-stdio-") as directory:
        await check_in_directory(command, directory, monitor)


async def check_in_directory(command: str, directory: str, monitor: bool = False) -> None:
    args = ["--profile", "keyless"]
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
        command=command, args=args, env=env,
    )
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
            created = await session.call_tool("start_research", {"question": QUESTION})
            job = created.structuredContent or {}
            if created.isError or job.get("status") != "awaiting_input":
                raise RuntimeError("Could not create a keyless research job")
            loaded = await session.call_tool("get_research", {"job_id": job["job_id"]})
            if loaded.isError or (loaded.structuredContent or {}).get("job_id") != job["job_id"]:
                raise RuntimeError("Research job was not readable")
            deleted = await session.call_tool(
                "cancel_research", {"job_id": job["job_id"], "delete": True}
            )
            if deleted.isError or (deleted.structuredContent or {}).get("status") != "deleted":
                raise RuntimeError("Research checkpoint cleanup failed")
            if monitor:
                async with httpx.AsyncClient(trust_env=False, base_url=url) as client:
                    denied = await client.get("/status")
                    if denied.status_code != 401:
                        raise RuntimeError("Monitoring allowed an unauthenticated request")
                    headers = {"Authorization": f"Bearer {token}"}
                    status = await client.get("/status", headers=headers)
                    if status.status_code != 200 or QUESTION in status.text or token in status.text:
                        raise RuntimeError("Monitoring status failed or disclosed content")
                    if status.json()["tools"]["start_research"]["outcomes"]["completed"] != 1:
                        raise RuntimeError("MCP tool invocation did not reach monitoring")
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
    args = parser.parse_args()
    try:
        asyncio.run(check(args.command, args.monitor))
    except Exception as exc:
        print(f"MCP stdio smoke check failed: {exc}", file=sys.stderr)
        return 1
    print("MCP stdio smoke check passed with twelve keyless tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
