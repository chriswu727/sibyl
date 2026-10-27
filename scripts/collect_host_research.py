"""Collect a frozen live host-research suite over real MCP; no model API calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    seen = set()
    if not 1 <= len(cases) <= 20:
        raise ValueError("Use 1-20 bounded research cases")
    for case in cases:
        identifier = case["id"]
        if not re.fullmatch(r"[a-z0-9_]{1,64}", identifier) or identifier in seen:
            raise ValueError("Case IDs must be unique safe names")
        seen.add(identifier)
        if not 1 <= len(case["queries"]) <= 4:
            raise ValueError("Use 1-4 initial queries per case")
    return cases


async def collect(cases: list[dict], output: Path, executable: str) -> None:
    parameters = StdioServerParameters(
        command=executable,
        env={"SIBYL_OUTPUT_ROOT": str(output), "SIBYL_SEARCH_PROVIDER": "keyless"},
        cwd=str(output),
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for case in cases:
                path = output / f"{case['id']}.json"
                if path.exists():
                    raise ValueError(f"Result already exists: {path.name}")
                trace = []
                record = {"case": case, "trace": trace, "job": None}

                async def call(name: str, **arguments):
                    result = await session.call_tool(name, arguments)
                    data = result.structuredContent or result.model_dump(mode="json")
                    trace.append(
                        {
                            "tool": name,
                            "arguments": arguments,
                            "is_error": bool(result.isError),
                        }
                    )
                    if result.isError:
                        record["error"] = data
                    else:
                        record["job"] = data
                    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                    if result.isError:
                        raise RuntimeError(f"{name} failed: {data}")
                    return data

                job = await call(
                    "start_research",
                    question=case["question"],
                    max_steps=6,
                    source_policy=case["policy"],
                    preferred_domains=case["domains"],
                )
                for index, query in enumerate(case["queries"]):
                    job = await call(
                        "continue_research",
                        job_id=job["job_id"],
                        query=query,
                        question_id="Q0",
                        purpose="explore" if index == 0 else "counterevidence",
                        rationale="Establish primary evidence."
                        if index == 0
                        else "Investigate scope, exceptions, and limits before drawing conclusions.",
                    )
                # Mutations return compact briefs; record the full checkpoint view.
                job = await call("get_research", job_id=job["job_id"], view="full")
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "job_id": job["job_id"],
                            "attempts": len(job["attempts"]),
                            "sources": len(job["sources"]),
                            "urls": [s["url"] for s in job["sources"]],
                        }
                    ),
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evals/host_research_cases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New private directory for checkpoints and full evidence",
    )
    parser.add_argument(
        "--executable", default=str(Path(sys.executable).with_name("sibyl-mcp"))
    )
    args = parser.parse_args()
    cases = load_cases(args.cases)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    asyncio.run(
        collect(cases, args.output.resolve(), str(Path(args.executable).resolve()))
    )


if __name__ == "__main__":
    main()
