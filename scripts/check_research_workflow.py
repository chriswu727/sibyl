"""Replay a complete host research workflow over real MCP stdio, without network or models."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

QUESTION = "Does SQLite WAL eliminate write contention?"
WAL = "https://sqlite.org/wal.html"
ISOLATION = "https://sqlite.org/isolation.html"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


async def check(
    directory: Path, include_pdf: bool = False, agent: bool = False
) -> dict:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            str(Path(__file__).with_name("fixture_mcp_server.py")),
            *(["--agent"] if agent else []),
        ],
        env={"SIBYL_OUTPUT_ROOT": str(directory)},
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if agent:
                tools = (await session.list_tools()).tools
                require(len(tools) == 8, "Agent profile did not expose eight tools")

            response_bytes = 0

            async def call(name: str, **arguments):
                nonlocal response_bytes
                response = await session.call_tool(name, arguments)
                require(not response.isError, f"{name} failed: {response.content}")
                response_bytes += len(
                    response.model_dump_json(exclude_none=True).encode()
                )
                return response.structuredContent or {}

            job = await call("start_research", question=QUESTION, max_steps=3)
            job_id = job["job_id"]
            job = await call(
                "continue_research",
                job_id=job_id,
                url=WAL,
                question_id="Q0",
                rationale="Establish documented concurrency benefits and limitations.",
            )
            wal_passage = job["sources"][0]["passage_ids"][0]
            passage = await call("get_research", job_id=job_id, passage_id=wal_passage)
            require(passage["passage"]["text"], "Exact WAL evidence unavailable")
            require(passage["retrieved_at"], "Passage lost provenance")
            if agent:
                require(job["view"] == "brief", "Agent mutation did not return a brief")
                require("evidence" not in job, "Agent brief repeated raw evidence")
            job = await call(
                "assess_research",
                job_id=job_id,
                revision=job["revision"],
                question_id="Q0",
                status="unresolved",
                supporting_passage_ids=[wal_passage],
                conclusion="WAL improves reader/writer concurrency, but writer concurrency remains unestablished.",
                counterevidence_note="Investigate whether multiple writers can commit concurrently.",
                follow_up_queries=["SQLite simultaneous writers serialization"],
            )
            require(
                not job["research_plan"]["ready_to_complete"],
                "Unresolved evidence became ready",
            )
            require(job["research_plan"]["suggested_queries"], "Missing gap follow-up")
            first_revision = job["revision"]
            job = await call(
                "continue_research",
                job_id=job_id,
                url=ISOLATION,
                question_id="Q0",
                purpose="counterevidence",
                rationale="Test the stronger claim that WAL allows simultaneous writers.",
            )
            require(
                (job["latest_attempt"] if agent else job["attempts"][-1])["prior_gaps"],
                "Follow-up lost its research gaps",
            )
            require(
                (job["latest_attempt"] if agent else job["attempts"][-1])[
                    "based_on_revision"
                ]
                == first_revision,
                "Follow-up revision lost",
            )
            require(
                not job["research_plan"]["ready_to_complete"],
                "New evidence did not require reassessment",
            )
            passages = {
                source["url"]: source["passage_ids"][0] for source in job["sources"]
            }
            for passage_id in passages.values():
                evidence = await call(
                    "get_research", job_id=job_id, passage_id=passage_id
                )
                require(
                    hashlib.sha256(evidence["passage"]["text"].encode()).hexdigest()
                    == evidence["passage"]["content_hash"],
                    "Passage read changed exact text",
                )
            catalog = await call(
                "get_research",
                job_id=job_id,
                view="sources",
                revision=job["revision"],
                limit=1,
            )
            require(
                catalog["total"] == 2 and catalog["next_offset"] == 1,
                "Source pagination contract failed",
            )
            page2 = await call(
                "get_research",
                job_id=job_id,
                view="sources",
                revision=job["revision"],
                offset=catalog["next_offset"],
                limit=1,
            )
            require(page2["next_offset"] is None, "Source catalog did not end")
            job = await call(
                "assess_research",
                job_id=job_id,
                revision=job["revision"],
                question_id="Q0",
                status="supported",
                supporting_passage_ids=list(passages.values()),
                conclusion="No. WAL improves reader/writer overlap, but SQLite serializes writers.",
                counterevidence_note="The isolation documentation explicitly limits the database to one writer; the WAL page also documents SQLITE_BUSY.",
            )
            require(
                job["research_plan"]["ready_to_complete"],
                "Reviewed investigation remained incomplete",
            )
            require(job["report"] is None, "An agent ledger required a prose report")
            citations = {source["url"]: source["citation"] for source in job["sources"]}
            summary = f"WAL improves reader/writer concurrency but does not eliminate writer serialization. {citations[WAL]} {citations[ISOLATION]}"
            findings = [
                f"Readers and a writer can proceed concurrently in WAL mode. {citations[WAL]}",
                f"Only one writer can write at a time; other writers must take turns. {citations[ISOLATION]}",
                f"Applications must still handle SQLITE_BUSY. {citations[WAL]}",
            ]
            stale = await session.call_tool(
                "complete_research",
                {
                    "job_id": job_id,
                    "revision": first_revision,
                    "summary": summary,
                    "key_findings": findings,
                },
            )
            require(bool(stale.isError), "Stale completion was accepted")
            job = await call(
                "complete_research",
                job_id=job_id,
                revision=job["revision"],
                summary=summary,
                key_findings=findings,
                analysis="Reader/writer concurrency and writer/writer concurrency are different properties. A concurrency benefit does not establish freedom from contention.",
                limitations=[
                    "Deterministic replay of selected official excerpts recorded on 2026-10-21; no fresh retrieval or model synthesis.",
                    "Both pages have one publisher. This is not independent corroboration or a workload benchmark.",
                ],
            )
            require(
                job["report"]["status"] == "ok",
                "Complete investigation was not exported as ok",
            )
            require(
                job["report"]["verification_scope"] == "citation_targets_only",
                "Verification scope overstated",
            )
            for format in ["md", "json", *(["pdf"] if include_pdf else [])]:
                await call("save_report", job_id=job_id, format=format)
                require(
                    len(list(directory.glob(f"*.{format}"))) == 1,
                    f"Missing {format} artifact",
                )
            mutate = await session.call_tool(
                "continue_research", {"job_id": job_id, "url": WAL}
            )
            require(bool(mutate.isError), "Completed report was mutable")
            job = await call("get_research", job_id=job_id, view="full")
            print(
                f"{'agent' if agent else 'compatible'} successful replay response bytes: {response_bytes}"
            )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            restored = await session.call_tool(
                "get_research", {"job_id": job_id, "view": "full"}
            )
            require(not restored.isError, "Checkpoint did not survive process restart")
            require(
                restored.structuredContent["report"] == job["report"],
                "Restart changed finalized report",
            )
    exported = json.loads(next(directory.glob("*.json")).read_text())
    require(exported == job["report"], "JSON export changed the report")
    for source in exported["sources"]:
        for passage in source["evidence"]:
            require(
                hashlib.sha256(passage["text"].encode()).hexdigest()
                == passage["content_hash"],
                "Passage hash mismatch",
            )
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory for replay artifacts; temporary by default",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also verify PDF export; requires report extra",
    )
    parser.add_argument(
        "--agent", action="store_true", help="Use the compact eight-tool agent profile"
    )
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        asyncio.run(check(args.output.resolve(), args.pdf, args.agent))
    else:
        with tempfile.TemporaryDirectory(prefix="sibyl-workflow-") as directory:
            asyncio.run(check(Path(directory), args.pdf, args.agent))
    print(
        "Host research replay passed: two rounds, gap review, counterevidence, citations, exports, restart, and immutability; no network/model calls."
    )


if __name__ == "__main__":
    main()
