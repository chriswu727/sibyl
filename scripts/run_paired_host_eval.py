#!/usr/bin/env python
"""Run the frozen paired host evaluation through the Claude Code CLI.

Each (task, condition) pair runs once in a fresh working directory with only
the tools its condition names, no shell or file access, a dollar and time cap,
and the same model and effort. The raw stream, final answer and CLI-reported
usage are retained per run. See docs/LAUNCH_READINESS.md for the protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED = (
    "\n\nResearch this with web sources. Start with a direct answer. Then list "
    "each material claim on its own line, followed by the URL or URLs that "
    "support it. If the evidence does not establish an answer, or the question "
    "rests on a false premise, say so instead of guessing. Use at most eight "
    "source retrievals (searches or page reads). Do not run shell commands or "
    "read local files."
)
CONDITIONS = {
    "native": {
        "suffix": "",
        "tools": ["WebSearch", "WebFetch"],
        "mcp": False,
    },
    "sibyl": {
        "suffix": (
            " Use the Sibyl tools to manage the research: start a job, read the "
            "sources you find into it, record evidence-linked claims with "
            "assess_research, and follow its suggested searches."
        ),
        "tools": ["WebSearch", "WebFetch"],
        "mcp": True,
    },
    # A managed multi-agent research workflow without Sibyl, as a comparator.
    "deep_research": {
        "suffix": " Use the anthropic-skills:deep-research skill for this research.",
        "tools": ["WebSearch", "WebFetch", "Agent", "Skill"],
        "mcp": False,
        "skills": True,
    },
}


def run_one(case, condition, args, out_dir: Path) -> dict:
    spec = CONDITIONS[condition]
    record_path = out_dir / condition / f"{case['id']}.json"
    if record_path.exists():
        return json.loads(record_path.read_text(encoding="utf-8"))
    record_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"paired-{condition}-{case['id']}-"))
    command = [
        args.claude,
        "-p",
        case["question"] + SHARED + spec["suffix"],
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        args.model,
        "--effort",
        args.effort,
        "--max-budget-usd",
        str(args.max_budget_usd),
        "--strict-mcp-config",
        *(
            []
            if spec.get("skills")
            else ["--setting-sources", "", "--disable-slash-commands"]
        ),
        "--tools",
        ",".join(spec["tools"]),
        "--allowedTools",
        " ".join(spec["tools"] + (["mcp__sibyl"] if spec["mcp"] else [])),
    ]
    if spec["mcp"]:
        config = workdir / "mcp.json"
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "sibyl": {
                            "command": args.sibyl_mcp,
                            "env": {"SIBYL_OUTPUT_ROOT": str(workdir / "sibyl")},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        command += ["--mcp-config", str(config)]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env={**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
        )
        stream, timed_out = completed.stdout, False
    except subprocess.TimeoutExpired as exc:
        stream = (
            exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        )
        timed_out = True
    elapsed = time.monotonic() - started
    events = [json.loads(line) for line in stream.splitlines() if line.startswith("{")]
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    tool_calls: dict[str, int] = {}
    for event in events:
        for block in (event.get("message") or {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_calls[block["name"]] = tool_calls.get(block["name"], 0) + 1
    final = result.get("result", "")
    raw_path = record_path.with_suffix(".stream.jsonl")
    raw_path.write_text(stream, encoding="utf-8")
    record = {
        "case_id": case["id"],
        "condition": condition,
        "elapsed_seconds": round(elapsed, 1),
        "timed_out": timed_out,
        "subtype": result.get("subtype"),
        "is_error": result.get("is_error"),
        "num_turns": result.get("num_turns"),
        "cli_reported_cost_usd": result.get("total_cost_usd"),
        "usage": result.get("usage"),
        "model_usage": result.get("modelUsage"),
        "tool_calls": tool_calls,
        "final_answer": final,
        "final_answer_sha256": hashlib.sha256(final.encode()).hexdigest(),
        "raw_stream_sha256": hashlib.sha256(stream.encode()).hexdigest(),
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
    shutil.rmtree(workdir, ignore_errors=True)
    print(
        f"{condition}/{case['id']}: {record['subtype']} "
        f"${record['cli_reported_cost_usd']} {record['elapsed_seconds']}s",
        file=sys.stderr,
        flush=True,
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "evals" / "paired_host_cases.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-budget-usd", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--claude", default=shutil.which("claude") or "claude")
    parser.add_argument("--sibyl-mcp", default=shutil.which("sibyl-mcp") or "sibyl-mcp")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.only:
        cases = [case for case in cases if case["id"] in args.only]
    jobs = [(case, condition) for case in cases for condition in args.conditions]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        records = list(
            pool.map(lambda job: run_one(job[0], job[1], args, args.output), jobs)
        )
    total = sum(r["cli_reported_cost_usd"] or 0 for r in records)
    print(f"runs={len(records)} total_cost_usd={total:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
