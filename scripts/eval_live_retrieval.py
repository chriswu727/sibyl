#!/usr/bin/env python
"""Run the model-free live launch gate against public web retrieval."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from sibyl import __version__
from sibyl.live_retrieval_eval import LiveRetrievalCase, evaluate_live_retrieval


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS = [
    ROOT / "evals" / "gold" / "simpleqa_real_30.jsonl",
    ROOT / "evals" / "gold" / "hard_24.jsonl",
    ROOT / "evals" / "gold" / "adversarial_12.jsonl",
]


def load_cases(paths, limit=0):
    cases = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            cases.append(
                LiveRetrievalCase(
                    case_id=item["id"],
                    question=item["question"],
                    gold=item["gold"],
                    aliases=item.get("aliases", []),
                    case_type=item.get("type", ""),
                )
            )
    return cases[:limit] if limit else cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-sources", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--min-answer-coverage", type=float, default=0.8)
    parser.add_argument("--min-trap-safe-rate", type=float, default=0.8)
    parser.add_argument("--min-stable-case-rate", type=float, default=0.9)
    parser.add_argument("--min-ready-bundle-rate", type=float, default=0.75)
    parser.add_argument("--max-p95-latency-ms", type=int, default=30000)
    args = parser.parse_args()

    datasets = args.dataset or DEFAULT_DATASETS
    cases = load_cases(datasets, args.limit)

    def report_progress(done, total, case_result):
        if args.quiet:
            return
        statuses = ",".join(run.status for run in case_result.runs)
        latencies = ",".join(str(run.latency_ms) for run in case_result.runs)
        print(
            f"[{done}/{total}] {case_result.case_id}: "
            f"status={statuses} latency_ms={latencies}",
            file=sys.stderr,
            flush=True,
        )

    try:
        result = asyncio.run(
            evaluate_live_retrieval(
                cases,
                repeats=args.repeats,
                concurrency=args.concurrency,
                max_sources=args.max_sources,
                progress=report_progress,
            )
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2

    passed = (
        result.answer_coverage >= args.min_answer_coverage
        and result.trap_safe_rate >= args.min_trap_safe_rate
        and result.stable_case_rate >= args.min_stable_case_rate
        and result.ready_bundle_rate >= args.min_ready_bundle_rate
        and result.p95_latency_ms <= args.max_p95_latency_ms
    )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sibyl_version": __version__,
        "datasets": [str(path) for path in datasets],
        "repeats": args.repeats,
        **result.to_dict(),
        "thresholds": {
            "min_answer_coverage": args.min_answer_coverage,
            "min_trap_safe_rate": args.min_trap_safe_rate,
            "min_stable_case_rate": args.min_stable_case_rate,
            "min_ready_bundle_rate": args.min_ready_bundle_rate,
            "max_p95_latency_ms": args.max_p95_latency_ms,
        },
        "passed": passed,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
