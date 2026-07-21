#!/usr/bin/env python
"""Evaluate the fixed host-planned multi-step evidence workflows."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from sibyl import __version__
from sibyl.evidence_loop_eval import EvidenceLoopEvalCase, evaluate_evidence_loops


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "evals" / "gold" / "evidence_loop_4.jsonl"


def load_cases(path: Path) -> list[EvidenceLoopEvalCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cases.append(
            EvidenceLoopEvalCase(
                case_id=item["id"],
                question=item["question"],
                gold=item["gold"],
                aliases=item.get("aliases", []),
                queries=item["queries"],
            )
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-sources", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = asyncio.run(
        evaluate_evidence_loops(
            load_cases(args.dataset),
            concurrency=args.concurrency,
            max_sources=args.max_sources,
        )
    )
    passed = (
        result.decomposition_rate == 1.0
        and result.plan_execution_rate == 1.0
        and result.ready_rate == 1.0
        and result.answer_coverage == 1.0
        and result.pass_rate == 1.0
    )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sibyl_version": __version__,
        "method": "fixed_host_plans_model_free_retrieval_only",
        "method_limitations": "Does not evaluate host query planning or final synthesis; single repeat.",
        "configured_search_provider": os.environ.get(
            "SIBYL_SEARCH_PROVIDER", "keyless"
        ).strip().lower(),
        "repeats": 1,
        "dataset": str(args.dataset.resolve().relative_to(ROOT)),
        **result.to_dict(),
        "thresholds": {
            "min_decomposition_rate": 1.0,
            "min_plan_execution_rate": 1.0,
            "min_ready_rate": 1.0,
            "min_answer_coverage": 1.0,
            "min_pass_rate": 1.0,
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
