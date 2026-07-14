#!/usr/bin/env python
"""Run the fixed offline end-to-end retrieval pipeline regression set."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sibyl.retrieval_eval import load_retrieval_cases
from sibyl.retrieval_pipeline_eval import evaluate_pipeline_cases


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "evals" / "retrieval_ranker_cases.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ranker", choices=["lexical", "flashrank"], default="lexical")
    parser.add_argument("--min-top-source-accuracy", type=float, default=1.0)
    parser.add_argument("--min-usable-bundle-rate", type=float, default=1.0)
    parser.add_argument("--min-structure-valid-rate", type=float, default=1.0)
    parser.add_argument("--min-case-pass-rate", type=float, default=1.0)
    args = parser.parse_args()

    try:
        cases = load_retrieval_cases(args.dataset)
        result = asyncio.run(evaluate_pipeline_cases(cases, ranker=args.ranker))
    except Exception as exc:
        print(json.dumps({"ranker": args.ranker, "error": str(exc)}, ensure_ascii=False))
        return 2

    passed = (
        result.top_source_accuracy >= args.min_top_source_accuracy
        and result.usable_bundle_rate >= args.min_usable_bundle_rate
        and result.structure_valid_rate >= args.min_structure_valid_rate
        and result.case_pass_rate >= args.min_case_pass_rate
    )
    output = {
        "dataset": str(args.dataset),
        "ranker": args.ranker,
        **result.to_dict(),
        "thresholds": {
            "min_top_source_accuracy": args.min_top_source_accuracy,
            "min_usable_bundle_rate": args.min_usable_bundle_rate,
            "min_structure_valid_rate": args.min_structure_valid_rate,
            "min_case_pass_rate": args.min_case_pass_rate,
        },
        "passed": passed,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
