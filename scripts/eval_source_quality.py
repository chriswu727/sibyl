#!/usr/bin/env python
"""Run the fixed contextual source-quality baseline evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sibyl.source_quality_eval import (
    evaluate_source_quality_cases,
    load_source_quality_cases,
    source_type_prior_scores,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "evals" / "source_quality_cases.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--min-selective-accuracy", type=float, default=0.8)
    parser.add_argument("--min-overall-accuracy", type=float, default=0.625)
    args = parser.parse_args()

    try:
        cases = load_source_quality_cases(args.dataset)
        result = evaluate_source_quality_cases(cases, source_type_prior_scores)
    except Exception as exc:
        print(json.dumps({"baseline": "source_type_prior", "error": str(exc)}))
        return 2

    passed = (
        result.coverage >= args.min_coverage
        and result.selective_accuracy >= args.min_selective_accuracy
        and result.overall_accuracy >= args.min_overall_accuracy
    )
    output = {
        "dataset": str(args.dataset),
        "baseline": "source_type_prior",
        **result.to_dict(),
        "thresholds": {
            "min_coverage": args.min_coverage,
            "min_selective_accuracy": args.min_selective_accuracy,
            "min_overall_accuracy": args.min_overall_accuracy,
        },
        "passed": passed,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
