#!/usr/bin/env python
"""Run the fixed offline retrieval-ranker regression set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sibyl.ranking import flashrank_relevance_scores, lexical_relevance_scores
from sibyl.retrieval_eval import evaluate_retrieval_cases, load_retrieval_cases


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "evals" / "retrieval_ranker_cases.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ranker", choices=["lexical", "flashrank"], default="lexical")
    parser.add_argument("--min-hit-at-1", type=float, default=1.0)
    parser.add_argument("--min-mrr", type=float, default=1.0)
    args = parser.parse_args()

    scorer = (
        flashrank_relevance_scores
        if args.ranker == "flashrank"
        else lexical_relevance_scores
    )
    try:
        cases = load_retrieval_cases(args.dataset)
        result = evaluate_retrieval_cases(cases, scorer)
    except Exception as exc:
        print(json.dumps({"ranker": args.ranker, "error": str(exc)}, ensure_ascii=False))
        return 2

    passed = (
        result.hit_at_1 >= args.min_hit_at_1
        and result.mean_reciprocal_rank >= args.min_mrr
    )
    output = {
        "dataset": str(args.dataset),
        "ranker": args.ranker,
        **result.to_dict(),
        "thresholds": {
            "min_hit_at_1": args.min_hit_at_1,
            "min_mrr": args.min_mrr,
        },
        "passed": passed,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
