"""Paired host evaluation helpers: cited URL extraction and metric aggregation."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "judge_paired_host_eval.py"
)
spec = importlib.util.spec_from_file_location("judge_paired_host_eval", SCRIPT)
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


class TestPairedEval(unittest.TestCase):
    def test_cited_urls_are_deduplicated_and_trimmed(self):
        text = (
            "- Claim. https://a.example/x.\n"
            "  (https://b.example/y), https://a.example/x **https://c.example/z**"
        )
        self.assertEqual(
            judge.cited_urls(text),
            ["https://a.example/x", "https://b.example/y", "https://c.example/z"],
        )

    def test_aggregate_weights_support_and_excludes_unreachable(self):
        cases = [
            {"id": "f", "kind": "factual"},
            {"id": "t", "kind": "trap"},
        ]
        record = {
            "cli_reported_cost_usd": 0.5,
            "elapsed_seconds": 10,
            "num_turns": 3,
            "is_error": False,
            "timed_out": False,
        }
        runs = {"f": {"a": record}, "t": {"a": record}}
        judged = {
            "f": {
                "by_condition": {
                    "a": {
                        "verdict": "correct",
                        "claims": [
                            {"label": "supported"},
                            {"label": "partial"},
                            {"label": "unsupported"},
                            {"label": "unreachable"},
                        ],
                        "uncited_claims": 1,
                    }
                }
            },
            "t": {"by_condition": {"a": {"verdict": "fail", "claims": []}}},
        }
        row = judge.aggregate(cases, runs, judged)["a"]
        self.assertEqual((row["factual_correct"], row["trap_pass"]), (1, 0))
        self.assertEqual(row["judged_pairs"], 3)
        self.assertEqual(row["unreachable_pairs"], 1)
        self.assertEqual(row["citation_support_rate"], 0.5)
        self.assertEqual(row["cost_usd"], 1.0)


if __name__ == "__main__":
    unittest.main()
