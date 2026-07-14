"""Offline contextual source-quality evaluation tests."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sibyl.source_quality_eval import (
    SourceQualityCandidate,
    SourceQualityCase,
    evaluate_source_quality_cases,
    load_source_quality_cases,
    source_type_prior_scores,
)


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "source_quality_cases.jsonl"


class TestSourceQualityEval(unittest.TestCase):
    def test_fixed_dataset_pins_honest_source_type_baseline(self):
        cases = load_source_quality_cases(DATASET)

        result = evaluate_source_quality_cases(cases, source_type_prior_scores)

        self.assertEqual(result.total_cases, 8)
        self.assertEqual(result.assessed_cases, 6)
        self.assertEqual(result.correct_cases, 5)
        self.assertEqual(result.coverage, 0.75)
        self.assertEqual(result.selective_accuracy, 0.833333)
        self.assertEqual(result.overall_accuracy, 0.625)
        statuses = {case.case_id: case.status for case in result.cases}
        self.assertEqual(statuses["lived_product_experience"], "incorrect")
        self.assertEqual(statuses["official_release_notes"], "abstained")
        self.assertEqual(statuses["conflicting_news_reports"], "abstained")

    def test_equal_top_scores_count_as_abstention(self):
        case = SourceQualityCase(
            case_id="tie",
            query="query",
            candidates=[
                SourceQualityCandidate("a", "A", "https://a.example", "web"),
                SourceQualityCandidate("b", "B", "https://b.example", "web"),
            ],
            preferred_candidate_ids=["b"],
            label_reason="B is primary evidence.",
        )

        result = evaluate_source_quality_cases(
            [case], lambda query, candidates: [0.5, 0.5]
        )

        self.assertEqual(result.assessed_cases, 0)
        self.assertEqual(result.cases[0].status, "abstained")
        self.assertIsNone(result.cases[0].selected_candidate_id)

    def test_loader_rejects_unknown_preferred_candidate(self):
        invalid = {
            "id": "case",
            "query": "query",
            "candidates": [
                {
                    "id": "a",
                    "title": "A",
                    "url": "https://a.example",
                    "source_type": "web",
                },
                {
                    "id": "b",
                    "title": "B",
                    "url": "https://b.example",
                    "source_type": "web",
                },
            ],
            "preferred_candidate_ids": ["missing"],
            "label_reason": "A reason.",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps(invalid), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown preferred candidate"):
                load_source_quality_cases(path)

    def test_evaluator_rejects_non_finite_score(self):
        case = load_source_quality_cases(DATASET)[0]

        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluate_source_quality_cases(
                [case],
                lambda query, candidates: [math.nan] * len(candidates),
            )

    def test_cli_returns_machine_readable_result(self):
        completed = subprocess.run(
            [sys.executable, "scripts/eval_source_quality.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads(completed.stdout)
        self.assertTrue(output["passed"])
        self.assertEqual(output["baseline"], "source_type_prior")
        self.assertEqual(output["total_cases"], 8)


if __name__ == "__main__":
    unittest.main()
