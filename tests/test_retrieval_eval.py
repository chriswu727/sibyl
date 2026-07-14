"""Offline retrieval-ranker regression tests."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sibyl.ranking import lexical_relevance_scores
from sibyl.retrieval_eval import (
    RetrievalCase,
    RetrievalDocument,
    evaluate_retrieval_cases,
    load_retrieval_cases,
)


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "retrieval_ranker_cases.jsonl"


class TestRetrievalEval(unittest.TestCase):
    def test_fixed_dataset_meets_lexical_regression_floor(self):
        cases = load_retrieval_cases(DATASET)

        result = evaluate_retrieval_cases(cases, lexical_relevance_scores)

        self.assertEqual(result.total_cases, 8)
        self.assertEqual(result.hit_at_1, 1.0)
        self.assertEqual(result.mean_reciprocal_rank, 1.0)
        self.assertEqual(len(result.cases), result.total_cases)

    def test_metrics_use_first_relevant_rank(self):
        case = RetrievalCase(
            case_id="case",
            query="query",
            documents=[
                RetrievalDocument("noise", "Noise", "text"),
                RetrievalDocument("relevant", "Relevant", "text"),
                RetrievalDocument("other", "Other", "text"),
            ],
            relevant_document_ids=["relevant"],
        )

        result = evaluate_retrieval_cases([case], lambda query, documents: [0.9, 0.8, 0.1])

        self.assertEqual(result.hits_at_1, 0)
        self.assertEqual(result.hit_at_1, 0.0)
        self.assertEqual(result.mean_reciprocal_rank, 0.5)
        self.assertEqual(result.cases[0].top_document_id, "noise")
        self.assertEqual(result.cases[0].first_relevant_rank, 2)

    def test_loader_rejects_unknown_relevant_document(self):
        invalid = {
            "id": "case",
            "query": "query",
            "documents": [
                {"id": "a", "title": "A", "text": "text"},
                {"id": "b", "title": "B", "text": "text"},
            ],
            "relevant_document_ids": ["missing"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps(invalid), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown relevant document"):
                load_retrieval_cases(path)

    def test_evaluator_rejects_score_count_mismatch(self):
        case = load_retrieval_cases(DATASET)[0]

        with self.assertRaisesRegex(ValueError, "returned 1 scores"):
            evaluate_retrieval_cases([case], lambda query, documents: [1.0])

    def test_cli_returns_machine_readable_result(self):
        completed = subprocess.run(
            [sys.executable, "scripts/eval_retrieval.py", "--ranker", "lexical"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads(completed.stdout)
        self.assertTrue(output["passed"])
        self.assertEqual(output["ranker"], "lexical")
        self.assertEqual(output["total_cases"], 8)


if __name__ == "__main__":
    unittest.main()
