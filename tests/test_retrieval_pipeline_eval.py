"""Offline end-to-end retrieval pipeline regression tests."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

from sibyl.retrieval_eval import load_retrieval_cases
from sibyl.retrieval_pipeline_eval import evaluate_pipeline_cases


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "retrieval_ranker_cases.jsonl"


class TestRetrievalPipelineEval(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_dataset_passes_the_full_pipeline(self):
        result = await evaluate_pipeline_cases(load_retrieval_cases(DATASET))

        self.assertEqual(result.total_cases, 8)
        self.assertEqual(result.top_source_accuracy, 1.0)
        self.assertEqual(result.usable_bundle_rate, 1.0)
        self.assertEqual(result.structure_valid_rate, 1.0)
        self.assertEqual(result.case_pass_rate, 1.0)


class TestRetrievalPipelineEvalCli(unittest.TestCase):
    def test_cli_returns_machine_readable_result(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/eval_retrieval_pipeline.py",
                "--ranker",
                "lexical",
            ],
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
