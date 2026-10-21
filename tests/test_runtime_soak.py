"""Short deterministic resource-soak regression; extended runs use the script."""

import asyncio
import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.check_runtime_soak import check_outcomes


class TestRuntimeSoak(unittest.TestCase):
    def test_timeout_cycle_accepts_either_timeout_boundary(self):
        check_outcomes(2, [{"status": "awaiting_input"}, TimeoutError()])

    def test_other_failures_are_not_hidden_by_timeout_outcomes(self):
        for mode, results in (
            (0, [TimeoutError()]),
            (1, [{}]),
            (1, [TimeoutError()]),
            (2, [ValueError("unexpected")]),
            (2, [asyncio.CancelledError()]),
        ):
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                check_outcomes(mode, results)

    def test_failure_cycles_reclaim_workers_files_and_checkpoints(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts/check_runtime_soak.py"),
                "--rounds",
                "6",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(result.stdout)
        self.assertTrue(record["passed"])
        self.assertEqual(record["rounds"], 6)
        self.assertEqual(record["backup_restore"], "passed with runtime stopped")
