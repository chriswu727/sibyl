"""Short deterministic resource-soak regression; extended runs use the script."""

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestRuntimeSoak(unittest.TestCase):
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
