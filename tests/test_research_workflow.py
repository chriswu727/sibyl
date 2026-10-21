"""End-to-end transport replay with inspectable source fixtures."""

import subprocess
import sys
import unittest
from pathlib import Path


class TestResearchWorkflow(unittest.TestCase):
    def test_real_stdio_research_and_export_replay(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "scripts/check_research_workflow.py")],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no network/model calls", result.stdout)
