"""End-to-end transport replay with inspectable source fixtures."""

import subprocess
import sys
import unittest
from pathlib import Path


class TestResearchWorkflow(unittest.TestCase):
    def replay(self, *arguments):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts/check_research_workflow.py"),
                *arguments,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("successful replay response bytes", result.stdout)
        self.assertIn("structured error recovery", result.stdout)
        self.assertIn("no network/model calls", result.stdout)

    def test_real_stdio_research_and_export_replay(self):
        self.replay()

    def test_legacy_agent_flag_runs_the_same_replay(self):
        self.replay("--agent")
