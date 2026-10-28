"""CLI validation tests. No network."""

import unittest
from unittest import mock

from click.testing import CliRunner

from sibyl.cli import cli
from sibyl.evidence import BundleDiagnostics, SourceBundle


class TestCliValidation(unittest.TestCase):
    def test_gather_command_returns_structured_json_without_llm_config(self):
        bundle = SourceBundle(
            "1.6",
            "sb_test",
            "question",
            "insufficient_evidence",
            [],
            BundleDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 7000, 7000, 1),
            "No sources found.",
        )
        with mock.patch(
            "sibyl.cli.gather_bundle", new=mock.AsyncMock(return_value=bundle)
        ):
            result = CliRunner().invoke(cli, ["gather", "question", "--format", "json"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn('"bundle_id": "sb_test"', result.output)
        self.assertIn('"status": "insufficient_evidence"', result.output)

    def test_root_help_only_exposes_evidence_inspection(self):
        result = CliRunner().invoke(cli, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("gather", result.output)
        self.assertNotIn("research", cli.commands)


if __name__ == "__main__":
    unittest.main()
