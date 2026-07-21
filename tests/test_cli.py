"""CLI validation tests. No network."""
import os
import unittest
from unittest import mock

from click.testing import CliRunner

from sibyl.cli import cli, research_cli
from sibyl.evidence import BundleDiagnostics, SourceBundle


class TestCliValidation(unittest.TestCase):
    def test_default_effort_parses_and_missing_key_is_actionable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            result = CliRunner().invoke(research_cli, ["test query"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("requires an LLM provider key", result.output)
        self.assertNotIn("Invalid value for '--effort'", result.output)

    def test_report_extra_is_required_before_loading_provider_code(self):
        with mock.patch("sibyl.cli.find_spec", return_value=None):
            result = CliRunner().invoke(research_cli, ["test query"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("sibyl-research[report]", result.output)

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
        with mock.patch("sibyl.cli.gather_bundle", new=mock.AsyncMock(return_value=bundle)):
            result = CliRunner().invoke(cli, ["gather", "question", "--format", "json"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn('"bundle_id": "sb_test"', result.output)
        self.assertIn('"status": "insufficient_evidence"', result.output)

    def test_root_help_leads_with_two_product_modes(self):
        result = CliRunner().invoke(cli, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("gather", result.output)
        self.assertIn("research", result.output)


if __name__ == "__main__":
    unittest.main()
