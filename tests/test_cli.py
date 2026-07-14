"""CLI validation tests. No network."""
import os
import unittest
from unittest import mock

from click.testing import CliRunner

from sibyl.cli import main


class TestCliValidation(unittest.TestCase):
    def test_default_effort_parses_and_missing_key_is_actionable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            result = CliRunner().invoke(main, ["test query"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("requires an LLM provider key", result.output)
        self.assertNotIn("Invalid value for '--effort'", result.output)


if __name__ == "__main__":
    unittest.main()
