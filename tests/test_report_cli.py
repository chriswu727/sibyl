"""Markdown rendering and the command line. No network."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner
from sibyl_fixtures import ledger_data, page_reader

from sibyl.checks import run_check as real_run_check
from sibyl.cli import cli


def offline_run_check(ledger, digest, *, pages_dir=None):
    return real_run_check(ledger, digest, reader=page_reader(), pages_dir=pages_dir)


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.ledger = self.root / "wal.json"
        self.ledger.write_text(json.dumps(ledger_data()), encoding="utf-8")
        patcher = mock.patch("sibyl.cli.run_check", offline_run_check)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.runner = CliRunner()

    def invoke(self, *args):
        return self.runner.invoke(cli, [str(arg) for arg in args])

    def test_check_then_report(self):
        checked = self.invoke("check", self.ledger)
        self.assertEqual(checked.exit_code, 0, checked.output)
        self.assertIn("Read 3 of 3 cited pages.", checked.output)
        result = json.loads((self.root / "wal.check.json").read_text("utf-8"))
        self.assertTrue(result["passed"])

        reported = self.invoke("report", self.ledger)
        self.assertEqual(reported.exit_code, 0, reported.output)
        markdown = (self.root / "wal.md").read_text("utf-8")
        self.assertTrue(markdown.startswith("# When did SQLite add"))
        self.assertIn("1. SQLite added write-ahead logging in version 3.7.0", markdown)
        self.assertIn("*(disputed)*", markdown)
        self.assertIn("— Against [3]", markdown)
        self.assertIn("Based on finding 1, finding 2:", markdown)
        self.assertIn("## Unresolved", markdown)
        self.assertIn("- Searched for: Reports that WAL", markdown)
        self.assertIn(
            "1. Write-Ahead Logging — <https://www.sqlite.org/wal.html>", markdown
        )
        self.assertIn("- Finding 2: all supporting quotes come from one site", markdown)
        self.assertIn("they do not establish that a quote means", markdown)
        self.assertNotIn("wal-version", markdown)

    def test_invalid_ledger_exits_2(self):
        self.ledger.write_text("{}", encoding="utf-8")
        result = self.invoke("check", self.ledger)
        self.assertEqual(result.exit_code, 2)
        self.assertIn("question: Field required", result.output)

    def test_failed_check_exits_1_and_report_requires_override(self):
        data = ledger_data()
        data["claims"][0]["evidence"][0]["quote"] = "WAL arrived in version 3.8.0."
        self.ledger.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.invoke("check", self.ledger).exit_code, 1)
        refused = self.invoke("report", self.ledger)
        self.assertEqual(refused.exit_code, 1)
        self.assertIn("--allow-issues", refused.output)
        allowed = self.invoke(
            "report", self.ledger, "--allow-issues", "-o", self.root / "r.md"
        )
        self.assertEqual(allowed.exit_code, 0, allowed.output)
        markdown = (self.root / "r.md").read_text("utf-8")
        self.assertIn("contains claims that failed Sibyl's checks", markdown)
        self.assertIn("failed checks: quote_not_found", markdown)
        self.assertIn("*(rests on a claim that failed checks)*", markdown)

    def test_report_requires_a_current_check(self):
        missing = self.invoke("report", self.ledger)
        self.assertEqual(missing.exit_code, 2)
        self.assertIn("run `sibyl check", missing.output)
        self.invoke("check", self.ledger)
        self.ledger.write_text(json.dumps(ledger_data(), indent=1), encoding="utf-8")
        stale = self.invoke("report", self.ledger)
        self.assertEqual(stale.exit_code, 2)
        self.assertIn("changed after it was checked", stale.output)

    def test_template_validates_and_skill_installs(self):
        template = self.invoke("template")
        self.assertEqual(json.loads(template.output)["ledger"], "sibyl-ledger/1")
        shown = self.invoke("skill", "show")
        self.assertTrue(shown.output.startswith("---\nname: sibyl\n"))
        installed = self.invoke("skill", "install", "--dir", self.root / "skills")
        self.assertEqual(installed.exit_code, 0, installed.output)
        self.assertEqual(
            (self.root / "skills" / "SKILL.md").read_text("utf-8"), shown.output
        )

    def test_check_json_output(self):
        result = self.invoke("check", self.ledger, "--json")
        self.assertEqual(json.loads(result.output)["passed"], True)


if __name__ == "__main__":
    unittest.main()
