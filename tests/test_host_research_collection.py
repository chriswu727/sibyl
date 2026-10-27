"""Offline validation of the bounded live MCP case collector."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_host_research import load_cases


class TestHostResearchCases(unittest.TestCase):
    def load(self, cases):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            return load_cases(path)

    def test_frozen_suite_has_ten_cases(self):
        path = Path(__file__).resolve().parents[1] / "evals/host_research_cases.json"
        self.assertEqual(len(load_cases(path)), 10)

    def test_rejects_unsafe_or_duplicate_names(self):
        for cases in (
            [{"id": "../escape", "queries": ["query"]}],
            [{"id": "same", "queries": ["query"]}] * 2,
        ):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                self.load(cases)

    def test_enforces_case_and_query_bounds(self):
        for cases in (
            [],
            [{"id": f"case_{i}", "queries": ["query"]} for i in range(21)],
            [{"id": "case", "queries": []}],
            [{"id": "case", "queries": ["query"] * 5}],
        ):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                self.load(cases)
