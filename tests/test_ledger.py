"""Ledger format validation. No network."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from sibyl_fixtures import ledger_data

from sibyl.ledger import TEMPLATE, Ledger, LedgerError, check_path, read_ledger


class TestLedger(unittest.TestCase):
    def write(self, data) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "topic.json"
        path.write_text(
            data if isinstance(data, str) else json.dumps(data), encoding="utf-8"
        )
        return path

    def error(self, data) -> str:
        with self.assertRaises(LedgerError) as caught:
            read_ledger(self.write(data))
        return str(caught.exception)

    def test_fixture_and_template_are_valid(self):
        ledger, digest = read_ledger(self.write(ledger_data()))
        self.assertEqual(len(ledger.claims), 4)
        self.assertEqual(len(digest), 64)
        Ledger.model_validate(TEMPLATE)

    def test_digest_tracks_exact_bytes(self):
        first = self.write(ledger_data())
        second = self.write(json.dumps(ledger_data(), indent=2))
        self.assertNotEqual(read_ledger(first)[1], read_ledger(second)[1])

    def test_errors_name_the_field(self):
        data = ledger_data()
        del data["claims"][0]["evidence"][0]["quote"]
        self.assertIn("claims[0].evidence[0].quote: Field required", self.error(data))

    def test_unknown_fields_are_rejected(self):
        data = ledger_data()
        data["claims"][0]["confidence"] = 0.9
        self.assertIn("claims[0].confidence", self.error(data))

    def test_claim_shapes(self):
        cases = {
            "supporting quote": {"id": "x", "text": "t"},
            "one supporting and one challenging": {
                "id": "x",
                "text": "t",
                "status": "disputed",
                "evidence": [{"url": "https://a.org", "quote": "q"}],
            },
            "needs a note": {"id": "x", "text": "t", "status": "unresolved"},
            "based_on and a rationale": {"id": "x", "text": "t", "kind": "inference"},
            "cannot be disputed": {
                "id": "x",
                "text": "t",
                "kind": "inference",
                "status": "disputed",
                "based_on": ["wal-version"],
                "rationale": "r",
            },
            "not evidence": {
                "id": "x",
                "text": "t",
                "kind": "inference",
                "based_on": ["wal-version"],
                "rationale": "r",
                "evidence": [{"url": "https://a.org", "quote": "q"}],
            },
            "belongs to inferences": {
                "id": "x",
                "text": "t",
                "based_on": ["wal-version"],
                "evidence": [{"url": "https://a.org", "quote": "q"}],
            },
        }
        for expected, claim in cases.items():
            with self.subTest(expected=expected):
                data = ledger_data()
                data["claims"].append(claim)
                self.assertIn(expected, self.error(data))

    def test_references_ids_and_cycles(self):
        data = ledger_data()
        data["claims"][1]["id"] = "wal-version"
        self.assertIn("duplicate claim ids: wal-version", self.error(data))
        data = ledger_data()
        data["claims"][2]["based_on"] = ["missing"]
        self.assertIn("unknown claims: missing", self.error(data))
        data = ledger_data()
        data["claims"].append(
            {
                "id": "loop",
                "kind": "inference",
                "text": "t",
                "based_on": ["advice"],
                "rationale": "r",
            }
        )
        data["claims"][2]["based_on"] = ["loop"]
        self.assertIn("circular based_on", self.error(data))

    def test_dense_dependency_graph_validates_quickly(self):
        data = ledger_data()
        previous = ["wal-version", "wal-concurrency"]
        for index in range(150):
            data["claims"].append(
                {
                    "id": f"i{index}",
                    "kind": "inference",
                    "text": "t",
                    "based_on": previous[-2:],
                    "rationale": "r",
                }
            )
            previous.append(f"i{index}")
        started = time.monotonic()
        read_ledger(self.write(data))
        self.assertLess(time.monotonic() - started, 2)

    def test_unreadable_inputs(self):
        self.assertIn("not valid JSON", self.error("{"))
        self.assertIn("cannot read", self.error_for_missing())
        self.assertIn("exceeds", self.error(" " * (1024 * 1024 + 1)))

    def error_for_missing(self) -> str:
        with self.assertRaises(LedgerError) as caught:
            read_ledger(Path(tempfile.gettempdir()) / "missing-sibyl-ledger.json")
        return str(caught.exception)

    def test_check_path_sits_beside_the_ledger(self):
        self.assertEqual(
            check_path(Path("research/topic.json")), Path("research/topic.check.json")
        )


if __name__ == "__main__":
    unittest.main()
