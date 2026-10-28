"""Quote, number, name and dependency checks over fake pages. No network."""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from sibyl_fixtures import PAGES, ledger_data, page_reader

from sibyl.checks import locate, normalize, read_sources, run_check, summary
from sibyl.ledger import Ledger


def checked(data=None, pages=None, failures=None, pages_dir=None):
    ledger = Ledger.model_validate(data or ledger_data())
    return run_check(
        ledger, "0" * 64, reader=page_reader(pages, failures), pages_dir=pages_dir
    )


def row(result, claim_id):
    return next(item for item in result["claims"] if item["id"] == claim_id)


class TestMatching(unittest.TestCase):
    def test_normalization_ignores_typography_case_and_whitespace(self):
        page = normalize("A “Write-Ahead Log”\n option — avail­able")
        self.assertEqual(locate('"write-ahead log" option - available', page), 2)

    def test_ellipsis_parts_must_occur_in_order(self):
        page = normalize("Readers do not block writers and a writer does not block.")
        self.assertIsNotNone(locate("Readers ... does not block", page))
        self.assertIsNotNone(locate("Readers […] does not block", page))
        self.assertIsNone(locate("does not block ... Readers", page))
        self.assertIsNone(locate("...", page))

    def test_missing_text_is_not_found(self):
        self.assertIsNone(locate("readers block writers", normalize("readers wait")))


class TestChecks(unittest.TestCase):
    def test_fixture_passes_with_disclosed_warnings(self):
        result = checked()
        self.assertTrue(result["passed"], summary(result))
        self.assertEqual(result["check"], "sibyl-check/1")
        self.assertEqual(row(result, "wal-version")["warnings"], [])
        concurrency = row(result, "wal-concurrency")
        self.assertEqual(concurrency["warnings"], ["single_source"])
        self.assertTrue(all(item["found"] for item in concurrency["evidence"]))
        self.assertEqual(result["notes"], [])
        self.assertIn("PASSED", summary(result))

    def test_altered_quote_blocks_and_propagates_to_inference(self):
        data = ledger_data()
        data["claims"][0]["evidence"][0]["quote"] = "WAL was added in version 3.8.0."
        result = checked(data)
        self.assertFalse(result["passed"])
        self.assertEqual(row(result, "wal-version")["blocking"], ["quote_not_found"])
        self.assertEqual(row(result, "advice")["blocking"], ["basis_has_issues"])
        text = summary(result)
        self.assertIn("BLOCKING quote_not_found", text)
        self.assertIn("not verified: https://www.sqlite.org/wal.html", text)
        self.assertIn("FAILED", text)

    def test_unreadable_page_blocks(self):
        result = checked(failures={"https://example.com/forum": "HTTP 403"})
        self.assertEqual(
            row(result, "wal-concurrency")["blocking"], ["source_unreadable"]
        )
        source = next(s for s in result["sources"] if "forum" in s["url"])
        self.assertEqual((source["error"], source["text_sha256"]), ("HTTP 403", ""))
        self.assertIn(
            "unreadable: https://example.com/forum (HTTP 403)", summary(result)
        )

    def test_malformed_url_is_reported_unreadable(self):
        data = ledger_data()
        data["claims"][1]["evidence"][0]["url"] = "https://[invalid"
        result = checked(data, failures={"https://[invalid": "Unsafe URL: invalid URL"})
        self.assertIn("source_unreadable", row(result, "wal-concurrency")["blocking"])

    def test_dense_inference_graph_checks_quickly(self):
        data = ledger_data()
        data["claims"][0]["evidence"][0]["quote"] = "Not on the page at all."
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
        result = checked(data)
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(row(result, "i149")["blocking"], ["basis_has_issues"])

    def test_number_missing_from_quotes_blocks(self):
        data = ledger_data()
        data["claims"][0]["text"] = "SQLite added write-ahead logging in 3.7.0 in 2011."
        claim = row(checked(data), "wal-version")
        self.assertEqual(claim["blocking"], ["number_not_quoted"])
        self.assertEqual(claim["details"]["unquoted_numbers"], ["2011"])

    def test_name_absent_near_quotes_warns(self):
        data = ledger_data()
        data["claims"][1]["text"] = (
            "In WAL mode PostgreSQL readers do not block writers."
        )
        claim = row(checked(data), "wal-concurrency")
        self.assertIn("name_not_in_sources", claim["warnings"])
        self.assertEqual(claim["details"]["unmatched_names"], ["PostgreSQL"])

    def test_conflicting_numbers_and_short_quotes_warn(self):
        data = ledger_data()
        data["claims"].append(
            {
                "id": "wal-version-other",
                "text": "SQLite added write-ahead logging in version 3.6.0 in 2010.",
                "evidence": [
                    {
                        "url": "https://en.wikipedia.org/wiki/SQLite",
                        "quote": "version 3.7.0",
                    }
                ],
            }
        )
        result = checked(data)
        other = row(result, "wal-version-other")
        self.assertIn("short_quote", other["warnings"])
        self.assertIn("number_not_quoted", other["blocking"])
        self.assertEqual(
            row(result, "wal-version")["details"]["conflicts_with"],
            ["wal-version-other"],
        )

    def test_missing_counterevidence_is_noted(self):
        data = ledger_data()
        del data["counterevidence"]
        result = checked(data)
        self.assertEqual(result["notes"], ["no counterevidence search is recorded"])

    def test_each_url_is_read_once_and_pages_can_be_kept(self):
        calls = []
        reader = page_reader()

        async def counting(urls):
            calls.append(list(urls))
            return await reader(urls)

        sources = asyncio.run(read_sources(["https://example.com/forum"] * 1, counting))
        self.assertEqual(len(sources), 1)
        with tempfile.TemporaryDirectory() as directory:
            result = checked(pages_dir=Path(directory))
            kept = sorted(path.name for path in Path(directory).iterdir())
            self.assertEqual(len(kept), len(PAGES))
            self.assertEqual(
                sorted(f"{s['text_sha256']}.txt" for s in result["sources"]), kept
            )


if __name__ == "__main__":
    unittest.main()
