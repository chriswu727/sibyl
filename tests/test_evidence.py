"""Structured evidence protocol tests."""
import unittest

from sibyl.evidence import BundleDiagnostics, EvidencePassage, EvidenceSource, SourceBundle


class TestSourceBundle(unittest.TestCase):
    def test_to_dict_preserves_nested_schema_and_null_scores(self):
        passage = EvidencePassage("P1", "sb_1/S1/P1", "evidence", "a" * 64)
        source = EvidenceSource(
            "S1",
            "https://example.com",
            "Example",
            "2026-07-14T00:00:00+00:00",
            "b" * 64,
            "web",
            8,
            [passage],
        )
        diagnostics = BundleDiagnostics(
            search_results=1,
            unique_urls=1,
            urls_attempted=1,
            pages_scraped=1,
            scrape_failures=0,
            snippet_fallbacks=0,
            wikipedia_fallbacks=0,
            sources_returned=1,
            requested_max_sources=10,
            effective_max_sources=10,
            requested_chars_per_source=7000,
            effective_chars_per_source=7000,
            latency_ms=5,
        )
        bundle = SourceBundle("1.0", "sb_1", "query", "ok", [source], diagnostics)

        data = bundle.to_dict()

        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["sources"][0]["evidence"][0]["citation_id"], "sb_1/S1/P1")
        self.assertIsNone(data["sources"][0]["relevance_score"])
        self.assertIsNone(data["sources"][0]["quality_score"])
        self.assertIsNone(data["sources"][0]["evidence"][0]["score"])


if __name__ == "__main__":
    unittest.main()
