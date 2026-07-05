"""Shared context helpers — numbering, snippet selection. No network.

Run: python -m unittest discover tests
"""
import unittest

from sibyl.context import build_source_context, best_snippet
from sibyl.scraper import WebPage


class TestBuildSourceContext(unittest.TestCase):
    def test_numbering_and_limit(self):
        pages = [WebPage(url=f"u{i}", title=f"T{i}", text=f"body {i} " * 10) for i in range(5)]
        ctx = build_source_context(pages, limit=3)
        self.assertIn("[Source 1: T0]", ctx)
        self.assertIn("[Source 3: T2]", ctx)
        self.assertNotIn("[Source 4:", ctx)  # limit respected

    def test_per_char_truncation(self):
        pages = [WebPage(url="u", title="T", text="x" * 9000)]
        ctx = build_source_context(pages, limit=1, per_char=100)
        # the body slice is capped at per_char
        self.assertLess(len(ctx), 400)


class TestBestSnippet(unittest.TestCase):
    def test_picks_overlapping_sentence(self):
        text = ("The weather was nice today. NVIDIA reported record data center revenue of "
                "40 billion dollars. Cats are mammals.")
        snip = best_snippet("NVIDIA data center revenue", text)
        self.assertIn("NVIDIA", snip)
        self.assertIn("revenue", snip)

    def test_empty_query_falls_back_to_head(self):
        text = "First sentence here. Second sentence follows."
        self.assertEqual(best_snippet("", text, max_len=20), text[:20])

    def test_cjk_query_falls_back(self):
        text = "Some english sentence about markets and data."
        self.assertEqual(best_snippet("市场数据", text, max_len=15), text[:15])

    def test_empty_text(self):
        self.assertEqual(best_snippet("q", ""), "")


if __name__ == "__main__":
    unittest.main()
