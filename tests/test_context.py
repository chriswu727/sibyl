"""Shared context helpers — numbering, snippet selection. No network.

Run: python -m unittest discover tests
"""
import unittest

from sibyl.context import build_source_context, best_snippet, relevant_window
from sibyl.scraper import WebPage


class TestRelevantWindow(unittest.TestCase):
    def test_returns_tail_region_with_the_answer(self):
        # answer buried far past the head; window must surface it
        head = "intro paragraph about the game. " * 300           # ~9600 chars of filler
        tail = "History: Desktop 1.4.1 reduced titanium ore from 5 to 4."
        text = head + tail + (" more trailing notes." * 50)
        win = relevant_window("titanium ore 1.4.1 patch history", text, width=2000)
        self.assertIn("1.4.1", win)

    def test_checks_the_final_window_and_respects_width(self):
        text = "filler " * 200 + "boundaryneedle"
        win = relevant_window("boundaryneedle", text, width=500)

        self.assertIn("boundaryneedle", win)
        self.assertEqual(len(win), 500)

    def test_short_text_unchanged(self):
        self.assertEqual(relevant_window("q", "short text", width=2000), "short text")

    def test_empty_query_returns_head(self):
        text = "a" * 5000
        self.assertTrue(relevant_window("", text, width=1000).startswith("a"))
        self.assertEqual(len(relevant_window("", text, width=1000)), 1000)

    def test_non_positive_width_returns_empty(self):
        self.assertEqual(relevant_window("query", "text", width=0), "")


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
