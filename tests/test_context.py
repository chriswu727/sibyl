"""Shared context helpers — numbering, snippet selection. No network.

Run: python -m unittest discover tests
"""

import unittest

from sibyl.context import relevant_window, relevant_window_span


class TestRelevantWindow(unittest.TestCase):
    def test_returns_tail_region_with_the_answer(self):
        # answer buried far past the head; window must surface it
        head = "intro paragraph about the game. " * 300  # ~9600 chars of filler
        tail = "History: Desktop 1.4.1 reduced titanium ore from 5 to 4."
        text = head + tail + (" more trailing notes." * 50)
        win = relevant_window("titanium ore 1.4.1 patch history", text, width=2000)
        self.assertIn("1.4.1", win)

    def test_checks_the_final_window_and_respects_width(self):
        text = "filler " * 200 + "boundaryneedle"
        win = relevant_window("boundaryneedle", text, width=500)

        self.assertIn("boundaryneedle", win)
        self.assertEqual(len(win), 500)

        start, end = relevant_window_span("boundaryneedle", text, width=500)
        self.assertEqual(end, len(text))
        self.assertEqual(end - start, 500)

    def test_short_text_unchanged(self):
        self.assertEqual(relevant_window("q", "short text", width=2000), "short text")

    def test_empty_query_returns_head(self):
        text = "a" * 5000
        self.assertTrue(relevant_window("", text, width=1000).startswith("a"))
        self.assertEqual(len(relevant_window("", text, width=1000)), 1000)

    def test_non_positive_width_returns_empty(self):
        self.assertEqual(relevant_window("query", "text", width=0), "")


if __name__ == "__main__":
    unittest.main()
