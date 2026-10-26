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

    def test_chinese_query_selects_the_answer_region(self):
        text = "安装介绍。" * 1000 + "\n写入限制：数据库只支持一个写入者。" * 50
        self.assertIn(
            "只支持一个写入者", relevant_window("数据库写入限制", text, width=600)
        )

    def test_body_matches_beat_repeated_contents_headings(self):
        text = (
            "Contents\n"
            + "Token retention\nLate retries\n" * 80
            + "\nBackground material. " * 200
            + "\nToken retention cannot be indefinite. Late retries require handling after expiry. "
            * 20
        )
        start, end = relevant_window_span(
            "token retention late retries", text, width=800
        )
        self.assertGreater(start, 2000)
        self.assertIn("cannot be indefinite", text[start:end])


if __name__ == "__main__":
    unittest.main()
