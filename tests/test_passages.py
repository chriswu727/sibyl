"""Deterministic passage splitting tests."""
import unittest

from sibyl.passages import split_passages


class TestSplitPassages(unittest.TestCase):
    def test_short_text_is_preserved_with_offsets(self):
        passages = split_passages("  alpha beta  ", max_chars=100)

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].text, "alpha beta")
        self.assertEqual(passages[0].start_char, 2)
        self.assertEqual(passages[0].end_char, 12)

    def test_long_text_covers_the_tail_with_bounded_chunks(self):
        text = "alpha " * 300 + "tail evidence"
        passages = split_passages(text, max_chars=500, overlap_chars=100)

        self.assertGreater(len(passages), 1)
        self.assertTrue(all(len(passage.text) <= 500 for passage in passages))
        self.assertEqual(passages[-1].end_char, len(text))
        self.assertIn("tail evidence", passages[-1].text)

    def test_prefers_readable_boundaries(self):
        text = "a" * 70 + ". " + "b" * 70
        passages = split_passages(text, max_chars=100, overlap_chars=0)

        self.assertEqual(passages[0].text, "a" * 70 + ".")
        self.assertEqual(passages[1].text, "b" * 70)

    def test_overlap_keeps_offsets_consistent(self):
        text = "0123456789" * 30
        passages = split_passages(text, max_chars=100, overlap_chars=20)

        self.assertEqual(passages[1].start_char, 80)
        for passage in passages:
            self.assertEqual(text[passage.start_char:passage.end_char], passage.text)

    def test_empty_text_or_non_positive_limit_returns_empty(self):
        self.assertEqual(split_passages("", max_chars=100), [])
        self.assertEqual(split_passages("text", max_chars=0), [])


if __name__ == "__main__":
    unittest.main()
