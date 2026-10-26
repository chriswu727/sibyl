"""Deterministic passage splitting tests."""

import unittest

from sibyl.passages import select_passages, split_passages


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
            self.assertEqual(text[passage.start_char : passage.end_char], passage.text)

    def test_empty_text_or_non_positive_limit_returns_empty(self):
        self.assertEqual(split_passages("", max_chars=100), [])
        self.assertEqual(split_passages("text", max_chars=0), [])


class TestSelectPassages(unittest.TestCase):
    def test_patch_identifier_in_tail_is_found_with_exact_offsets(self):
        text = (
            "Intro paragraph about the game. " * 300
            + "History: Desktop 1.4.1 reduced titanium ore from 5 to 4."
            + " More trailing notes." * 50
        )
        passages = select_passages(
            "titanium ore 1.4.1 patch history", text, max_chars=2000
        )
        self.assertTrue(any("Desktop 1.4.1" in p.text for p in passages))
        self.assertLessEqual(sum(len(p.text) for p in passages), 2000)
        for passage in passages:
            self.assertEqual(text[passage.start_char : passage.end_char], passage.text)

    def test_final_boundary_evidence_is_not_skipped(self):
        text = "filler " * 200 + "boundaryneedle"
        passages = select_passages(
            "boundaryneedle", text, max_chars=500, max_passages=1
        )
        self.assertEqual(passages[0].end_char, len(text))
        self.assertIn("boundaryneedle", passages[0].text)

    def test_selects_body_from_separate_sections_instead_of_contents(self):
        contents = "Contents\n" + (
            "WAL concurrency\nReaders and writers\nNetwork filesystem limitations\n"
            * 18
        )
        text = (
            contents
            + "\n"
            + "Background explanation of storage. " * 100
            + "\nWAL concurrency\n"
            + "Readers do not block writers, but there can only be one writer at a time. "
            * 8
            + "\n"
            + "Unrelated implementation details. " * 160
            + "\nNetwork filesystem limitations\n"
            + "The shared memory requirement prevents WAL use over a network filesystem. "
            * 8
        )
        passages = select_passages(
            "WAL concurrency readers writers network filesystem limitations",
            text,
            max_chars=2400,
            max_passages=3,
        )
        selected = "\n".join(p.text for p in passages)
        self.assertIn("only be one writer", selected)
        self.assertIn("shared memory requirement", selected)
        self.assertLessEqual(sum(len(p.text) for p in passages), 2400)
        self.assertLessEqual(len(passages), 3)
        for passage in passages:
            self.assertEqual(text[passage.start_char : passage.end_char], passage.text)

    def test_chinese_query_can_select_chinese_evidence_after_the_head(self):
        text = (
            "产品介绍与安装说明。" * 500
            + "\n并发限制\n数据库只允许一个写入者，同时支持多个读取者。" * 20
        )
        passages = select_passages(
            "数据库并发写入限制", text, max_chars=600, max_passages=2
        )
        self.assertIn("只允许一个写入者", "\n".join(p.text for p in passages))

    def test_short_sources_keep_all_their_content(self):
        text = "  Limits and exceptions. " * 20
        passages = select_passages("exceptions", text, max_chars=1000, max_passages=3)
        self.assertEqual(
            "".join(p.text.replace(" ", "") for p in passages), text.replace(" ", "")
        )

    def test_empty_query_and_invalid_limits(self):
        text = "First section. " * 200 + "Last section. " * 200
        passages = select_passages("", text, max_chars=500, max_passages=1)
        self.assertEqual(passages[0].start_char, 0)
        self.assertLessEqual(len(passages[0].text), 500)
        self.assertEqual(select_passages("q", text, max_chars=0), [])
        self.assertEqual(select_passages("q", text, max_passages=0), [])
        self.assertEqual(select_passages("q", ""), [])

    def test_selection_is_repeatable_nonoverlapping_and_deduplicated(self):
        text = (
            "Important retry behavior requires a unique request token. " * 12 + "\n"
        ) * 20
        first = select_passages("retry token", text, max_chars=1800)
        self.assertEqual(first, select_passages("retry token", text, max_chars=1800))
        self.assertEqual(len({p.text for p in first}), len(first))
        ordered = sorted(first, key=lambda p: p.start_char)
        for passage in ordered:
            if passage.start_char:
                self.assertFalse(
                    text[passage.start_char - 1].isalnum()
                    and text[passage.start_char].isalnum()
                )
        self.assertTrue(
            all(a.end_char <= b.start_char for a, b in zip(ordered, ordered[1:]))
        )


if __name__ == "__main__":
    unittest.main()
