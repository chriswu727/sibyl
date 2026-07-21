"""Search-query variant tests."""
import unittest

from sibyl.queries import search_query_variants


class TestSearchQueryVariants(unittest.TestCase):
    def test_keeps_original_and_adds_focused_question_variant(self):
        query = "Who was the Serbian quarterfinalist in the 2018 Madrid Open men's singles?"

        self.assertEqual(
            search_query_variants(query),
            [
                query,
                "Serbian quarterfinalist 2018 Madrid Open men's singles",
            ],
        )

    def test_does_not_rewrite_keyword_query(self):
        query = "Serbian quarterfinalist 2018 Madrid Open men's singles"

        self.assertEqual(search_query_variants(query), [query])

    def test_quoted_title_becomes_the_focused_variant(self):
        query = (
            'What is the publication date of the paper "Articulatory constraints '
            'on stop insertion and elision in consonant clusters"?'
        )

        self.assertEqual(
            search_query_variants(query),
            [
                query,
                "Articulatory constraints on stop insertion and elision in consonant clusters",
            ],
        )

    def test_empty_query_has_no_variants(self):
        self.assertEqual(search_query_variants("  "), [])


if __name__ == "__main__":
    unittest.main()
