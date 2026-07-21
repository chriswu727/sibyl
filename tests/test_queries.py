"""Search-query variant tests."""
import unittest

from sibyl.queries import (
    historical_role_requirement,
    query_requires_decomposition,
    search_query_variants,
)


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

    def test_rewrites_prepositional_question(self):
        query = "In what year was the company that created CUDA founded?"

        self.assertEqual(
            search_query_variants(query),
            [query, "year company created CUDA founded"],
        )

    def test_rewrites_question_after_leading_context(self):
        query = "In distributed systems, what algorithm is easier than Paxos?"

        self.assertEqual(
            search_query_variants(query),
            [query, "distributed systems algorithm easier than Paxos"],
        )

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

    def test_detects_dependent_relative_clause(self):
        self.assertTrue(
            query_requires_decomposition(
                "In what year was the company that created CUDA founded?"
            )
        )

    def test_detects_intermediate_country_chain(self):
        self.assertTrue(
            query_requires_decomposition(
                "Which river crosses the capital of the country that won the final?"
            )
        )

    def test_detects_question_after_leading_claim(self):
        self.assertTrue(
            query_requires_decomposition(
                "A director made one film. What other film did they direct?"
            )
        )

    def test_keeps_atomic_question_single_step(self):
        self.assertFalse(
            query_requires_decomposition("In what year was NVIDIA founded?")
        )

    def test_keeps_direct_relative_description_single_step(self):
        self.assertFalse(
            query_requires_decomposition(
                "What is the asteroid moonlet that DART impacted in 2022?"
            )
        )

    def test_keeps_direct_event_date_single_step(self):
        self.assertFalse(
            query_requires_decomposition(
                "What year was it when the headquarters was completed?"
            )
        )

    def test_extracts_historical_role_requirement(self):
        self.assertEqual(
            historical_role_requirement(
                "Who was the rector of Hacettepe University in 2006?"
            ),
            ("rector", 2006),
        )

    def test_ignores_non_historical_role_question(self):
        self.assertIsNone(
            historical_role_requirement("Who founded Hacettepe University?")
        )


if __name__ == "__main__":
    unittest.main()
