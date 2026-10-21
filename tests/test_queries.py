"""Search-query variant tests."""
import unittest

from sibyl.queries import (
    focused_site_query,
    historical_role_requirement,
    query_requires_decomposition,
    query_site_hosts,
    search_query_variants,
)


class TestSearchQueryVariants(unittest.TestCase):
    def test_focused_site_query_removes_union_operators(self):
        self.assertEqual(focused_site_query('2026 results (site:first.example OR site:"second.example")', "second.example"),
                         "2026 results site:second.example")

    def test_site_hosts_strip_syntax_not_domain_boundaries(self):
        self.assertEqual(query_site_hosts('(site:First.example OR site:second.example)'),
                         ["first.example", "second.example"])
        self.assertEqual(query_site_hosts('site:"docs.example.com/path" -site:excluded.com'),
                         ["docs.example.com"])
        self.assertEqual(query_site_hosts("[site:https://example.com/path], site:example.com."),
                         ["example.com"])
        self.assertEqual(query_site_hosts("website:ignored.com -site:excluded.com"), [])

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
