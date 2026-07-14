"""Dependency-free lexical relevance tests."""
import sys
import types
import unittest
from unittest import mock

import sibyl.ranking as ranking
from sibyl.ranking import (
    flashrank_relevance_scores,
    lexical_query_coverage,
    lexical_relevance_scores,
)


class TestLexicalRelevance(unittest.TestCase):
    def test_direct_evidence_ranks_above_unrelated_text(self):
        documents = [
            ("Cooking", "A recipe for sourdough bread and tomato soup."),
            (
                "2018 Madrid Open men's singles",
                "Dušan Lajović reached the quarterfinals of the Madrid Open in 2018.",
            ),
            ("Tennis", "A general history of professional tennis."),
        ]

        scores = lexical_relevance_scores(
            "Serbian quarterfinalist 2018 Madrid Open men's singles", documents
        )

        self.assertGreater(scores[1], scores[2])
        self.assertEqual(scores[2], scores[0])
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

    def test_title_match_can_lift_a_document(self):
        scores = lexical_relevance_scores(
            "alpha beta",
            [("alpha beta", "short text"), ("other", "alpha appears once")],
        )

        self.assertGreater(scores[0], scores[1])

    def test_specific_title_beats_person_page_with_scattered_terms(self):
        correct, person = lexical_relevance_scores(
            "Serbian quarterfinalist 2018 Madrid Open men's singles",
            [
                (
                    "2018 Madrid Open men's singles",
                    "The tournament draw included a Serbian quarterfinalist.",
                ),
                (
                    "Borna Gojo",
                    "A men's player entered an Open in Madrid. His 2018 season included a quarterfinalist.",
                ),
            ],
        )

        self.assertGreater(correct, person)

    def test_exact_phrase_adds_a_bounded_signal(self):
        exact, separated = lexical_relevance_scores(
            "alpha beta",
            [("", "alpha beta"), ("", "alpha words between beta")],
        )

        self.assertGreater(exact, separated)
        self.assertLessEqual(exact, 1.0)

    def test_cjk_bigrams_match_without_external_tokenizer(self):
        relevant, unrelated = lexical_relevance_scores(
            "加拿大移民政策",
            [("加拿大政策", "加拿大移民政策变化"), ("烹饪", "面包制作方法")],
        )

        self.assertGreater(relevant, unrelated)
        self.assertGreater(relevant, 0.5)

    def test_stopword_only_query_and_empty_documents_score_zero(self):
        self.assertEqual(lexical_relevance_scores("the and of", [("title", "body")]), [0.0])
        self.assertEqual(lexical_relevance_scores("query", []), [])

    def test_scores_are_deterministic(self):
        documents = [("Title", "alpha beta gamma")]
        self.assertEqual(
            lexical_relevance_scores("alpha beta", documents),
            lexical_relevance_scores("alpha beta", documents),
        )


class TestLexicalCoverage(unittest.TestCase):
    def test_reports_union_coverage_across_evidence(self):
        coverage = lexical_query_coverage(
            "alpha beta gamma", ["alpha appears here", "gamma appears there"]
        )

        self.assertEqual(coverage.query_terms, 3)
        self.assertEqual(coverage.matched_terms, 2)
        self.assertEqual(coverage.score, 0.666667)

    def test_stopword_only_query_has_zero_coverage(self):
        self.assertEqual(
            lexical_query_coverage("the and of", ["the evidence"]),
            lexical_query_coverage("", []),
        )


class TestFlashRankRelevance(unittest.TestCase):
    def test_scores_are_aligned_to_input_order_and_ranker_is_cached(self):
        created = []

        class FakeRequest:
            def __init__(self, query, passages):
                self.query = query
                self.passages = passages

        class FakeRanker:
            def __init__(self, max_length):
                created.append(max_length)

            def rerank(self, request):
                if request.query != "target" or len(request.passages) != 2:
                    raise AssertionError("unexpected FlashRank request")
                return [
                    {"id": 1, "score": 0.9},
                    {"id": 0, "score": 0.2},
                ]

        fake_module = types.ModuleType("flashrank")
        fake_module.Ranker = FakeRanker
        fake_module.RerankRequest = FakeRequest
        documents = [("First", "noise"), ("Second", "target")]

        with mock.patch.dict(sys.modules, {"flashrank": fake_module}), \
             mock.patch.object(ranking, "_flashrank_ranker", None):
            first = flashrank_relevance_scores("target", documents)
            second = flashrank_relevance_scores("target", documents)

        self.assertEqual(first, [0.2, 0.9])
        self.assertEqual(second, first)
        self.assertEqual(created, [128])

    def test_incomplete_scores_are_rejected(self):
        fake_module = types.ModuleType("flashrank")
        fake_module.RerankRequest = lambda query, passages: object()
        ranker = mock.Mock()
        ranker.rerank.return_value = [{"id": 0, "score": 0.5}]

        with mock.patch.dict(sys.modules, {"flashrank": fake_module}), \
             mock.patch("sibyl.ranking._get_flashrank_ranker", return_value=ranker):
            with self.assertRaisesRegex(ValueError, "every document"):
                flashrank_relevance_scores("query", [("A", "a"), ("B", "b")])


if __name__ == "__main__":
    unittest.main()
