"""Dependency-free lexical relevance tests."""
import unittest

from sibyl.ranking import lexical_relevance_scores


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


if __name__ == "__main__":
    unittest.main()
