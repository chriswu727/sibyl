"""Deterministic next-search guidance derived from recorded research state."""

import unittest
from dataclasses import replace
from datetime import datetime, timezone

from research_fixtures import evidence_bundle

from sibyl.research_review import research_plan
from sibyl.research_state import ResearchNotebook

YEAR = datetime.now(timezone.utc).year


def notebook(question="How does SQLite WAL handle concurrent writers?", **kwargs):
    return ResearchNotebook(
        question, require_review=True, claim_contract_version=2, **kwargs
    )


def read(book, bundle, question_id="Q0", purpose="explore", url=""):
    book.begin_search(bundle.query)
    book.attempts[-1].update(question_id=question_id, purpose=purpose, url=url)
    book.finish_search(bundle.query, bundle)


def rehost(bundle, host):
    """The same fixture bundle with every source moved to another host."""
    return replace(
        bundle,
        sources=[
            replace(s, url=s.url.replace("example.com", host)) for s in bundle.sources
        ],
    )


def suggestions(book):
    return research_plan(book)["suggested_queries"]


def kinds(book):
    return [s["kind"] for s in suggestions(book)]


class TestSearchGuidance(unittest.TestCase):
    def test_unread_question_gets_a_search_box_query_with_a_reason(self):
        book = notebook()
        first = suggestions(book)[0]
        self.assertEqual(first["kind"], "initial")
        self.assertEqual(first["query"], "SQLite WAL handle concurrent writers")
        self.assertIn("No source has been read", first["reason"])
        self.assertIn("counterevidence", kinds(book))

    def test_unreadable_results_are_not_called_background_sources(self):
        book = notebook()
        read(book, evidence_bundle("blocked", 0))
        first = suggestions(book)[0]
        self.assertEqual(first["kind"], "initial")
        self.assertIn("no readable source", first["reason"])
        self.assertNotIn("primary_source", kinds(book))

    def test_background_only_evidence_asks_for_the_primary_record(self):
        book = notebook()
        read(book, rehost(evidence_bundle("background", 2), "en.wikipedia.org"))
        primary = next(s for s in suggestions(book) if s["kind"] == "primary_source")
        self.assertEqual(
            primary["query"], "SQLite WAL handle concurrent writers official source"
        )
        self.assertIn("background sources", primary["reason"])
        academic = notebook(source_policy="academic")
        read(academic, rehost(evidence_bundle("background", 2), "en.wikipedia.org"))
        self.assertTrue(
            any(s["query"].endswith("original paper") for s in suggestions(academic))
        )
        preferred = notebook(preferred_domains=["sqlite.org"])
        read(preferred, rehost(evidence_bundle("background", 2), "en.wikipedia.org"))
        self.assertTrue(
            any(s["query"].endswith("site:sqlite.org") for s in suggestions(preferred))
        )

    def test_single_domain_evidence_asks_for_an_independent_source(self):
        book = notebook()
        read(book, evidence_bundle("docs", 2))
        independent = next(
            s for s in suggestions(book) if s["kind"] == "independent_source"
        )
        self.assertTrue(independent["query"].endswith("-site:example.com"))
        self.assertIn("example.com", independent["reason"])
        self.assertNotIn("primary_source", kinds(book))
        read(book, rehost(evidence_bundle("other", 1), "other.example"))
        self.assertNotIn("independent_source", kinds(book))

    def test_counterevidence_is_suggested_until_one_is_read(self):
        book = notebook()
        read(book, evidence_bundle("docs", 1))
        counter = next(s for s in suggestions(book) if s["kind"] == "counterevidence")
        self.assertEqual(counter["purpose"], "counterevidence")
        read(book, evidence_bundle("counter", 1), purpose="counterevidence")
        self.assertNotIn("counterevidence", kinds(book))
        read(book, evidence_bundle("failed", 0), purpose="counterevidence")
        self.assertNotIn("counterevidence", kinds(book))

    def test_current_questions_ask_for_dated_sources(self):
        book = notebook("What is the latest SQLite release?")
        read(book, evidence_bundle("old", 1))
        recency = next(s for s in suggestions(book) if s["kind"] == "recency")
        self.assertTrue(recency["query"].endswith(str(YEAR)))
        dated = notebook("What is the latest SQLite release?")
        bundle = evidence_bundle("new", 1)
        bundle = replace(
            bundle,
            sources=[replace(bundle.sources[0], published_at=f"{YEAR}-01-02")],
        )
        read(dated, bundle)
        self.assertNotIn("recency", kinds(dated))
        self.assertNotIn("recency", kinds(notebook()))

    def test_host_follow_ups_come_first_and_are_not_repeated(self):
        book = notebook()
        read(book, evidence_bundle("docs", 1))
        book.reviews["Q0"] = {
            "question_id": "Q0",
            "status": "unresolved",
            "conclusion": "Open",
            "supporting_passage_ids": [],
            "challenging_passage_ids": [],
            "counterevidence_note": "None yet",
            "follow_up_queries": ["wal checkpoint starvation"],
            "reviewed_attempts": 1,
            "reviewed_revision": 1,
        }
        first = suggestions(book)[0]
        self.assertEqual(
            (first["kind"], first["query"]),
            ("host_follow_up", "wal checkpoint starvation"),
        )
        read(book, evidence_bundle("wal checkpoint starvation", 1))
        self.assertNotIn(
            "wal checkpoint starvation", [s["query"] for s in suggestions(book)]
        )

    def test_at_most_three_per_question_and_none_without_steps(self):
        book = notebook("What is the latest SQLite release?")
        read(book, rehost(evidence_bundle("background", 1), "en.wikipedia.org"))
        self.assertEqual(len(suggestions(book)), 3)
        spent = notebook(max_steps=1)
        read(spent, evidence_bundle("docs", 1))
        self.assertEqual(suggestions(spent), [])


if __name__ == "__main__":
    unittest.main()
