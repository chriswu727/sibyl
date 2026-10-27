"""Claim-level source assessment: kinds, independence and weak-support flags."""

import unittest
from dataclasses import replace

from research_fixtures import evidence_bundle

from sibyl.research_claims import claim_gaps
from sibyl.research_review import research_plan
from sibyl.research_state import ResearchNotebook
from sibyl.source_policy import source_kind


def bundle_at(name, hosts):
    """A fixture bundle whose sources live on the given hosts."""
    bundle = evidence_bundle(name, len(hosts))
    return replace(
        bundle,
        sources=[
            replace(
                source,
                url=f"https://{host}/{name}/{index}",
            )
            for index, (source, host) in enumerate(zip(bundle.sources, hosts))
        ],
    )


def job(*bundles):
    book = ResearchNotebook(
        "Does SQLite allow one writer?", require_review=True, claim_contract_version=2
    )
    for bundle in bundles:
        book.begin_search(bundle.query)
        book.attempts[-1].update(question_id="Q0")
        book.finish_search(bundle.query, bundle)
    return book


def assess(book, passage_ids, status="supported"):
    book.reviews["Q0"] = {
        "question_id": "Q0",
        "status": "supported",
        "conclusion": "One writer at a time.",
        "supporting_passage_ids": passage_ids,
        "challenging_passage_ids": [],
        "counterevidence_note": "None found.",
        "follow_up_queries": [],
        "reviewed_attempts": len(book.attempts),
        "reviewed_revision": 1,
        "claims": [
            {
                "claim_id": "Q0.C1",
                "kind": "fact",
                "status": status,
                "text": "SQLite allows only one writer at a time.",
                "supporting_passage_ids": passage_ids,
                "challenging_passage_ids": [],
                "basis_claim_ids": [],
                "rationale": "",
                "assumptions": [],
                "criteria": [],
                "quotes": [],
            }
        ],
    }
    return research_plan(book)["questions"][0]


class TestClaimSupport(unittest.TestCase):
    def test_single_primary_source_is_flagged_but_does_not_block(self):
        book = job(bundle_at("docs", ["sqlite.org"]))
        item = assess(book, ["docs/0/P1"])
        support = item["claim_support"][0]
        self.assertEqual(support["sources"], 1)
        self.assertEqual(support["flags"], ["single_source"])
        self.assertFalse(any("Q0.C1" in gap for gap in claim_gaps(book, "Q0")))

    def test_independent_sources_have_no_flags(self):
        book = job(bundle_at("two", ["sqlite.org", "blog.example"]))
        support = assess(book, ["two/0/P1", "two/1/P1"])["claim_support"][0]
        self.assertEqual(support["sources"], 2)
        self.assertEqual(support["flags"], [])

    def test_background_only_support_blocks_completion(self):
        book = job(bundle_at("wiki", ["en.wikipedia.org"]))
        item = assess(book, ["wiki/0/P1"])
        support = item["claim_support"][0]
        self.assertIn("background_only", support["flags"])
        self.assertEqual(support["source_kinds"], ["encyclopedia"])
        self.assertTrue(
            any(
                "Q0.C1: support rests only on background sources" in gap
                for gap in item["gaps"]
            )
        )
        self.assertFalse(research_plan(book)["ready_to_complete"])

    def test_unresolved_facts_have_no_support_summary(self):
        book = job(bundle_at("docs", ["sqlite.org"]))
        self.assertEqual(assess(book, [], status="unresolved")["claim_support"], [])

    def test_single_source_facts_get_a_corroboration_search(self):
        book = job(bundle_at("docs", ["sqlite.org"]))
        assess(book, ["docs/0/P1"])
        suggestion = next(
            s
            for s in research_plan(book)["suggested_queries"]
            if s["kind"] == "corroboration"
        )
        self.assertEqual(
            suggestion["query"], "SQLite allows only one writer time -site:sqlite.org"
        )
        self.assertIn("Q0.C1", suggestion["reason"])

    def test_report_states_the_support_behind_each_fact(self):
        from sibyl.research_claims import render_claims

        book = job(bundle_at("docs", ["sqlite.org"]))
        assess(book, ["docs/0/P1"])
        records, _, analysis = render_claims(book, ["Q0.C1"])
        self.assertEqual(records[0]["source_support"]["sources"], 1)
        self.assertIn(
            "Source support: 1 source(s); "
            "kinds: documentation_candidate; "
            "flags: single_source",
            analysis,
        )


class TestSourceKinds(unittest.TestCase):
    def kind(self, url):
        return source_kind(
            evidence_bundle("k", 1)
            .sources[0]
            .__class__("S1", url, "t", "now", "h", "web", 1, [])
        )

    def test_institutional_hosts(self):
        for url in (
            "https://www.cdc.gov/x",
            "https://data.gov.uk/x",
            "https://www.mhlw.go.jp/x",
            "https://www.who.int/x",
            "https://gov.uk/x",
            "https://ec.europa.eu/x",
        ):
            self.assertEqual(self.kind(url), "institutional", url)
        for url in ("https://govtech.com/x", "https://gov.example.com/x"):
            self.assertEqual(self.kind(url), "web", url)


if __name__ == "__main__":
    unittest.main()
