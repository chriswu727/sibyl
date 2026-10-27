"""Claim-level source assessment: kinds, independence and weak-support flags."""

import unittest
from dataclasses import replace

from research_fixtures import evidence_bundle

from sibyl.claim_checks import names, numbers, possible_conflict, unmatched_names
from sibyl.research_claims import claim_gaps
from sibyl.research_review import research_plan
from sibyl.research_state import ResearchNotebook
from sibyl.source_policy import source_kind

TEXT = "In WAL mode SQLite allows only one writer at a time, since 2010. " * 8


def bundle_at(name, hosts, text=TEXT):
    """A fixture bundle whose sources live on the given hosts."""
    bundle = evidence_bundle(name, len(hosts))
    return replace(
        bundle,
        sources=[
            replace(
                source,
                url=f"https://{host}/{name}/{index}",
                evidence=[replace(source.evidence[0], text=text)],
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


def fact(passage_ids, text="SQLite allows only one writer at a time.", **fields):
    return {
        "claim_id": "Q0.C1",
        "kind": "fact",
        "status": "supported",
        "text": text,
        "supporting_passage_ids": passage_ids,
        "challenging_passage_ids": [],
        "basis_claim_ids": [],
        "rationale": "",
        "assumptions": [],
        "criteria": [],
        "quotes": [],
        **fields,
    }


def assess(book, passage_ids, status="supported", claims=None, question_id="Q0"):
    book.reviews[question_id] = {
        "question_id": question_id,
        "status": "supported",
        "conclusion": "One writer at a time.",
        "supporting_passage_ids": passage_ids,
        "challenging_passage_ids": [],
        "counterevidence_note": "None found.",
        "follow_up_queries": [],
        "reviewed_attempts": len(book.attempts),
        "reviewed_revision": 1,
        "claims": claims or [fact(passage_ids, status=status)],
    }
    return next(
        item
        for item in research_plan(book)["questions"]
        if item["question_id"] == question_id
    )


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


class TestClaimChecks(unittest.TestCase):
    def test_numbers_are_normalized(self):
        self.assertEqual(
            numbers("1,000 users in 2024; ２０２５ and 3.45.0"),
            {"1000", "2024", "2025", "3.45.0"},
        )
        self.assertEqual(numbers("version v3 or SQLite3"), set())
        self.assertEqual(numbers("1,234.56 and 1234.56"), {"1234.56"})
        self.assertEqual(numbers("该版本于2024年发布，共３个"), {"2024", "3"})

    def test_names_skip_plain_sentence_starts(self):
        self.assertEqual(
            names("Writers wait. SQLite's WAL mode lets Readers continue in Linux."),
            ["SQLite", "WAL", "Readers", "Linux"],
        )
        self.assertEqual(names("数据库只允许一个写入者。"), [])
        self.assertEqual(
            unmatched_names("PostgreSQL and Write-Ahead logs", ["write ahead logs"]),
            ["PostgreSQL"],
        )

    def test_possible_conflict_needs_same_subject_and_different_numbers(self):
        size = "The maximum SQLite database size is {} terabytes."
        self.assertTrue(possible_conflict(size.format(281), size.format(140)))
        self.assertFalse(possible_conflict(size.format(281), size.format("281")))
        self.assertFalse(
            possible_conflict(size.format(281), "Python 3.12 was released in 2023.")
        )
        self.assertFalse(
            possible_conflict(size.format(281), "The maximum SQLite database size.")
        )

    def test_unquoted_number_blocks_completion(self):
        book = job(bundle_at("docs", ["sqlite.org"]))
        quote = {"passage_id": "docs/0/P1", "text": "SQLite allows only one writer"}
        claim = fact(
            ["docs/0/P1"],
            text="SQLite has allowed only one writer since 2009.",
            quotes=[quote],
        )
        item = assess(book, ["docs/0/P1"], claims=[claim])
        support = item["claim_support"][0]
        self.assertIn("number_not_quoted", support["flags"])
        self.assertEqual(support["unquoted_numbers"], ["2009"])
        self.assertTrue(
            any("2009 not found in the claim's quotes" in g for g in item["gaps"])
        )
        quote["text"] = "SQLite allows only one writer at a time, since 2010"
        claim["text"] = "SQLite has allowed only one writer since 2010."
        item = assess(book, ["docs/0/P1"], claims=[claim])
        self.assertNotIn("unquoted_numbers", item["claim_support"][0])
        self.assertFalse(
            any("not found in the claim's quotes" in g for g in item["gaps"])
        )

    def test_claims_without_quotes_skip_the_number_check(self):
        book = job(bundle_at("docs", ["sqlite.org"]))
        claim = fact(["docs/0/P1"], text="SQLite allows one writer since 2009.")
        support = assess(book, ["docs/0/P1"], claims=[claim])["claim_support"][0]
        self.assertNotIn("number_not_quoted", support["flags"])

    def test_names_missing_from_cited_text_are_flagged_without_blocking(self):
        book = job(bundle_at("docs", ["sqlite.org"]))
        claim = fact(["docs/0/P1"], text="PostgreSQL allows only one writer at a time.")
        item = assess(book, ["docs/0/P1"], claims=[claim])
        support = item["claim_support"][0]
        self.assertEqual(support["unmatched_names"], ["PostgreSQL"])
        self.assertIn("name_not_in_sources", support["flags"])
        self.assertFalse(any("PostgreSQL" in gap for gap in item["gaps"]))

    def test_facts_stating_different_numbers_are_cross_referenced(self):
        from sibyl.research_claims import render_claims

        book = job(bundle_at("docs", ["sqlite.org", "blog.example"]))
        size = "The maximum SQLite database size is {} terabytes."
        claims = [
            fact(["docs/0/P1"], text=size.format(281)),
            fact(["docs/1/P1"], text=size.format(140), claim_id="Q0.C2"),
        ]
        item = assess(book, ["docs/0/P1", "docs/1/P1"], claims=claims)
        first, second = item["claim_support"]
        self.assertEqual(first["conflicts_with"], ["Q0.C2"])
        self.assertEqual(second["conflicts_with"], ["Q0.C1"])
        self.assertIn("possible_conflict", second["flags"])
        _, _, analysis = render_claims(book, ["Q0.C1", "Q0.C2"])
        self.assertIn("Check: states different numbers than: Q0.C2", analysis)


class TestCounterevidenceSummary(unittest.TestCase):
    def test_status_follows_reads_and_links(self):
        book = job(bundle_at("docs", ["sqlite.org"]))
        item = assess(book, ["docs/0/P1"])
        self.assertEqual(item["counterevidence"]["status"], "not_read")
        counter = bundle_at("counter", ["blog.example"])
        book.begin_search(counter.query)
        book.attempts[-1].update(question_id="Q0", purpose="counterevidence")
        book.finish_search(counter.query, counter)
        item = assess(book, ["docs/0/P1"])
        self.assertEqual(
            item["counterevidence"],
            {"status": "none_linked", "reads": 1, "linked_passage_ids": []},
        )
        item = assess(book, ["docs/0/P1", "counter/0/P1"])
        self.assertEqual(item["counterevidence"]["status"], "linked")
        self.assertEqual(
            item["counterevidence"]["linked_passage_ids"], ["counter/0/P1"]
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
