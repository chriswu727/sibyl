"""Deterministic deep-research gates and evidence-provenance regressions."""

import asyncio
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from research_fixtures import evidence_bundle

from sibyl.config import Config
from sibyl.mcp_server import create_server
from sibyl.report_models import ResearchReport
from sibyl.reporter import _report_to_markdown
from sibyl.research_review import research_plan, review_question
from sibyl.research_sources import read_source_bundle, read_sources
from sibyl.research_state import ResearchBudgetExceeded, ResearchNotebook
from sibyl.retrieval import gather_source_bundle
from sibyl.runtime import RuntimePolicy, SibylRuntime
from sibyl.scraper import WebPage, _extract_content
from sibyl.search import SearchResult
from sibyl.source_policy import normalize_domains, source_kind


class TestSourceSelection(unittest.TestCase):
    def test_period_matching_sources_survive_full_window_across_rounds(self):
        bundle = evidence_bundle(count=3)
        old, new, unknown = [
            replace(s, title=t)
            for s, t in zip(
                bundle.sources, ["2025 results", "2026 results", "Undated research"]
            )
        ]
        notebook = ResearchNotebook(
            "2026 market",
            synthesis_limit=1,
            bundles=[
                replace(bundle, sources=[new]),
                replace(bundle, sources=[old, unknown]),
            ],
        )
        self.assertEqual(notebook.selected_sources(), [new])
        review_question(
            notebook,
            "Q0",
            "supported",
            [old.evidence[0].citation_id],
            [],
            "Historical comparison",
            "Reviewed.",
        )
        self.assertEqual(notebook.selected_sources(), [old])

    def test_primary_sources_displace_background_across_questions(self):
        notebook = ResearchNotebook("market", synthesis_limit=5)
        for index in range(5):
            bundle = evidence_bundle(str(index), 2)
            sources = [
                replace(
                    bundle.sources[0],
                    url=f"https://en.wikipedia.org/wiki/{index}",
                    relevance_score=1,
                ),
                replace(
                    bundle.sources[1],
                    url=f"https://docs.vendor{index}.com/models",
                    relevance_score=0.1,
                ),
            ]
            notebook.bundles.append(replace(bundle, sources=sources))
        self.assertEqual(len(notebook.selected_sources()), 5)
        self.assertTrue(
            all(
                source_kind(s) == "documentation_candidate"
                for s in notebook.selected_sources()
            )
        )

    def test_preprint_and_metadata_are_not_peer_review_verdicts(self):
        source = evidence_bundle(count=1).sources[0]
        self.assertEqual(
            source_kind(replace(source, url="https://arxiv.org/abs/1234")), "preprint"
        )
        self.assertEqual(
            source_kind(replace(source, content_origin="crossref_api")),
            "scholarly_metadata",
        )
        self.assertEqual(
            source_kind(replace(source, url="https://openreview.net/forum?id=1")),
            "research_candidate",
        )

    def test_preferred_domain_cannot_promote_community_discussion(self):
        source = replace(
            evidence_bundle(count=1).sources[0],
            url="https://huggingface.co/team/model/discussions/1",
        )
        self.assertEqual(source_kind(source, ["huggingface.co"]), "community")

    def test_domain_matching_is_not_substring_matching(self):
        source = replace(
            evidence_bundle(count=1).sources[0], url="https://vendor.com.evil.org/page"
        )
        self.assertEqual(source_kind(source, ["vendor.com"]), "web")
        for value in ["https://vendor.com", "vendor.com/path", "localhost", "x..com"]:
            with self.assertRaises(ValueError):
                normalize_domains([value])

    def test_mailing_archives_and_qa_remain_community_on_preferred_domains(self):
        source = evidence_bundle(count=1).sources[0]
        for url, preferred in (
            ("https://www.postgresql.org/message-id/question", ["postgresql.org"]),
            ("https://stackoverflow.com/questions/1", ["stackoverflow.com"]),
            ("https://dba.stackexchange.com/questions/1", ["stackexchange.com"]),
        ):
            with self.subTest(url=url):
                self.assertEqual(
                    source_kind(replace(source, url=url), preferred), "community"
                )

    def test_reviewed_source_is_not_evicted_by_later_search(self):
        notebook = ResearchNotebook(
            "q", bundles=[evidence_bundle("early", 1)], synthesis_limit=2
        )
        ref = notebook.bundles[0].sources[0].evidence[0].citation_id
        review_question(
            notebook,
            "Q0",
            "supported",
            [ref],
            [],
            "Conclusion",
            "No contrary result found.",
        )
        notebook.bundles.append(evidence_bundle("later", 6))
        self.assertIn(
            ref,
            [p.citation_id for s in notebook.selected_sources() for p in s.evidence],
        )

    def test_duplicate_content_cannot_be_pinned_as_independent_sources(self):
        bundle = evidence_bundle(count=2)
        sources = [
            bundle.sources[0],
            replace(bundle.sources[1], content_hash=bundle.sources[0].content_hash),
        ]
        notebook = ResearchNotebook("q", bundles=[replace(bundle, sources=sources)])
        refs = [s.evidence[0].citation_id for s in sources]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            review_question(
                notebook,
                "Q0",
                "disputed",
                refs[:1],
                refs[1:],
                "Conflict",
                "Compared sources.",
            )
        self.assertFalse(notebook.reviews)

    def test_old_checkpoint_loads_without_new_required_fields(self):
        notebook = ResearchNotebook("q")
        data = notebook.to_dict()
        for key in ["require_review", "reviews", "source_policy", "preferred_domains"]:
            data.pop(key)
        restored = ResearchNotebook.from_dict(data)
        self.assertFalse(restored.require_review)
        self.assertEqual(restored.reviews, {})

    def test_explicit_question_link_overrides_query_text(self):
        notebook = ResearchNotebook(
            "q",
            sub_questions=["First", "Second"],
            attempts=[
                {
                    "query": "Second",
                    "question_id": "Q1",
                    "status": "ok",
                    "purpose": "explore",
                },
            ],
        )
        plan = research_plan(notebook)
        self.assertEqual([item["attempts"] for item in plan["questions"]], [1, 0])
        self.assertNotIn("Not separately searched: First", notebook.limitations())
        self.assertIn("Not separately searched: Second", notebook.limitations())


class TestExtraction(unittest.TestCase):
    def test_short_numbers_and_table_columns_survive(self):
        page = _extract_content(
            """<html><title>Prices</title><main>
        <table><tr><th><button>Model</button></th><th>A</th><th>B</th></tr>
        <tr><td>Input</td><td>$0.1</td><td>$2</td></tr>
        <tr><td rowspan="2">Output</td><td>1M</td><td>✓</td></tr></table>
        </main></html>""",
            "https://example.com",
            5000,
        )
        self.assertIn("| Model | A | B |", page.text)
        self.assertIn("| Input | $0.1 | $2 |", page.text)
        self.assertIn("rowspan=2", page.text)
        self.assertIn("1M", page.text)

    def test_main_is_not_replaced_by_tiny_related_article(self):
        page = _extract_content(
            "<main><div>Full research content and methods.</div><article>4 items</article></main>",
            "https://example.com",
            5000,
        )
        self.assertIn("Full research content", page.text)


class TestDirectEvidence(unittest.IsolatedAsyncioTestCase):
    async def test_failed_read_exposes_bounded_fetch_reason_without_evidence(self):
        page = WebPage("https://example.com", "", "", error="HTTP 403")
        with mock.patch(
            "sibyl.research_sources.scrape_url", mock.AsyncMock(return_value=page)
        ):
            bundle = await read_source_bundle(page.url, "q")
        notebook = ResearchNotebook("q")
        notebook.begin_search("q")
        notebook.finish_search("q", bundle)
        self.assertFalse(bundle.sources)
        self.assertEqual(bundle.error, "HTTP 403")
        self.assertEqual(notebook.attempts[0]["error"], "HTTP 403")

    async def test_read_preserves_hashes_offsets_and_does_not_declare_truth(self):
        text = "Methods and limitations. " * 200
        with mock.patch(
            "sibyl.research_sources.scrape_url",
            mock.AsyncMock(
                return_value=WebPage("https://arxiv.org/html/test", "Paper", text)
            ),
        ):
            bundle = await read_source_bundle("https://arxiv.org/html/test", "Methods?")
        self.assertEqual(bundle.status, "ok")
        self.assertEqual(bundle.diagnostics.evidence_sufficiency, "limited")
        source = bundle.sources[0]
        self.assertEqual(source.content_hash, hashlib.sha256(text.encode()).hexdigest())
        for passage in source.evidence:
            self.assertEqual(passage.text, text[passage.start_char : passage.end_char])

    async def test_thin_read_has_no_evidence(self):
        with mock.patch(
            "sibyl.research_sources.scrape_url",
            mock.AsyncMock(
                return_value=WebPage("https://example.com", "Paper", "4 items")
            ),
        ):
            bundle = await read_source_bundle("https://example.com", "q")
        self.assertEqual(bundle.status, "insufficient_evidence")
        self.assertFalse(bundle.sources)

    async def test_targeted_read_reaches_beyond_the_old_head_limit(self):
        text = (
            "General introductory background. " * 650
            + "\nToken retention\n"
            + "Late retries can arrive after token expiry; retention is not indefinite. "
            * 50
            + "\nOther considerations\n"
            + "Other implementation material. " * 650
        )
        page = WebPage(
            "https://example.com/retries",
            "Retries",
            text,
            extraction_warnings=("HTML text truncated to 60000 characters.",),
        )
        with mock.patch(
            "sibyl.research_sources.scrape_url", mock.AsyncMock(return_value=page)
        ) as read:
            first = await read_source_bundle(
                page.url, "token retention expiry late retries"
            )
            second = await read_source_bundle(page.url, "other implementation material")
        read.assert_awaited_with(
            page.url,
            max_chars=60000,
            extractor="bs4",
            jina_fallback=False,
            js_render=False,
        )
        evidence = first.sources[0].evidence
        self.assertTrue(
            any(
                "retention is not indefinite" in p.text and p.start_char > 16000
                for p in evidence
            )
        )
        self.assertLessEqual(sum(len(p.text) for p in evidence), 16000)
        self.assertLessEqual(len(evidence), 9)
        self.assertEqual(
            first.diagnostics.evidence_chars, sum(len(p.text) for p in evidence)
        )
        self.assertNotEqual(first.bundle_id, second.bundle_id)
        self.assertEqual(first.sources[0].content_hash, second.sources[0].content_hash)
        self.assertIn(
            page.extraction_warnings[0], first.diagnostics.sufficiency_reasons
        )
        self.assertTrue(
            any(
                "omitted sections remain unread" in reason
                for reason in first.diagnostics.sufficiency_reasons
            )
        )
        for passage in evidence:
            self.assertEqual(text[passage.start_char : passage.end_char], passage.text)
            self.assertEqual(
                hashlib.sha256(passage.text.encode()).hexdigest(), passage.content_hash
            )

    async def test_site_query_does_not_fetch_or_fallback_to_wikipedia(self):
        results = [
            SearchResult("Study", "https://docs.example.com/paper", "", "web"),
            SearchResult("Other", "https://en.wikipedia.org/wiki/Other", "", "web"),
        ]
        page = WebPage(
            results[0].url, "Study", "Research methods evidence limitations. " * 20
        )
        with (
            mock.patch(
                "sibyl.retrieval.search_web", mock.AsyncMock(return_value=results)
            ),
            mock.patch(
                "sibyl.retrieval.search_wikipedia", mock.AsyncMock()
            ) as entities,
            mock.patch(
                "sibyl.retrieval.wikipedia_lookup", mock.AsyncMock()
            ) as fallback,
            mock.patch(
                "sibyl.retrieval.scrape_urls", mock.AsyncMock(return_value=[page])
            ) as scrape,
        ):
            await gather_source_bundle(
                "site:docs.example.com Research", client=object()
            )
        entities.assert_not_called()
        fallback.assert_not_called()
        self.assertEqual(scrape.call_args.args[0], [page.url])


class TestHostSourceReads(unittest.IsolatedAsyncioTestCase):
    """Host-supplied URLs get the same assessment as discovered sources."""

    ORIGINAL = (
        "The SQLite write-ahead log allows readers to continue while one writer "
        "appends to the log. Only one writer may be active at a time. "
    ) * 20

    def pages(self, *items):
        by_url = {page.url: page for page in items}

        async def scrape(url, **kwargs):
            return by_url[url]

        return mock.patch("sibyl.research_sources.scrape_url", side_effect=scrape)

    async def test_batch_read_ranks_clusters_and_assesses_sources(self):
        original = WebPage("https://sqlite.org/wal.html", "WAL", self.ORIGINAL)
        copy = WebPage("https://mirror.example/wal", "WAL copy", self.ORIGINAL)
        other = WebPage(
            "https://blog.example/locks",
            "Locking",
            "Database locking and unrelated deployment notes. " * 30,
        )
        with self.pages(original, copy, other):
            bundle = await read_sources(
                [original.url, copy.url, other.url], "one writer write-ahead log"
            )
        self.assertEqual(bundle.status, "ok")
        self.assertEqual(len(bundle.sources), 3)
        clusters = {s.url: s.content_cluster_id for s in bundle.sources}
        self.assertEqual(clusters[original.url], clusters[copy.url])
        self.assertNotEqual(clusters[original.url], clusters[other.url])
        self.assertEqual(bundle.diagnostics.duplicate_sources, 1)
        self.assertEqual(bundle.diagnostics.independent_content_clusters, 2)
        self.assertTrue(all(s.relevance_score is not None for s in bundle.sources))
        self.assertNotEqual(bundle.sources[-1].url, original.url)
        self.assertIn(
            "Direct source read; host must assess support, freshness and completeness.",
            bundle.diagnostics.sufficiency_reasons,
        )
        for source in bundle.sources:
            page = {p.url: p for p in (original, copy, other)}[source.url]
            for passage in source.evidence:
                self.assertEqual(
                    passage.text, page.text[passage.start_char : passage.end_char]
                )

    async def test_single_read_reports_limited_sufficiency_honestly(self):
        page = WebPage("https://sqlite.org/wal.html", "WAL", self.ORIGINAL)
        with self.pages(page):
            bundle = await read_sources([page.url], "one writer")
        self.assertEqual(bundle.status, "ok")
        self.assertEqual(bundle.diagnostics.evidence_sufficiency, "limited")
        self.assertIn("single_domain", bundle.diagnostics.sufficiency_reasons)
        self.assertTrue(bundle.sources[0].content_cluster_id)

    async def test_partial_failure_keeps_successes_and_discloses_the_failure(self):
        good = WebPage("https://sqlite.org/wal.html", "WAL", self.ORIGINAL)
        blocked = WebPage("https://blocked.example/x", "", "", error="HTTP 403")
        with self.pages(good, blocked):
            bundle = await read_sources([good.url, blocked.url], "one writer")
        self.assertEqual(bundle.status, "ok")
        self.assertEqual([s.url for s in bundle.sources], [good.url])
        self.assertIn("HTTP 403", bundle.diagnostics.sufficiency_reasons)
        self.assertEqual(bundle.diagnostics.scrape_failures, 1)
        self.assertEqual(bundle.error, "")

    async def test_fact_chain_reads_keep_the_decomposition_safeguard(self):
        query = "In what year was the company that created CUDA founded?"
        first = WebPage(
            "https://a.example/cuda",
            "CUDA creator",
            "The company created CUDA as a parallel computing platform. " * 20,
        )
        second = WebPage(
            "https://b.example/history",
            "Company history",
            "The company was founded in 1993 and later created CUDA. " * 20,
        )
        with self.pages(first, second):
            bundle = await read_sources([first.url, second.url], query)
        self.assertEqual(bundle.status, "ok")
        self.assertEqual(bundle.diagnostics.query_complexity, "multi_step")
        self.assertEqual(bundle.diagnostics.evidence_sufficiency, "insufficient")
        self.assertEqual(bundle.diagnostics.recommended_action, "decompose_query")
        self.assertIn("multi_step_query", bundle.diagnostics.sufficiency_reasons)

    async def test_read_limits_are_enforced_before_fetching(self):
        with mock.patch("sibyl.research_sources.scrape_url") as scrape:
            for urls in ([], [f"https://e{i}.example" for i in range(6)]):
                with self.assertRaises(ValueError):
                    await read_sources(urls, "q")
        scrape.assert_not_called()


class TestDeepResearchWorkflow(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.policy = RuntimePolicy(output_root=Path(self.directory.name))
        self.runtime = SibylRuntime(self.policy, Config())
        self.jobs = self.runtime.research_jobs

    async def asyncTearDown(self):
        await self.runtime.aclose()
        self.directory.cleanup()

    def start(self, steps=4):
        return self.jobs.start(
            "Question", [], "host", 2, "en", steps, "academic", ["example.com"]
        )

    async def retrieve(self, job, query="support", purpose="explore", **kwargs):
        with mock.patch(
            "sibyl.research_jobs.gather_source_bundle",
            mock.AsyncMock(return_value=evidence_bundle(query, 1)),
        ) as search:
            result = await self.jobs.continue_job(
                job["job_id"], query, purpose=purpose, **kwargs
            )
        self.assertTrue(search.call_args.kwargs["include_academic"])
        return result

    def quote(self, job, passage_id):
        try:
            text = self.jobs.get(job["job_id"], passage_id)["passage"]["text"]
        except Exception:
            text = "Evidence"
        return {"passage_id": passage_id, "text": text.split(". ")[0]}

    def claims(self, job, status, support, challenge):
        """One atomic fact claim matching the question-level review."""
        if status == "unresolved":
            return [
                {
                    "claim_id": "Q0.C1",
                    "kind": "fact",
                    "status": "unresolved",
                    "text": "The answer could not be established.",
                    "rationale": "The retrieved evidence does not settle the question.",
                }
            ]
        return [
            {
                "claim_id": "Q0.C1",
                "kind": "fact",
                "status": status,
                "text": "The reviewed sources describe the measured result.",
                "supporting_passage_ids": support,
                "challenging_passage_ids": challenge,
                "quotes": [self.quote(job, pid) for pid in support[:1] + challenge[:1]],
            }
        ]

    def assess(
        self, job, status="supported", support=None, challenge=None, claims=True
    ):
        """claims=True builds a matching claim, False omits claims, a list is used."""
        support = (
            support if support is not None else [job["sources"][0]["passage_ids"][0]]
        )
        challenge = challenge or []
        if claims is True:
            claims = self.claims(job, status, support, challenge)
        return self.jobs.assess(
            job["job_id"],
            job["revision"],
            "Q0",
            status,
            support,
            challenge,
            "Conclusion with explicit scope.",
            "Examined possible contrary evidence.",
            claims=claims or None,
        )

    async def test_full_deep_workflow_survives_restart_exports_evidence(self):
        job = await self.retrieve(self.start())
        support = job["sources"][0]["passage_ids"][0]
        job = await self.retrieve(job, "replication limitations", "counterevidence")
        challenge = job["sources"][0]["passage_ids"][0]
        # Each source's report is a supported fact; the question stays disputed.
        claims = [
            {
                "claim_id": f"Q0.C{index}",
                "kind": "fact",
                "status": "supported",
                "text": text,
                "supporting_passage_ids": [passage],
                "quotes": [self.quote(job, passage)],
            }
            for index, (passage, text) in enumerate(
                (
                    (support, "The original study reports the effect."),
                    (challenge, "A replication reports a conflicting result."),
                ),
                1,
            )
        ]
        job = self.assess(job, "disputed", [support], [challenge], claims=claims)
        self.assertTrue(job["research_plan"]["ready_to_complete"])
        self.assertEqual(job["research_plan"]["next_action"], "synthesize")
        await self.runtime.aclose()
        self.runtime = SibylRuntime(self.policy, Config())
        self.jobs = self.runtime.research_jobs
        job = self.jobs.get(job["job_id"])
        self.assertTrue(job["research_plan"]["ready_to_complete"])
        job = self.jobs.complete(job["job_id"], job["revision"])
        report = ResearchReport.from_dict(job["report"])
        self.assertEqual(report.status, "ok")
        self.assertEqual(
            [claim["text"] for claim in report.claims],
            [claim["text"] for claim in claims],
        )
        self.assertIn("[Source 1]", report.summary)
        self.assertIn("[Source 2]", report.summary)
        self.assertEqual(
            report.research_plan["questions"][0]["review"]["status"], "disputed"
        )
        self.assertTrue(
            all(source.evidence and source.content_hash for source in report.sources)
        )
        md = _report_to_markdown(report)
        self.assertIn("Evidence-linked conclusions", md)
        self.assertIn('Quote [Source 1]: "Evidence replication limitations 0"', md)
        self.assertIn(challenge, md)
        self.assertNotIn("Evidence support 0. Evidence support", md)

    async def test_completion_with_open_gates_requires_explicit_incomplete_flag(self):
        job = await self.retrieve(self.start())
        with self.assertRaisesRegex(ValueError, "claim IDs"):
            self.jobs.complete(job["job_id"], job["revision"], allow_incomplete=True)
        # Reviewed, but no counterevidence round was run.
        job = self.assess(job)
        self.assertFalse(job["research_plan"]["ready_to_complete"])
        with self.assertRaisesRegex(ValueError, "gaps"):
            self.jobs.complete(job["job_id"], job["revision"])
        self.assertIsNone(self.jobs.store.get(job["job_id"])["report"])
        result = self.jobs.complete(
            job["job_id"], job["revision"], allow_incomplete=True
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["report"]["status"], "partial")
        md = _report_to_markdown(ResearchReport.from_dict(result["report"]))
        self.assertIn("Partial report", md)
        self.assertIn("The reviewed sources describe the measured result.", md)
        self.assertIn("[Source 1]", md)
        self.assertFalse(result["report"]["research_plan"]["ready_to_complete"])
        self.assertTrue(
            any("counterevidence" in s for s in result["report"]["limitations"])
        )

    async def test_new_evidence_invalidates_review(self):
        job = await self.retrieve(self.start(), purpose="counterevidence")
        job = self.assess(job)
        self.assertTrue(job["research_plan"]["ready_to_complete"])
        job = await self.retrieve(job, "new evidence")
        self.assertFalse(job["research_plan"]["ready_to_complete"])
        self.assertTrue(
            any(
                "New retrieval" in s
                for s in job["research_plan"]["questions"][0]["gaps"]
            )
        )

    async def test_review_drives_next_round_across_restart_and_export(self):
        server = create_server(self.runtime)
        job = await self.retrieve(self.start(3))
        job = await self.structured(
            server,
            "assess_research",
            {
                "job_id": job["job_id"],
                "revision": job["revision"],
                "question_id": "Q0",
                "status": "unresolved",
                "conclusion": "The replication evidence is missing.",
                "counterevidence_note": "The initial source only describes the original experiment.",
                "follow_up_queries": ["independent replication methods"],
            },
        )
        self.assertEqual(job["view"], "update")
        self.assertEqual(
            job["suggested_queries"][0]["query"],
            "independent replication methods",
        )
        await self.runtime.aclose()
        self.runtime = SibylRuntime(self.policy, Config())
        self.jobs = self.runtime.research_jobs
        server = create_server(self.runtime)
        with mock.patch(
            "sibyl.research_jobs.gather_source_bundle",
            mock.AsyncMock(return_value=evidence_bundle("replication", 1)),
        ):
            updated = await self.structured(
                server,
                "continue_research",
                {
                    "job_id": job["job_id"],
                    "query": "independent replication methods",
                    "question_id": "Q0",
                    "purpose": "counterevidence",
                    "rationale": "Resolve the missing independent replication evidence.",
                },
            )
        step = updated["attempt"]
        updated = self.jobs.get(job["job_id"])
        self.assertEqual(updated["attempts"][-1], step)
        self.assertEqual(step["round"], 2)
        self.assertEqual(step["based_on_revision"], job["revision"])
        self.assertEqual(step["prior_review_status"], "unresolved")
        self.assertTrue(step["prior_gaps"])
        self.assertIn("replication", step["rationale"])
        self.assertNotIn(
            "independent replication methods",
            [s["query"] for s in updated["research_plan"]["suggested_queries"]],
        )
        updated = self.assess(updated)
        self.assertTrue(updated["research_plan"]["ready_to_complete"])
        self.assertEqual(updated["research_plan"]["suggested_queries"], [])
        result = self.jobs.complete(updated["job_id"], updated["revision"])
        self.assertEqual(result["report"]["status"], "ok")
        self.assertEqual(result["report"]["research_stats"]["steps"][-1], step)

    async def test_follow_up_input_limits_do_not_mutate_job(self):
        job = self.start()
        for queries in [[""], ["x" * 1001], ["a", "b", "c", "d", "e"]]:
            with self.assertRaisesRegex(ValueError, "follow-up"):
                self.jobs.assess(
                    job["job_id"],
                    job["revision"],
                    "Q0",
                    "unresolved",
                    [],
                    [],
                    "Missing evidence",
                    "No counterevidence yet",
                    queries,
                )
        with self.assertRaisesRegex(ValueError, "rationale"):
            await self.jobs.continue_job(job["job_id"], "q", rationale="x" * 1001)
        current = self.jobs.get(job["job_id"])
        self.assertEqual(current["revision"], job["revision"])
        self.assertEqual(current["attempts"], [])

    async def test_failed_follow_up_remains_retryable_but_budget_exhaustion_stops_suggestions(
        self,
    ):
        job = self.start(2)
        job = self.jobs.assess(
            job["job_id"],
            job["revision"],
            "Q0",
            "unresolved",
            [],
            [],
            "Missing paper",
            "Need replication",
            ["replication study"],
        )
        with mock.patch(
            "sibyl.research_jobs.gather_source_bundle",
            mock.AsyncMock(side_effect=RuntimeError("failed")),
        ):
            job = await self.jobs.continue_job(job["job_id"], "replication study")
            self.assertEqual(
                job["research_plan"]["suggested_queries"][0]["query"],
                "replication study",
            )
            job = await self.jobs.continue_job(job["job_id"], "replication study")
        self.assertEqual(job["research_plan"]["suggested_queries"], [])
        self.assertEqual(
            job["research_plan"]["next_action"], "review_or_report_limitations"
        )
        self.assertFalse(job["research_plan"]["ready_to_complete"])

    async def test_review_rejects_stale_revision_unknown_reference_and_false_conflict(
        self,
    ):
        initial = self.start()
        job = await self.retrieve(initial)
        for claims in (False, True):
            with self.subTest(claims=claims):
                with self.assertRaisesRegex(ValueError, "revision"):
                    self.assess(
                        initial,
                        support=[job["sources"][0]["passage_ids"][0]],
                        claims=claims,
                    )
                with self.assertRaisesRegex(ValueError, "unknown"):
                    self.assess(job, support=["invented/P1"], claims=claims)
                with self.assertRaisesRegex(ValueError, "both"):
                    self.assess(job, "disputed", claims=claims)
        self.assertEqual(self.jobs.get(job["job_id"])["revision"], job["revision"])

    async def test_metadata_or_wiki_only_does_not_pass_primary_gate(self):
        job = self.start()
        bundle = evidence_bundle(count=1)
        source = replace(bundle.sources[0], content_origin="crossref_api")
        with mock.patch(
            "sibyl.research_jobs.gather_source_bundle",
            mock.AsyncMock(return_value=replace(bundle, sources=[source])),
        ):
            job = await self.jobs.continue_job(
                job["job_id"], "query", purpose="counterevidence"
            )
        job = self.assess(job)
        self.assertFalse(job["research_plan"]["ready_to_complete"])

    async def test_direct_read_shares_budget_and_retains_lineage(self):
        job = await self.retrieve(self.start(2))
        parent = job["sources"][0]["passage_ids"][0]
        with mock.patch(
            "sibyl.research_jobs.read_sources",
            mock.AsyncMock(return_value=evidence_bundle("paper", 1)),
        ) as read:
            job = await self.jobs.continue_job(
                job["job_id"],
                "methods",
                url="https://example.com/paper",
                purpose="follow_citation",
                parent_passage_id=parent,
            )
        read.assert_awaited_once()
        self.assertEqual(job["remaining_steps"], 0)
        self.assertEqual(job["attempts"][-1]["parent_passage_id"], parent)
        self.assertTrue(self.jobs.get(job["job_id"], parent)["passage"]["text"])
        with self.assertRaises(ResearchBudgetExceeded):
            await self.jobs.continue_job(
                job["job_id"], "another", url="https://example.com/new"
            )

    async def test_failed_direct_read_does_not_pass_counterevidence_gate(self):
        job = self.start()
        with mock.patch(
            "sibyl.research_jobs.read_sources",
            mock.AsyncMock(side_effect=ValueError("secret")),
        ):
            job = await self.jobs.continue_job(
                job["job_id"],
                "counter",
                url="http://127.0.0.1",
                purpose="counterevidence",
            )
        self.assertEqual(job["remaining_steps"], 3)
        self.assertNotIn("secret", json.dumps(job))
        self.assertTrue(
            any(
                "counterevidence" in s
                for s in job["research_plan"]["questions"][0]["gaps"]
            )
        )

    async def test_unresolved_review_cannot_silently_pass(self):
        job = await self.retrieve(self.start(), purpose="counterevidence")
        job = self.assess(job, "unresolved", [])
        self.assertFalse(job["research_plan"]["ready_to_complete"])

    async def structured(self, server, name, arguments, error=False):
        response = await server.call_tool(name, arguments)
        self.assertEqual(bool(response.isError), error, str(response))
        self.assertEqual(
            json.loads(response.content[0].text), response.structuredContent
        )
        return response.structuredContent

    async def test_mcp_assessment_tool_returns_structured_plan(self):
        server = create_server(self.runtime)
        job = await self.retrieve(self.start(), purpose="counterevidence")
        support = [job["sources"][0]["passage_ids"][0]]
        response = await self.structured(
            server,
            "assess_research",
            {
                "job_id": job["job_id"],
                "revision": job["revision"],
                "question_id": "Q0",
                "status": "supported",
                "conclusion": "Bounded conclusion",
                "counterevidence_note": "Checked limitations",
                "supporting_passage_ids": support,
                "claims": self.claims(job, "supported", support, []),
            },
        )
        self.assertEqual(response["view"], "update")
        self.assertTrue(response["ready_to_complete"])
        self.assertEqual(response["open_questions"], [])
        self.assertNotIn("retrieved", response)
        self.assertEqual(response["question"]["review"]["status"], "supported")
        material = response["answer_material"]
        self.assertIn("never Sibyl claim or passage IDs", material["guidance"])
        claim = material["claims"][0]
        self.assertEqual(claim["status"], "supported")
        self.assertTrue(claim["supporting_urls"][0].startswith("https://"))
        self.assertNotIn(support[0], json.dumps(material))
        self.assertEqual(material["counterevidence"]["status"], "linked")
        self.assertEqual(material["open_gaps"], [])
        self.assertEqual(response["before_answering"], [])

    async def test_mcp_retrieval_returns_new_passages_inline(self):
        server = create_server(self.runtime)
        job = self.start()
        with mock.patch(
            "sibyl.research_jobs.read_sources",
            mock.AsyncMock(return_value=evidence_bundle("inline", 2)),
        ):
            update = await self.structured(
                server,
                "continue_research",
                {
                    "job_id": job["job_id"],
                    "query": "inline",
                    "urls": ["https://example.com/a", "https://example.com/b"],
                },
            )
        self.assertEqual(update["view"], "update")
        self.assertEqual(update["attempt"]["status"], "ok")
        self.assertEqual(update["question"]["question_id"], "Q0")
        first = update["retrieved"][0]
        self.assertEqual(first["citation"], "[Source 1]")
        self.assertEqual(first["passages"][0]["passage_id"], "inline/0/P1")
        self.assertTrue(first["passages"][0]["text"].startswith("Evidence inline 0."))
        self.assertFalse(first["passages"][0]["truncated"])
        self.assertEqual(first["omitted_passage_ids"], [])
        full = self.jobs.passages(job["job_id"], ["inline/0/P1"])["passages"][0]
        self.assertTrue(
            full["passage"]["text"].startswith(first["passages"][0]["text"])
        )
        self.assertNotIn("sources", update)
        self.assertNotIn("answer_material", update)
        steps = " ".join(update["before_answering"])
        self.assertIn('purpose="counterevidence"', steps)
        self.assertIn("urls=[...]", steps)

    def test_inline_passages_share_a_bounded_budget(self):
        from sibyl.research_views import INLINE_CHARS, INLINE_PASSAGES, _inline_sources

        retrieved = [
            {
                "source_id": f"S{source}",
                "passages": [
                    {"passage_id": f"S{source}/P{rank}", "text": "x" * 1000}
                    for rank in range(9)
                ],
            }
            for source in range(5)
        ]
        inline = _inline_sources(retrieved)
        shown = [p for source in inline for p in source["passages"]]
        self.assertEqual(len(shown), INLINE_PASSAGES)
        self.assertLessEqual(sum(len(p["text"]) for p in shown), INLINE_CHARS)
        # Round-robin: every source shows its best passage first.
        self.assertTrue(
            all(s["passages"][0]["passage_id"].endswith("P0") for s in inline)
        )
        self.assertEqual(len(inline[0]["omitted_passage_ids"]), 6)

    async def test_mcp_gaps_and_stale_revision_are_actionable(self):
        server = create_server(self.runtime)
        job = await self.retrieve(self.start())
        job = self.assess(job)
        gaps = await self.structured(
            server,
            "complete_research",
            {"job_id": job["job_id"], "revision": job["revision"]},
            error=True,
        )
        self.assertEqual(gaps["error"]["code"], "research_gaps")
        self.assertIn("Inspect research_plan", gaps["error"]["message"])
        self.assertEqual(gaps["recovery"]["action"], "inspect_gaps")
        self.assertEqual(gaps["current_revision"], job["revision"])
        self.assertIsNone(self.jobs.store.get(job["job_id"])["report"])
        stale = await self.structured(
            server,
            "assess_research",
            {
                "job_id": job["job_id"],
                "revision": job["revision"] - 1,
                "question_id": "Q0",
                "status": "supported",
                "conclusion": "Conclusion",
                "counterevidence_note": "Checked limits",
            },
            error=True,
        )
        self.assertEqual(stale["error"]["code"], "stale_revision")
        self.assertIn("current revision", stale["error"]["message"])
        self.assertEqual(stale["current_revision"], job["revision"])
        self.assertEqual(stale["recovery"]["action"], "reload_research")

    async def test_batch_read_is_one_step_with_every_url_recorded(self):
        self.runtime._config = Config(extractor="trafilatura")
        job = self.start(2)
        urls = ["https://example.com/a?token=SECRET", "https://example.com/b"]
        with mock.patch(
            "sibyl.research_jobs.read_sources",
            mock.AsyncMock(return_value=evidence_bundle("batch", 2)),
        ) as read:
            job = await self.jobs.continue_job(job["job_id"], "batch", urls=urls)
        read.assert_awaited_once()
        self.assertEqual(read.call_args.args[0], urls)
        self.assertEqual(read.call_args.kwargs["extractor"], "trafilatura")
        attempt = job["attempts"][-1]
        self.assertEqual(job["remaining_steps"], 1)
        self.assertEqual(len(attempt["urls"]), 2)
        self.assertNotIn("SECRET", json.dumps(attempt))
        self.assertEqual(attempt["url"], attempt["urls"][0])

    async def test_too_many_urls_are_rejected_without_an_attempt(self):
        server = create_server(self.runtime)
        job = self.start()
        with mock.patch("sibyl.research_jobs.read_sources") as read:
            response = await server.call_tool(
                "continue_research",
                {
                    "job_id": job["job_id"],
                    "query": "q",
                    "urls": [f"https://e{i}.example" for i in range(6)],
                },
            )
        read.assert_not_called()
        self.assertTrue(response.isError)
        self.assertEqual(response.structuredContent["error"]["field"], "urls")
        self.assertEqual(self.jobs.get(job["job_id"])["attempts"], [])

    async def test_cancel_direct_read_retains_attempt(self):
        entered = asyncio.Event()

        async def wait(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        job = self.start()
        with mock.patch("sibyl.research_jobs.read_sources", wait):
            task = asyncio.create_task(
                self.jobs.continue_job(job["job_id"], "q", url="https://example.com")
            )
            await entered.wait()
            result = await self.jobs.cancel(job["job_id"])
        self.assertTrue(task.cancelled())
        self.assertEqual(result["remaining_steps"], 3)
        self.assertFalse(self.jobs.tasks)


if __name__ == "__main__":
    unittest.main()
