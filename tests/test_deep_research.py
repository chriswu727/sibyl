"""Deterministic deep-research gates and evidence-provenance regressions."""

import asyncio
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from mcp.server.fastmcp.exceptions import ToolError

from sibyl.config import Config
from sibyl.mcp_server import create_server
from sibyl.reporter import _report_to_markdown
from sibyl.report_models import ResearchReport
from sibyl.research_review import research_plan, review_question
from sibyl.research_sources import read_source_bundle
from sibyl.research_state import ResearchBudgetExceeded, ResearchNotebook
from sibyl.retrieval import gather_source_bundle
from sibyl.runtime import RuntimePolicy, SibylRuntime
from sibyl.scraper import WebPage, _extract_content
from sibyl.search import SearchResult
from sibyl.source_policy import normalize_domains, source_kind
from test_researcher import evidence_bundle


class TestSourceSelection(unittest.TestCase):
    def test_primary_sources_displace_background_across_questions(self):
        notebook = ResearchNotebook("market", synthesis_limit=5)
        for index in range(5):
            bundle = evidence_bundle(str(index), 2)
            sources = [replace(bundle.sources[0], url=f"https://en.wikipedia.org/wiki/{index}", relevance_score=1),
                       replace(bundle.sources[1], url=f"https://docs.vendor{index}.com/models", relevance_score=0.1)]
            notebook.bundles.append(replace(bundle, sources=sources))
        self.assertEqual(len(notebook.selected_sources()), 5)
        self.assertTrue(all(source_kind(s) == "documentation_candidate" for s in notebook.selected_sources()))

    def test_preprint_and_metadata_are_not_peer_review_verdicts(self):
        source = evidence_bundle(count=1).sources[0]
        self.assertEqual(source_kind(replace(source, url="https://arxiv.org/abs/1234")), "preprint")
        self.assertEqual(source_kind(replace(source, content_origin="crossref_api")), "scholarly_metadata")
        self.assertEqual(source_kind(replace(source, url="https://openreview.net/forum?id=1")), "research_candidate")

    def test_preferred_domain_cannot_promote_community_discussion(self):
        source = replace(evidence_bundle(count=1).sources[0], url="https://huggingface.co/team/model/discussions/1")
        self.assertEqual(source_kind(source, ["huggingface.co"]), "community")

    def test_domain_matching_is_not_substring_matching(self):
        source = replace(evidence_bundle(count=1).sources[0], url="https://vendor.com.evil.org/page")
        self.assertEqual(source_kind(source, ["vendor.com"]), "web")
        for value in ["https://vendor.com", "vendor.com/path", "localhost", "x..com"]:
            with self.assertRaises(ValueError):
                normalize_domains([value])

    def test_reviewed_source_is_not_evicted_by_later_search(self):
        notebook = ResearchNotebook("q", bundles=[evidence_bundle("early", 1)], synthesis_limit=2)
        ref = notebook.bundles[0].sources[0].evidence[0].citation_id
        review_question(notebook, "Q0", "supported", [ref], [], "Conclusion", "No contrary result found.")
        notebook.bundles.append(evidence_bundle("later", 6))
        self.assertIn(ref, [p.citation_id for s in notebook.selected_sources() for p in s.evidence])

    def test_duplicate_content_cannot_be_pinned_as_independent_sources(self):
        bundle = evidence_bundle(count=2)
        sources = [bundle.sources[0], replace(bundle.sources[1], content_hash=bundle.sources[0].content_hash)]
        notebook = ResearchNotebook("q", bundles=[replace(bundle, sources=sources)])
        refs = [s.evidence[0].citation_id for s in sources]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            review_question(notebook, "Q0", "disputed", refs[:1], refs[1:], "Conflict", "Compared sources.")
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
        notebook = ResearchNotebook("q", sub_questions=["First", "Second"], attempts=[
            {"query": "Second", "question_id": "Q1", "status": "ok", "purpose": "explore"},
        ])
        plan = research_plan(notebook)
        self.assertEqual([item["attempts"] for item in plan["questions"]], [1, 0])
        self.assertNotIn("Not separately searched: First", notebook.limitations())
        self.assertIn("Not separately searched: Second", notebook.limitations())


class TestExtraction(unittest.TestCase):
    def test_short_numbers_and_table_columns_survive(self):
        page = _extract_content('''<html><title>Prices</title><main>
        <table><tr><th><button>Model</button></th><th>A</th><th>B</th></tr>
        <tr><td>Input</td><td>$0.1</td><td>$2</td></tr>
        <tr><td rowspan="2">Output</td><td>1M</td><td>✓</td></tr></table>
        </main></html>''', "https://example.com", 5000)
        self.assertIn("| Model | A | B |", page.text)
        self.assertIn("| Input | $0.1 | $2 |", page.text)
        self.assertIn("rowspan=2", page.text)
        self.assertIn("1M", page.text)

    def test_main_is_not_replaced_by_tiny_related_article(self):
        page = _extract_content('<main><div>Full research content and methods.</div><article>4 items</article></main>',
                                "https://example.com", 5000)
        self.assertIn("Full research content", page.text)


class TestDirectEvidence(unittest.IsolatedAsyncioTestCase):
    async def test_read_preserves_hashes_offsets_and_does_not_declare_truth(self):
        text = "Methods and limitations. " * 200
        with mock.patch("sibyl.research_sources.scrape_url", mock.AsyncMock(return_value=WebPage("https://arxiv.org/html/test", "Paper", text))):
            bundle = await read_source_bundle("https://arxiv.org/html/test", "Methods?")
        self.assertEqual(bundle.status, "ok")
        self.assertEqual(bundle.diagnostics.evidence_sufficiency, "limited")
        source = bundle.sources[0]
        self.assertEqual(source.content_hash, hashlib.sha256(text.encode()).hexdigest())
        for passage in source.evidence:
            self.assertEqual(passage.text, text[passage.start_char:passage.end_char])

    async def test_thin_read_has_no_evidence(self):
        with mock.patch("sibyl.research_sources.scrape_url", mock.AsyncMock(return_value=WebPage("https://example.com", "Paper", "4 items"))):
            bundle = await read_source_bundle("https://example.com", "q")
        self.assertEqual(bundle.status, "insufficient_evidence")
        self.assertFalse(bundle.sources)

    async def test_site_query_does_not_fetch_or_fallback_to_wikipedia(self):
        results = [SearchResult("Study", "https://docs.example.com/paper", "", "web"),
                   SearchResult("Other", "https://en.wikipedia.org/wiki/Other", "", "web")]
        page = WebPage(results[0].url, "Study", "Research methods evidence limitations. " * 20)
        with mock.patch("sibyl.retrieval.search_web", mock.AsyncMock(return_value=results)), \
             mock.patch("sibyl.retrieval.search_wikipedia", mock.AsyncMock()) as entities, \
             mock.patch("sibyl.retrieval.wikipedia_lookup", mock.AsyncMock()) as fallback, \
             mock.patch("sibyl.retrieval.scrape_urls", mock.AsyncMock(return_value=[page])) as scrape:
            await gather_source_bundle("site:docs.example.com Research", client=object())
        entities.assert_not_called()
        fallback.assert_not_called()
        self.assertEqual(scrape.call_args.args[0], [page.url])


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
        return self.jobs.start("Question", [], "host", 2, "en", steps, "academic", ["example.com"])

    async def retrieve(self, job, query="support", purpose="explore", **kwargs):
        with mock.patch("sibyl.research_jobs.gather_source_bundle", mock.AsyncMock(return_value=evidence_bundle(query, 1))) as search:
            result = await self.jobs.continue_job(job["job_id"], query, purpose=purpose, **kwargs)
        self.assertTrue(search.call_args.kwargs["include_academic"])
        return result

    def assess(self, job, status="supported", support=None, challenge=None):
        return self.jobs.assess(job["job_id"], job["revision"], "Q0", status,
                               support if support is not None else [job["sources"][0]["passage_ids"][0]],
                               challenge or [], "Conclusion with explicit scope.", "Examined possible contrary evidence.")

    async def test_full_deep_workflow_survives_restart_exports_evidence(self):
        job = await self.retrieve(self.start())
        support = job["sources"][0]["passage_ids"][0]
        job = await self.retrieve(job, "replication limitations", "counterevidence")
        challenge = job["sources"][0]["passage_ids"][0]
        job = self.assess(job, "disputed", [support], [challenge])
        self.assertTrue(job["research_plan"]["ready_to_complete"])
        self.assertEqual(job["research_plan"]["next_action"], "synthesize")
        await self.runtime.aclose()
        self.runtime = SibylRuntime(self.policy, Config())
        self.jobs = self.runtime.research_jobs
        job = self.jobs.get(job["job_id"])
        self.assertTrue(job["research_plan"]["ready_to_complete"])
        job = self.jobs.complete(job["job_id"], job["revision"], "Conflicting results [Source 1] [Source 2]",
                                 ["Disagreement [Source 1] [Source 2]"], "", [])
        report = ResearchReport.from_dict(job["report"])
        self.assertEqual(report.research_plan["questions"][0]["review"]["status"], "disputed")
        self.assertTrue(all(source.evidence and source.content_hash for source in report.sources))
        md = _report_to_markdown(report)
        self.assertIn("Evidence Review", md)
        self.assertIn(challenge, md)
        self.assertNotIn("Evidence support 0. Evidence support", md)

    async def test_completion_without_reviews_requires_explicit_incomplete_flag(self):
        job = await self.retrieve(self.start())
        args = (job["job_id"], job["revision"], "S [Source 1]", ["F [Source 1]"], "", [])
        with self.assertRaisesRegex(ValueError, "gaps"):
            self.jobs.complete(*args)
        result = self.jobs.complete(*args, allow_incomplete=True)
        self.assertFalse(result["report"]["research_plan"]["ready_to_complete"])
        self.assertTrue(any("counterevidence" in s for s in result["report"]["limitations"]))

    async def test_new_evidence_invalidates_review(self):
        job = await self.retrieve(self.start(), purpose="counterevidence")
        job = self.assess(job)
        self.assertTrue(job["research_plan"]["ready_to_complete"])
        job = await self.retrieve(job, "new evidence")
        self.assertFalse(job["research_plan"]["ready_to_complete"])
        self.assertTrue(any("New retrieval" in s for s in job["research_plan"]["questions"][0]["gaps"]))

    async def test_review_drives_next_round_across_restart_and_export(self):
        server = create_server(self.runtime)
        job = await self.retrieve(self.start(3))
        _, job = await server.call_tool("assess_research", {
            "job_id": job["job_id"], "revision": job["revision"], "question_id": "Q0",
            "status": "unresolved", "conclusion": "The replication evidence is missing.",
            "counterevidence_note": "The initial source only describes the original experiment.",
            "follow_up_queries": ["independent replication methods"],
        })
        self.assertEqual(job["research_plan"]["suggested_queries"][0]["query"], "independent replication methods")
        await self.runtime.aclose()
        self.runtime = SibylRuntime(self.policy, Config())
        self.jobs = self.runtime.research_jobs
        server = create_server(self.runtime)
        with mock.patch("sibyl.research_jobs.gather_source_bundle", mock.AsyncMock(
                return_value=evidence_bundle("replication", 1))):
            _, updated = await server.call_tool("continue_research", {
                "job_id": job["job_id"], "query": "independent replication methods", "question_id": "Q0",
                "purpose": "counterevidence", "rationale": "Resolve the missing independent replication evidence.",
            })
        step = updated["attempts"][-1]
        self.assertEqual(step["round"], 2)
        self.assertEqual(step["based_on_revision"], job["revision"])
        self.assertEqual(step["prior_review_status"], "unresolved")
        self.assertTrue(step["prior_gaps"])
        self.assertIn("replication", step["rationale"])
        self.assertNotIn("independent replication methods",
                         [s["query"] for s in updated["research_plan"]["suggested_queries"]])
        updated = self.assess(updated)
        self.assertTrue(updated["research_plan"]["ready_to_complete"])
        self.assertEqual(updated["research_plan"]["suggested_queries"], [])
        result = self.jobs.complete(updated["job_id"], updated["revision"], "S [Source 1]",
                                    ["F [Source 1]"], "", [])
        self.assertEqual(result["report"]["research_stats"]["steps"][-1], step)

    async def test_follow_up_input_limits_do_not_mutate_job(self):
        job = self.start()
        for queries in [[""], ["x" * 1001], ["a", "b", "c", "d", "e"]]:
            with self.assertRaisesRegex(ValueError, "follow-up"):
                self.jobs.assess(job["job_id"], job["revision"], "Q0", "unresolved", [], [],
                                 "Missing evidence", "No counterevidence yet", queries)
        with self.assertRaisesRegex(ValueError, "rationale"):
            await self.jobs.continue_job(job["job_id"], "q", rationale="x" * 1001)
        current = self.jobs.get(job["job_id"])
        self.assertEqual(current["revision"], job["revision"])
        self.assertEqual(current["attempts"], [])

    async def test_failed_follow_up_remains_retryable_but_budget_exhaustion_stops_suggestions(self):
        job = self.start(2)
        job = self.jobs.assess(job["job_id"], job["revision"], "Q0", "unresolved", [], [],
                               "Missing paper", "Need replication", ["replication study"])
        with mock.patch("sibyl.research_jobs.gather_source_bundle", mock.AsyncMock(side_effect=RuntimeError("failed"))):
            job = await self.jobs.continue_job(job["job_id"], "replication study")
            self.assertEqual(job["research_plan"]["suggested_queries"][0]["query"], "replication study")
            job = await self.jobs.continue_job(job["job_id"], "replication study")
        self.assertEqual(job["research_plan"]["suggested_queries"], [])
        self.assertEqual(job["research_plan"]["next_action"], "review_or_report_limitations")
        self.assertFalse(job["research_plan"]["ready_to_complete"])

    async def test_review_rejects_stale_revision_unknown_reference_and_false_conflict(self):
        initial = self.start()
        job = await self.retrieve(initial)
        with self.assertRaisesRegex(ValueError, "revision"):
            self.assess(initial, support=[job["sources"][0]["passage_ids"][0]])
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.assess(job, support=["invented/P1"])
        with self.assertRaisesRegex(ValueError, "both"):
            self.assess(job, "disputed")

    async def test_metadata_or_wiki_only_does_not_pass_primary_gate(self):
        job = self.start()
        bundle = evidence_bundle(count=1)
        source = replace(bundle.sources[0], content_origin="crossref_api")
        with mock.patch("sibyl.research_jobs.gather_source_bundle", mock.AsyncMock(return_value=replace(bundle, sources=[source]))):
            job = await self.jobs.continue_job(job["job_id"], "query", purpose="counterevidence")
        job = self.assess(job)
        self.assertFalse(job["research_plan"]["ready_to_complete"])

    async def test_direct_read_shares_budget_and_retains_lineage(self):
        job = await self.retrieve(self.start(2))
        parent = job["sources"][0]["passage_ids"][0]
        with mock.patch("sibyl.research_jobs.read_source_bundle", mock.AsyncMock(return_value=evidence_bundle("paper", 1))) as read:
            job = await self.jobs.continue_job(job["job_id"], "methods", url="https://example.com/paper",
                                              purpose="follow_citation", parent_passage_id=parent)
        read.assert_awaited_once()
        self.assertEqual(job["remaining_steps"], 0)
        self.assertEqual(job["attempts"][-1]["parent_passage_id"], parent)
        self.assertTrue(self.jobs.get(job["job_id"], parent)["passage"]["text"])
        with self.assertRaises(ResearchBudgetExceeded):
            await self.jobs.continue_job(job["job_id"], "another", url="https://example.com/new")

    async def test_failed_direct_read_does_not_pass_counterevidence_gate(self):
        job = self.start()
        with mock.patch("sibyl.research_jobs.read_source_bundle", mock.AsyncMock(side_effect=ValueError("secret"))):
            job = await self.jobs.continue_job(job["job_id"], "counter", url="http://127.0.0.1", purpose="counterevidence")
        self.assertEqual(job["remaining_steps"], 3)
        self.assertNotIn("secret", json.dumps(job))
        self.assertTrue(any("counterevidence" in s for s in job["research_plan"]["questions"][0]["gaps"]))

    async def test_unresolved_review_cannot_silently_pass(self):
        job = await self.retrieve(self.start(), purpose="counterevidence")
        job = self.assess(job, "unresolved", [])
        self.assertFalse(job["research_plan"]["ready_to_complete"])

    async def test_mcp_assessment_tool_returns_structured_plan(self):
        server = create_server(self.runtime)
        job = await self.retrieve(self.start(), purpose="counterevidence")
        _, response = await server.call_tool("assess_research", {
            "job_id": job["job_id"], "revision": job["revision"], "question_id": "Q0",
            "status": "supported", "conclusion": "Bounded conclusion", "counterevidence_note": "Checked limitations",
            "supporting_passage_ids": [job["sources"][0]["passage_ids"][0]],
        })
        self.assertTrue(response["research_plan"]["ready_to_complete"])

    async def test_mcp_gaps_and_stale_revision_are_actionable(self):
        server = create_server(self.runtime)
        job = await self.retrieve(self.start())
        with self.assertRaisesRegex(ToolError, "Inspect research_plan"):
            await server.call_tool("complete_research", {
                "job_id": job["job_id"], "revision": job["revision"],
                "summary": "Summary [Source 1]", "key_findings": ["Finding [Source 1]"],
            })
        with self.assertRaisesRegex(ToolError, "current revision"):
            await server.call_tool("assess_research", {
                "job_id": job["job_id"], "revision": job["revision"] - 1, "question_id": "Q0",
                "status": "supported", "conclusion": "Conclusion", "counterevidence_note": "Checked limits",
            })

    async def test_cancel_direct_read_retains_attempt(self):
        entered = asyncio.Event()

        async def wait(*args):
            entered.set()
            await asyncio.Event().wait()

        job = self.start()
        with mock.patch("sibyl.research_jobs.read_source_bundle", wait):
            task = asyncio.create_task(self.jobs.continue_job(job["job_id"], "q", url="https://example.com"))
            await entered.wait()
            result = await self.jobs.cancel(job["job_id"])
        self.assertTrue(task.cancelled())
        self.assertEqual(result["remaining_steps"], 3)
        self.assertFalse(self.jobs.tasks)


if __name__ == "__main__":
    unittest.main()
