"""Offline research lifecycle, persistence, resource and citation contracts."""

import asyncio
import hashlib
import json
import os
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from research_fixtures import evidence_bundle

from sibyl.async_cache import CapacityError
from sibyl.config import Config
from sibyl.job_store import JobStore
from sibyl.mcp_server import create_server
from sibyl.report_models import ResearchReport
from sibyl.research_state import ResearchBudgetExceeded, ResearchNotebook
from sibyl.runtime import RuntimePolicy, SibylRuntime


class TestNotebook(unittest.TestCase):
    def test_gap_evidence_enters_a_full_source_window(self):
        notebook = ResearchNotebook("q", synthesis_limit=3)
        notebook.bundles.append(evidence_bundle("initial", 6))
        notebook.bundles.append(evidence_bundle("gap", 2))
        selected = notebook.selected_sources()
        self.assertEqual(len(selected), 3)
        self.assertTrue(any("/gap/" in source.url for source in selected))
        self.assertTrue(any("/initial/" in source.url for source in selected))

    def test_duplicate_content_and_snippets_are_not_independent_evidence(self):
        bundle = evidence_bundle()
        notebook = ResearchNotebook(
            "q",
            bundles=[
                bundle,
                replace(bundle, bundle_id="another"),
                evidence_bundle("snippet", origin="search_snippet"),
            ],
        )
        self.assertEqual(len(notebook.selected_sources()), len(bundle.sources))

    def test_roundtrip_preserves_exact_evidence_and_citation_ids(self):
        notebook = ResearchNotebook("q", bundles=[evidence_bundle()])
        loaded = ResearchNotebook.from_dict(json.loads(json.dumps(notebook.to_dict())))
        self.assertEqual(notebook, loaded)
        self.assertEqual(notebook.report_sources(), loaded.report_sources())

    def test_budgets_do_not_reset_on_restore(self):
        notebook = ResearchNotebook(
            "q", max_steps=1, model_calls=1, reserved_output_tokens=100
        )
        notebook.begin_search("first")
        loaded = ResearchNotebook.from_dict(notebook.to_dict())
        with self.assertRaises(ResearchBudgetExceeded):
            loaded.begin_search("second")
        self.assertEqual(loaded.model_calls, 1)
        self.assertEqual(loaded.reserved_output_tokens, 100)


class TestResearchJobs(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.policy = RuntimePolicy(output_root=Path(self.directory.name))
        self.runtime = SibylRuntime(self.policy, Config())
        self.jobs = self.runtime.research_jobs

    async def asyncTearDown(self):
        await self.runtime.aclose()
        self.directory.cleanup()

    def start(self, **kwargs):
        return self.jobs.start(
            "Research question",
            ["Focused question"],
            "host",
            2,
            "en",
            **{"max_steps": 3, **kwargs},
        )

    async def retrieve(self, job):
        with mock.patch(
            "sibyl.research_jobs.gather_source_bundle",
            mock.AsyncMock(return_value=evidence_bundle("Focused question")),
        ):
            return await self.jobs.continue_job(job["job_id"], "Focused question")

    def assess(self, job, text="The source reports the focused evidence."):
        passage = job["sources"][0]["passage_ids"][0]
        return self.jobs.assess(
            job["job_id"],
            job["revision"],
            "Q1",
            "supported",
            [passage],
            [],
            "Bounded conclusion.",
            "No counterevidence round was run.",
            claims=[
                {
                    "claim_id": "Q1.C1",
                    "kind": "fact",
                    "status": "supported",
                    "text": text,
                    "supporting_passage_ids": [passage],
                    "quotes": [
                        {"passage_id": passage, "text": "Evidence Focused question 0"}
                    ],
                }
            ],
        )

    async def test_exact_passage_scope_and_budgeted_read_more_are_read_only(self):
        bundle = evidence_bundle("Focused question", 1)
        source = bundle.sources[0]
        passage = source.evidence[0]
        excerpt = passage.text[:80]
        passage = replace(
            passage,
            text=excerpt,
            end_char=passage.start_char + len(excerpt),
            content_hash=hashlib.sha256(excerpt.encode()).hexdigest(),
        )
        bundle = replace(bundle, sources=[replace(source, evidence=[passage])])
        for max_steps in (1, 3):
            job = self.start(max_steps=max_steps)
            with mock.patch(
                "sibyl.research_jobs.gather_source_bundle",
                mock.AsyncMock(return_value=bundle),
            ):
                job = await self.jobs.continue_job(job["job_id"], "Focused question")
            before = self.jobs.get(job["job_id"])
            view = self.jobs.get(job["job_id"], passage.citation_id)
            self.assertEqual(view["source_char_count"], source.char_count)
            self.assertTrue(view["is_excerpt"])
            self.assertEqual(view["passage"]["text"], excerpt)
            if max_steps == 1:
                self.assertIsNone(view["read_more"])
            else:
                self.assertEqual(view["read_more"]["tool"], "continue_research")
                self.assertEqual(view["read_more"]["url"], source.url)
                self.assertTrue(view["read_more"]["requires_focused_query"])
                self.assertTrue(view["read_more"]["uses_retrieval_step"])
            self.assertEqual(self.jobs.get(job["job_id"]), before)

    async def test_autonomous_start_rejected_without_creating_job(self):
        before = self.jobs.store.list_jobs()
        with self.assertRaisesRegex(ValueError, "no longer calls models"):
            self.jobs.start("q", [], "autonomous", 2, "en", 3)
        self.assertEqual(self.jobs.store.list_jobs(), before)

    async def test_legacy_autonomous_state_is_readable_but_cannot_resume(self):
        snapshot = await self.retrieve(self.start())
        job = self.jobs.store.get(snapshot["job_id"])
        job["mode"] = "autonomous"
        job["status"] = "interrupted"
        self.jobs.store.save(job)
        await self.runtime.aclose()
        self.runtime = SibylRuntime(self.policy)
        self.jobs = self.runtime.research_jobs
        before = self.jobs.get(job["job_id"])
        self.assertEqual(before["mode"], "autonomous")
        self.assertEqual(before["evidence"], snapshot["evidence"])
        with mock.patch("sibyl.research_jobs.gather_source_bundle") as gather:
            with self.assertRaisesRegex(ValueError, "removed"):
                await self.jobs.continue_job(job["job_id"], "resume")
            gather.assert_not_called()
        self.assertEqual(self.jobs.get(job["job_id"]), before)

    async def test_legacy_completed_autonomous_report_can_export(self):
        snapshot = self.assess(await self.retrieve(self.start()))
        self.jobs.complete(
            snapshot["job_id"], snapshot["revision"], allow_incomplete=True
        )
        job = self.jobs.store.get(snapshot["job_id"])
        job["mode"] = "autonomous"
        self.jobs.store.save(job)
        server = create_server(self.runtime)
        result = await server.call_tool("save_report", {"job_id": job["job_id"]})
        self.assertFalse(result.isError, str(result))
        self.assertEqual(result.structuredContent["status"], "saved")
        self.assertEqual(len(list(Path(self.directory.name).glob("*.md"))), 1)

    async def test_preferred_domains_reach_discovery(self):
        job = self.start(preferred_domains=["docs.example.com"])
        with mock.patch(
            "sibyl.research_jobs.gather_source_bundle",
            mock.AsyncMock(return_value=evidence_bundle("Focused question")),
        ) as gather:
            await self.jobs.continue_job(job["job_id"], "Focused question")
        self.assertEqual(
            gather.call_args.kwargs["preferred_domains"], ["docs.example.com"]
        )

    async def test_host_flow_survives_restart_and_exports_json(self):
        job = await self.retrieve(self.start())
        await self.runtime.aclose()
        self.runtime = SibylRuntime(self.policy, Config())
        self.jobs = self.runtime.research_jobs
        restored = self.jobs.get(job["job_id"])
        self.assertEqual(restored["evidence"], job["evidence"])
        restored = self.assess(restored)
        completed = self.jobs.complete(
            job["job_id"],
            restored["revision"],
            ["One question remains unresolved."],
            allow_incomplete=True,
        )
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(
            completed["report"]["verification_scope"], "citation_targets_only"
        )
        server = create_server(self.runtime)
        exported = await server.call_tool(
            "save_report", {"job_id": job["job_id"], "format": "json"}
        )
        self.assertFalse(exported.isError, str(exported))
        path = next(Path(self.directory.name).glob("*.json"))
        self.assertEqual(
            Path(exported.structuredContent["artifacts"][0]["path"]), path.resolve()
        )
        report = ResearchReport.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(report.status, "partial")
        self.assertIn("One question remains unresolved.", report.limitations)
        self.assertEqual(report.claims[0]["claim_id"], "Q1.C1")
        self.assertFalse(report.research_plan["ready_to_complete"])
        self.assertEqual(
            report.sources[0].passage_ids, completed["sources"][0]["passage_ids"]
        )
        if os.name == "posix":  # Windows has no POSIX modes; profile ACLs apply.
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        with self.assertRaises(ValueError):
            await self.jobs.continue_job(job["job_id"], "mutate completed")

    async def test_stale_revision_cannot_bind_claims_to_changed_citations(self):
        assessed = self.assess(await self.retrieve(self.start()))
        job = await self.retrieve(assessed)
        self.assertGreater(job["revision"], assessed["revision"])
        with self.assertRaisesRegex(ValueError, "revision"):
            self.jobs.complete(
                job["job_id"], assessed["revision"], allow_incomplete=True
            )
        self.assertIsNone(self.jobs.store.get(job["job_id"])["report"])

    async def test_unknown_claims_and_citations_are_rejected(self):
        job = self.assess(await self.retrieve(self.start()))
        before = self.jobs.store.get(job["job_id"])
        with self.assertRaisesRegex(ValueError, "Use passage IDs"):
            self.assess(job, "Unknown target [Source 999]")
        for claim_ids in (["Q1.C9"], [], ["Q1.C1", "Q9.C1"]):
            with (
                self.subTest(claim_ids=claim_ids),
                self.assertRaisesRegex(ValueError, "existing claim IDs"),
            ):
                self.jobs.complete(
                    job["job_id"],
                    job["revision"],
                    allow_incomplete=True,
                    claim_ids=claim_ids,
                )
        self.assertEqual(self.jobs.store.get(job["job_id"]), before)

    async def test_failed_step_uses_budget_and_reports_failure(self):
        job = self.start(max_steps=1)
        with mock.patch(
            "sibyl.research_jobs.gather_source_bundle",
            mock.AsyncMock(side_effect=RuntimeError("secret")),
        ):
            result = await self.jobs.continue_job(job["job_id"], "query")
        self.assertEqual(result["remaining_steps"], 0)
        self.assertEqual(result["attempts"][0]["status"], "failed")
        self.assertNotIn("secret", json.dumps(result))
        with self.assertRaises(ResearchBudgetExceeded):
            await self.jobs.continue_job(job["job_id"], "retry")

    async def test_cancel_interrupts_inflight_host_retrieval(self):
        entered = asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        job = self.start()
        with mock.patch("sibyl.research_jobs.gather_source_bundle", blocked):
            task = asyncio.create_task(self.jobs.continue_job(job["job_id"], "query"))
            await entered.wait()
            result = await self.jobs.cancel(job["job_id"])
        self.assertTrue(task.cancelled())
        self.assertEqual(result["status"], "cancelled")
        self.assertFalse(self.jobs.tasks)
        self.assertEqual(result["attempts"][0]["status"], "failed")

    async def test_oversized_checkpoint_retains_evidence_and_consumed_budget(self):
        initial = await self.retrieve(self.start())
        bundle = evidence_bundle("oversized")
        bundle.sources[0].evidence[0] = replace(
            bundle.sources[0].evidence[0], text="x" * (JobStore.MAX_PAYLOAD_BYTES + 1)
        )
        with mock.patch(
            "sibyl.research_jobs.gather_source_bundle",
            mock.AsyncMock(return_value=bundle),
        ):
            with self.assertRaisesRegex(CapacityError, "prior evidence"):
                await self.jobs.continue_job(initial["job_id"], "oversized")
        restored = self.jobs.get(initial["job_id"])
        self.assertEqual(restored["status"], "awaiting_input")
        self.assertEqual(restored["evidence"], initial["evidence"])
        self.assertEqual(restored["remaining_steps"], initial["remaining_steps"] - 1)
        self.assertEqual(restored["attempts"][-1]["status"], "failed")
        self.assertFalse(self.jobs.tasks)
        await self.runtime.aclose()
        self.runtime = SibylRuntime(self.policy, Config())
        self.jobs = self.runtime.research_jobs
        self.assertEqual(self.jobs.get(initial["job_id"])["status"], "awaiting_input")

    async def test_restart_marks_unfinished_attempt_failed_without_refunding_budget(
        self,
    ):
        initial = self.start()
        job = self.jobs.store.get(initial["job_id"])
        notebook = ResearchNotebook.from_dict(job["notebook"])
        notebook.begin_search("interrupted query")
        job.update(status="running", notebook=notebook.to_dict())
        self.jobs.store.save(job)
        await self.runtime.aclose()
        self.runtime = SibylRuntime(self.policy, Config())
        self.jobs = self.runtime.research_jobs
        restored = self.jobs.get(initial["job_id"])
        self.assertEqual(restored["status"], "interrupted")
        self.assertEqual(restored["attempts"][0]["status"], "failed")
        self.assertEqual(restored["remaining_steps"], 2)

    async def test_delete_reclaims_history_capacity(self):
        self.jobs.store.MAX_JOBS = 1
        job = self.start()
        with self.assertRaises(CapacityError):
            self.start()
        await self.jobs.cancel(job["job_id"], delete=True)
        self.assertEqual(len(self.jobs.store.list_jobs()), 0)
        self.start()

    async def test_nested_oversized_input_is_rejected_at_mcp_boundary(self):
        server = create_server(self.runtime)
        result = await server.call_tool(
            "start_research", {"question": "q", "sub_questions": ["x" * 10001]}
        )
        self.assertTrue(result.isError)
        self.assertEqual(result.structuredContent["view"], "error")
        self.assertIn("10000 characters", result.structuredContent["error"]["message"])
        self.assertNotIn("x" * 100, result.content[0].text)
        self.assertFalse(self.jobs.store.list_jobs())

    async def test_mcp_job_response_has_structured_content(self):
        server = create_server(self.runtime)
        result = await server.call_tool("start_research", {"question": "q"})
        self.assertFalse(result.isError)
        structured = result.structuredContent
        self.assertEqual(json.loads(result.content[0].text), structured)
        self.assertEqual(structured["status"], "awaiting_input")
        self.assertTrue(structured["job_id"])

    async def test_store_is_private_and_shared_by_concurrent_runtimes(self):
        store = self.jobs.store
        lock = store.instances / f"{store.instance_id}.lock"
        if os.name == "posix":  # Windows has no POSIX modes; profile ACLs apply.
            self.assertEqual(stat.S_IMODE(store.path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(lock.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(store.instances.stat().st_mode), 0o700)
        other = JobStore(self.policy.output_root)
        try:
            self.assertNotEqual(other.instance_id, store.instance_id)
            self.assertTrue(store.owner_alive(other.instance_id))
            self.assertTrue(other.owner_alive(store.instance_id))
        finally:
            other.close()
        self.assertFalse(store.owner_alive(other.instance_id))
        with store.connection() as connection:
            self.assertEqual(
                connection.execute("PRAGMA max_page_count").fetchone()[0], 16384
            )

    async def test_payload_overflow_preserves_prior_checkpoint(self):
        job = self.jobs.store.get(self.start()["job_id"])
        revision = job["revision"]
        job["oversized"] = "x" * (1024 * 1024)
        with self.assertRaises(CapacityError):
            self.jobs.store.save(job)
        loaded = self.jobs.store.get(job["job_id"])
        self.assertEqual(loaded["revision"], revision)
        self.assertNotIn("oversized", loaded)
