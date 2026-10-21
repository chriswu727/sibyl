"""Offline research lifecycle, persistence, resource and citation contracts."""

import asyncio
import json
import os
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from mcp.server.fastmcp.exceptions import ToolError

from sibyl.async_cache import CapacityError
from sibyl.config import Config, Provider
from sibyl.job_store import JobStore
from sibyl.mcp_server import configure_tool_profile, create_server
from sibyl.report_models import ResearchReport
from sibyl.research_state import ResearchBudgetExceeded, ResearchNotebook
from sibyl.runtime import RuntimePolicy, SibylRuntime
from test_researcher import evidence_bundle


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
        notebook = ResearchNotebook("q", bundles=[
            bundle, replace(bundle, bundle_id="another"), evidence_bundle("snippet", origin="search_snippet")
        ])
        self.assertEqual(len(notebook.selected_sources()), len(bundle.sources))

    def test_roundtrip_preserves_exact_evidence_and_citation_ids(self):
        notebook = ResearchNotebook("q", bundles=[evidence_bundle()])
        loaded = ResearchNotebook.from_dict(json.loads(json.dumps(notebook.to_dict())))
        self.assertEqual(notebook, loaded)
        self.assertEqual(notebook.report_sources(), loaded.report_sources())

    def test_budgets_do_not_reset_on_restore(self):
        notebook = ResearchNotebook("q", max_steps=1, max_model_calls=1)
        notebook.begin_search("first")
        notebook.reserve_model_call(100)
        loaded = ResearchNotebook.from_dict(notebook.to_dict())
        with self.assertRaises(ResearchBudgetExceeded):
            loaded.begin_search("second")
        with self.assertRaises(ResearchBudgetExceeded):
            loaded.reserve_model_call(100)


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
        return self.jobs.start("Research question", ["Focused question"], "host", 2, "en", **{"max_steps": 3, **kwargs})

    async def retrieve(self, job):
        with mock.patch("sibyl.research_jobs.gather_source_bundle", mock.AsyncMock(return_value=evidence_bundle("Focused question"))):
            return await self.jobs.continue_job(job["job_id"], "Focused question")

    async def test_host_flow_survives_restart_and_exports_json(self):
        job = await self.retrieve(self.start())
        await self.runtime.aclose()
        self.runtime = SibylRuntime(self.policy, Config())
        self.jobs = self.runtime.research_jobs
        restored = self.jobs.get(job["job_id"])
        self.assertEqual(restored["evidence"], job["evidence"])
        completed = self.jobs.complete(
            job["job_id"], restored["revision"], "Summary [Source 1]",
            ["Finding [Source 1]"], "", ["One question remains unresolved."],
            allow_incomplete=True,
        )
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["report"]["verification_scope"], "citation_targets_only")
        server = create_server(self.runtime)
        await server.call_tool("save_report", {"job_id": job["job_id"], "format": "json"})
        path = next(Path(self.directory.name).glob("*.json"))
        report = ResearchReport.from_dict(json.loads(path.read_text()))
        self.assertEqual(report.status, "partial")
        self.assertFalse(report.research_plan["ready_to_complete"])
        self.assertEqual(report.sources[0].passage_ids, completed["sources"][0]["passage_ids"])
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        with self.assertRaises(ValueError):
            await self.jobs.continue_job(job["job_id"], "mutate completed")

    async def test_stale_revision_cannot_bind_prose_to_changed_citations(self):
        initial = self.start()
        job = await self.retrieve(initial)
        with self.assertRaisesRegex(ValueError, "revision"):
            self.jobs.complete(job["job_id"], initial["revision"], "Summary [Source 1]", ["F [Source 1]"], "", [])

    async def test_missing_and_unknown_citations_are_rejected(self):
        job = await self.retrieve(self.start())
        for summary, finding in [("No citation", "F [Source 1]"), ("S [Source 1]", "F [Source 999]")]:
            with self.assertRaises(ValueError):
                self.jobs.complete(job["job_id"], job["revision"], summary, [finding], "", [], allow_incomplete=True)

    async def test_failed_step_uses_budget_and_reports_failure(self):
        job = self.start(max_steps=1)
        with mock.patch("sibyl.research_jobs.gather_source_bundle", mock.AsyncMock(side_effect=RuntimeError("secret"))):
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
        bundle.sources[0].evidence[0] = replace(bundle.sources[0].evidence[0], text="x" * (JobStore.MAX_PAYLOAD_BYTES + 1))
        with mock.patch("sibyl.research_jobs.gather_source_bundle", mock.AsyncMock(return_value=bundle)):
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

    async def test_restart_marks_unfinished_attempt_failed_without_refunding_budget(self):
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

    async def test_autonomous_resume_preserves_checkpoints(self):
        self.runtime._config = Config(providers=[Provider(model="test", api_key="not-real")])
        entered = asyncio.Event()
        checkpointed = []

        async def fake_research(researcher, query, **kwargs):
            if not researcher.notebook.bundles:
                researcher.notebook.begin_search("saved")
                researcher.notebook.finish_search("saved", evidence_bundle("saved"))
                researcher.checkpoint()
                entered.set()
                await asyncio.Event().wait()
            checkpointed.append(len(researcher.notebook.bundles))
            return ResearchReport(query, "Summary [Source 1]", ["F [Source 1]"], researcher.notebook.report_sources())

        with mock.patch("sibyl.researcher.Researcher.research", fake_research):
            job = self.jobs.start("q", [], "autonomous", 2, "en", 3)
            await entered.wait()
            await self.runtime.aclose()
            self.runtime = SibylRuntime(self.policy, Config(providers=[Provider(model="test", api_key="not-real")]))
            self.jobs = self.runtime.research_jobs
            self.assertEqual(self.jobs.get(job["job_id"])["status"], "interrupted")
            await self.jobs.continue_job(job["job_id"], "")
            await asyncio.gather(*list(self.jobs.tasks.values()))
        self.assertEqual(checkpointed, [1])
        self.assertEqual(self.jobs.get(job["job_id"])["status"], "completed")

    async def test_autonomous_oversize_recovery_preserves_model_reservations(self):
        self.runtime._config = Config(providers=[Provider(model="test", api_key="not-real")])

        async def oversized(researcher, query, **kwargs):
            researcher.notebook.begin_search("large")
            researcher.notebook.reserve_model_call(100)
            researcher.checkpoint()
            bundle = evidence_bundle("large")
            bundle.sources[0].evidence[0] = replace(bundle.sources[0].evidence[0], text="x" * (JobStore.MAX_PAYLOAD_BYTES + 1))
            researcher.notebook.finish_search("large", bundle)
            researcher.checkpoint()

        with mock.patch("sibyl.researcher.Researcher.research", oversized):
            job = self.jobs.start("q", [], "autonomous", 1, "en", 2)
            await asyncio.gather(*list(self.jobs.tasks.values()), return_exceptions=True)
        restored = self.jobs.get(job["job_id"])
        self.assertEqual(restored["status"], "failed")
        self.assertEqual(restored["model_calls"], 1)
        self.assertEqual(restored["reserved_output_tokens"], 100)
        self.assertEqual(restored["remaining_steps"], 1)
        self.assertEqual(restored["attempts"][0]["status"], "failed")
        self.assertFalse(self.jobs.tasks)

    async def test_immediate_autonomous_cancel_does_not_leave_busy_job(self):
        self.runtime._config = Config(providers=[Provider(model="test", api_key="not-real")])
        with mock.patch("sibyl.researcher.Researcher.research", mock.AsyncMock()) as research:
            job = self.jobs.start("q", [], "autonomous", 1, "en", 2)
            result = await self.jobs.cancel(job["job_id"])
            research.assert_not_called()
        self.assertFalse(self.jobs.tasks)
        self.assertEqual(result["status"], "cancelled")

    async def test_autonomous_timeout_retains_checkpoint_and_stops_worker(self):
        self.runtime.policy = replace(self.policy, report_timeout=0.02)
        self.runtime._config = Config(providers=[Provider(model="test", api_key="not-real")])

        async def blocked(*args, **kwargs):
            await asyncio.Event().wait()

        with mock.patch("sibyl.researcher.Researcher.research", blocked):
            job = self.jobs.start("q", [], "autonomous", 1, "en", 2)
            await asyncio.gather(*list(self.jobs.tasks.values()))
        self.assertEqual(self.jobs.get(job["job_id"])["status"], "budget_exhausted")
        with self.assertRaises(ResearchBudgetExceeded):
            await self.jobs.continue_job(job["job_id"], "")

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
        with self.assertRaises(ToolError):
            await server.call_tool("start_research", {"question": "q", "sub_questions": ["x" * 10001]})
        self.assertFalse(self.jobs.store.list_jobs())

    async def test_keyless_profile_includes_full_host_workflow(self):
        server = create_server(self.runtime)
        with mock.patch.dict(os.environ, {"SIBYL_MCP_PROFILE": "keyless"}):
            await configure_tool_profile(server, self.runtime)
        names = {tool.name for tool in await server.list_tools()}
        self.assertTrue({"start_research", "complete_research", "save_report"} <= names)
        self.assertNotIn("research", names)

    async def test_mcp_job_response_has_structured_content(self):
        server = create_server(self.runtime)
        _, structured = await server.call_tool("start_research", {"question": "q"})
        self.assertEqual(structured["status"], "awaiting_input")
        self.assertTrue(structured["job_id"])

    async def test_store_is_private_and_rejects_second_runtime_owner(self):
        self.assertEqual(stat.S_IMODE(self.jobs.store.path.stat().st_mode), 0o600)
        with self.assertRaises(CapacityError):
            JobStore(self.policy.output_root)
        with self.jobs.store.connection() as connection:
            self.assertEqual(connection.execute("PRAGMA max_page_count").fetchone()[0], 16384)

    async def test_payload_overflow_preserves_prior_checkpoint(self):
        job = self.jobs.store.get(self.start()["job_id"])
        revision = job["revision"]
        job["oversized"] = "x" * (1024 * 1024)
        with self.assertRaises(CapacityError):
            self.jobs.store.save(job)
        loaded = self.jobs.store.get(job["job_id"])
        self.assertEqual(loaded["revision"], revision)
        self.assertNotIn("oversized", loaded)

    async def test_autonomous_model_not_called_for_host_mode(self):
        with mock.patch("sibyl.researcher.litellm.acompletion", mock.AsyncMock(side_effect=AssertionError)) as model:
            await self.retrieve(self.start())
        model.assert_not_called()
