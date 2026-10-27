"""Agent projections preserve research judgments without repeating source text."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult
from research_fixtures import evidence_bundle

from sibyl.config import Config
from sibyl.mcp_server import create_server
from sibyl.research_views import research_brief, source_page
from sibyl.runtime import RuntimePolicy, SibylRuntime


class TestAgentResearch(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.runtime = SibylRuntime(
            RuntimePolicy(output_root=Path(self.directory.name)), Config()
        )
        self.jobs = self.runtime.research_jobs
        self.server = create_server(self.runtime, agent_mode=True)

    async def asyncTearDown(self):
        await self.runtime.aclose()
        self.directory.cleanup()

    def start(self):
        return self.jobs.start("Research question", [], "host", 2, "en", 3)

    async def retrieve(self, job, query="focused", purpose="explore"):
        with patch(
            "sibyl.research_jobs.gather_source_bundle",
            AsyncMock(return_value=evidence_bundle(query, 3)),
        ):
            return await self.jobs.continue_job(
                job["job_id"], query, question_id="Q0", purpose=purpose
            )

    async def call(self, name, **kwargs):
        response = await self.server.call_tool(name, kwargs)
        self.assertIsInstance(response, CallToolResult)
        self.assertFalse(response.isError)
        # Text-only hosts receive the same payload as structuredContent.
        self.assertEqual(
            json.loads(response.content[0].text), response.structuredContent
        )
        return response.structuredContent

    async def rejected(self, name, **arguments):
        response = await self.server.call_tool(name, arguments)
        self.assertIsInstance(response, CallToolResult)
        self.assertTrue(response.isError)
        self.assertEqual(response.structuredContent["view"], "error")
        self.assertEqual(
            json.loads(response.content[0].text), response.structuredContent
        )
        return response.structuredContent

    async def test_agent_surface_and_compatible_default(self):
        self.assertEqual(
            {tool.name for tool in await self.server.list_tools()},
            {
                "start_research",
                "get_research",
                "continue_research",
                "assess_research",
                "complete_research",
                "cancel_research",
                "save_report",
                "gather_bundle",
            },
        )
        job = await self.call("start_research", question="Question")
        self.assertEqual(job["view"], "brief")
        self.assertIsNone(job["report"])
        self.assertNotIn("evidence", job)
        self.assertFalse(job["research_plan"]["ready_to_complete"])
        default = create_server(self.runtime)
        _, full = await default.call_tool("get_research", {"job_id": job["job_id"]})
        self.assertIn("evidence", full)
        self.assertNotIn("view", full)
        self.assertEqual(
            (await self.call("get_research"))["jobs"][0]["job_id"], job["job_id"]
        )
        self.assertEqual(
            (await self.call("cancel_research", job_id=job["job_id"], delete=True))[
                "status"
            ],
            "deleted",
        )

    async def test_tool_schemas_expose_choices_and_export_semantics(self):
        for agent_mode in (False, True):
            tools = {
                tool.name: tool
                for tool in await create_server(
                    self.runtime, agent_mode=agent_mode
                ).list_tools()
            }
            for name, field, choices in (
                ("start_research", "source_policy", {"primary", "academic"}),
                (
                    "continue_research",
                    "purpose",
                    {"explore", "counterevidence", "follow_citation"},
                ),
                (
                    "assess_research",
                    "status",
                    {"supported", "disputed", "unresolved"},
                ),
                ("save_report", "format", {"md", "json", "pdf", "both"}),
            ):
                self.assertEqual(
                    set(tools[name].inputSchema["properties"][field]["enum"]),
                    choices,
                )
            self.assertIn("does not create JSON", tools["save_report"].description)
            self.assertIn("independent filename", tools["save_report"].description)
            fields = tools["save_report"].outputSchema["properties"]
            if agent_mode:
                self.assertTrue(
                    {"artifacts", "failures", "report_status"} <= fields.keys()
                )
            else:
                self.assertEqual(set(fields), {"result"})

    async def test_invalid_source_policy_does_not_create_a_job(self):
        error = await self.rejected(
            "start_research",
            question="Question",
            source_policy="Prefer official sources",
        )
        self.assertEqual(error["error"]["code"], "invalid_parameters")
        self.assertEqual(self.jobs.store.list_jobs(), [])

    async def test_brief_is_lossless_for_reviews_limits_and_citation_identity(self):
        job = await self.retrieve(self.start())
        passage = job["evidence"][0]["evidence"][0]["citation_id"]
        job = self.jobs.assess(
            job["job_id"],
            job["revision"],
            "Q0",
            "unresolved",
            [passage],
            [],
            "Evidence is inconclusive.",
            "Need a counterexample.",
            ["focused counterexample"],
        )
        before = json.dumps(job, sort_keys=True)
        brief = research_brief(job)
        self.assertEqual(brief["research_plan"], job["research_plan"])
        gaps = [
            f"{item['question_id']}: {gap}"
            for item in brief["research_plan"]["questions"]
            for gap in item["gaps"]
        ]
        self.assertTrue(gaps)
        self.assertFalse(set(gaps) & set(brief["limitations"]))
        self.assertEqual(set(brief["limitations"]) | set(gaps), set(job["limitations"]))
        self.assertEqual(brief["latest_attempt"], job["attempts"][-1])
        self.assertEqual(
            brief["sources"][0]["passage_ids"], job["sources"][0]["passage_ids"]
        )
        self.assertEqual(before, json.dumps(job, sort_keys=True))
        loaded = await self.call(
            "get_research", job_id=job["job_id"], passage_id=passage
        )
        self.assertEqual(loaded["passage"], job["evidence"][0]["evidence"][0])
        second = job["evidence"][1]["evidence"][0]["citation_id"]
        batch = await self.call(
            "get_research", job_id=job["job_id"], passage_ids=[second, passage, second]
        )
        self.assertEqual(batch["revision"], job["revision"])
        self.assertEqual(
            [item["passage"]["citation_id"] for item in batch["passages"]],
            [second, passage],
        )
        self.assertEqual(
            batch["passages"][1],
            {k: v for k, v in loaded.items() if k not in {"job_id", "revision"}},
        )
        for arguments in (
            {"passage_ids": []},
            {"passage_ids": [passage, "missing"]},
            {"passage_ids": [f"id{i}" for i in range(13)]},
            {"passage_ids": [passage], "passage_id": passage},
            {"passage_ids": [passage], "view": "brief"},
        ):
            await self.rejected("get_research", job_id=job["job_id"], **arguments)
        self.assertEqual(loaded["content_origin"], job["evidence"][0]["content_origin"])
        self.assertIn("retrieved_at", loaded)
        self.assertIn("content_cluster_id", loaded)
        full = await self.call("get_research", job_id=job["job_id"], view="full")
        self.assertEqual(full, job)

    async def test_source_pages_are_complete_bounded_and_revision_checked(self):
        job = await self.retrieve(self.start())
        first = await self.call(
            "get_research",
            job_id=job["job_id"],
            view="sources",
            limit=2,
            revision=job["revision"],
        )
        self.assertEqual(first["total"], 3)
        self.assertEqual(first["next_offset"], 2)
        second = source_page(job, 2, 2, job["revision"])
        self.assertIsNone(second["next_offset"])
        self.assertEqual(first["sources"] + second["sources"], job["source_catalog"])
        self.assertEqual(source_page(job, 99, 2, None)["sources"], [])
        for parameters in (
            {"limit": 0},
            {"limit": 21},
            {"offset": -1},
            {"revision": -1},
        ):
            await self.rejected(
                "get_research", job_id=job["job_id"], view="sources", **parameters
            )
        updated = await self.retrieve(job, "new")
        self.assertGreater(updated["revision"], job["revision"])
        error = await self.rejected(
            "get_research",
            job_id=job["job_id"],
            view="sources",
            revision=job["revision"],
        )
        self.assertIn("restart source pagination", error["error"]["message"])
        self.assertEqual(error["current_revision"], updated["revision"])

    async def test_finalized_partial_report_is_not_a_quality_pass(self):
        job = await self.retrieve(self.start())
        result = await self.call(
            "complete_research",
            job_id=job["job_id"],
            revision=job["revision"],
            summary="Summary [Source 1]",
            key_findings=["Finding [Source 1]"],
            allow_incomplete=True,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["report"]["status"], "partial")
        self.assertEqual(
            result["report"]["verification_scope"], "citation_targets_only"
        )
        self.assertFalse(result["research_plan"]["ready_to_complete"])
        self.assertTrue(result["report"]["limitations"])

    async def test_retrieval_failure_and_cancel_remain_visible(self):
        job = self.start()
        with patch(
            "sibyl.research_jobs.gather_source_bundle",
            AsyncMock(side_effect=RuntimeError("private")),
        ):
            result = await self.call(
                "continue_research", job_id=job["job_id"], query="q"
            )
        self.assertEqual(result["latest_attempt"]["status"], "failed")
        self.assertEqual(result["remaining_steps"], 2)
        self.assertNotIn("private", json.dumps(result))
        result = await self.call("cancel_research", job_id=job["job_id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertFalse(result["research_plan"]["ready_to_complete"])

    async def test_brief_is_smaller_not_a_delta(self):
        first = await self.retrieve(self.start())
        second = await self.retrieve(first, "counter", "counterevidence")
        for job in (first, second):
            brief = research_brief(job)
            self.assertEqual(brief["revision"], job["revision"])
            self.assertEqual(brief["research_plan"], job["research_plan"])
            self.assertLess(len(json.dumps(brief)), len(json.dumps(job)))
        self.assertEqual(research_brief(second)["source_catalog_count"], 6)

    async def test_views_reject_conflicting_passage_request(self):
        job = self.start()
        for arguments in (
            {"passage_id": "x"},
            {"job_id": job["job_id"], "view": "brief", "passage_id": "x"},
            {"job_id": job["job_id"], "view": "invented"},
        ):
            await self.rejected("get_research", **arguments)

    async def test_new_evidence_revokes_old_readiness_in_agent_brief(self):
        job = await self.retrieve(self.start())
        job = await self.retrieve(job, "counter", "counterevidence")
        refs = [source["passage_ids"][0] for source in job["sources"]]
        reviewed = await self.call(
            "assess_research",
            job_id=job["job_id"],
            revision=job["revision"],
            question_id="Q0",
            status="supported",
            supporting_passage_ids=refs,
            conclusion="A bounded conclusion.",
            counterevidence_note="The contrary evidence does not overturn this narrower claim.",
        )
        self.assertTrue(reviewed["research_plan"]["ready_to_complete"])
        await self.retrieve(job, "fresh")
        updated = await self.call("get_research", job_id=job["job_id"])
        self.assertFalse(updated["research_plan"]["ready_to_complete"])
        self.assertTrue(updated["research_plan"]["questions"][0]["gaps"])
        self.assertGreater(updated["revision"], reviewed["revision"])

    async def test_catalog_retains_evidence_outside_selected_window(self):
        job = self.start()
        with patch(
            "sibyl.research_jobs.gather_source_bundle",
            AsyncMock(return_value=evidence_bundle("many", 15)),
        ):
            job = await self.jobs.continue_job(job["job_id"], "many", question_id="Q0")
        selected = {pid for source in job["sources"] for pid in source["passage_ids"]}
        catalog = await self.call(
            "get_research", job_id=job["job_id"], view="sources", limit=20
        )
        excluded = [
            pid
            for source in catalog["sources"]
            for pid in source["passage_ids"]
            if pid not in selected
        ]
        self.assertTrue(excluded)
        passage = await self.call(
            "get_research", job_id=job["job_id"], passage_id=excluded[0]
        )
        self.assertEqual(passage["passage"]["citation_id"], excluded[0])
        self.assertTrue(passage["passage"]["text"])
