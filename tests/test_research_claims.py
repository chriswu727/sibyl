"""Evidence-linked research through the real MCP handlers, without live providers."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp.server.fastmcp.exceptions import ToolError
from research_fixtures import evidence_bundle

from sibyl.mcp_catalog import _validate_payload
from sibyl.mcp_server import create_server
from sibyl.report_models import ResearchReport
from sibyl.reporter import _report_to_markdown
from sibyl.research_claims import ResearchClaim
from sibyl.runtime import RuntimePolicy, SibylRuntime


class TestResearchClaims(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.policy = RuntimePolicy(output_root=Path(self.directory.name))
        self.runtime = SibylRuntime(self.policy)
        self.server = create_server(self.runtime, agent_mode=True)

    async def asyncTearDown(self):
        await self.runtime.aclose()
        self.directory.cleanup()

    async def call(self, name, **arguments):
        response = await self.server.call_tool(name, arguments)
        self.assertFalse(response.isError)
        return response.structuredContent

    async def start(self, sub_questions=None):
        return await self.call(
            "start_research",
            question="Choose a database for reader/writer overlap.",
            sub_questions=sub_questions,
            max_steps=8,
        )

    async def retrieve(self, job, question_id="Q0", purpose="counterevidence"):
        with patch(
            "sibyl.research_jobs.gather_source_bundle",
            AsyncMock(return_value=evidence_bundle("observed", 3)),
        ):
            return await self.call(
                "continue_research",
                job_id=job["job_id"],
                question_id=question_id,
                query="Inspect concurrency conditions",
                purpose=purpose,
                rationale="Check limitations before recommending.",
            )

    def facts(self, job, question_id="Q0"):
        return [
            {
                "claim_id": question_id + ".C1",
                "kind": "fact",
                "status": "supported",
                "text": "The inspected source describes a bounded concurrency capability.",
                "supporting_passage_ids": [job["sources"][0]["passage_ids"][0]],
            }
        ]

    def recommendation(self, owner="Q0", basis="Q0.C1"):
        return {
            "claim_id": owner + ".C2",
            "kind": "recommendation",
            "status": "supported",
            "text": "Consider this option if the documented conditions fit the workload.",
            "basis_claim_ids": [basis],
            "rationale": "The documented capability meets the stated concurrency need.",
            "assumptions": ["The workload has not been benchmarked."],
            "criteria": [
                {
                    "name": "Concurrency",
                    "preference": "Reader/writer overlap",
                    "origin": "user",
                }
            ],
        }

    async def assess(self, job, claims, question_id="Q0", status="supported"):
        return await self.call(
            "assess_research",
            job_id=job["job_id"],
            revision=job["revision"],
            question_id=question_id,
            status=status,
            conclusion="Conditional conclusion from inspected evidence.",
            counterevidence_note="Read the available limitations; empirical workload testing remains separate.",
            claims=claims,
        )

    async def test_complete_recommendation_roundtrip_and_exports(self):
        job = await self.retrieve(await self.start())
        self.assertEqual(job["claim_contract_version"], 2)
        job = await self.assess(job, self.facts(job) + [self.recommendation()])
        self.assertTrue(job["research_plan"]["ready_to_complete"])
        self.assertIsNone(job["report"])
        await self.runtime.aclose()
        self.runtime = SibylRuntime(self.policy)
        self.server = create_server(self.runtime, agent_mode=True)
        restored = await self.call("get_research", job_id=job["job_id"])
        self.assertEqual(restored["research_plan"], job["research_plan"])
        finalized = await self.call(
            "complete_research",
            job_id=job["job_id"],
            revision=restored["revision"],
            claim_ids=["Q0.C2"],
        )
        self.assertEqual(finalized["report"]["status"], "ok")
        full = await self.call("get_research", job_id=job["job_id"], view="full")
        report = full["report"]
        self.assertEqual(report["schema_version"], "1.1")
        self.assertEqual(
            {claim["claim_id"] for claim in report["claims"]}, {"Q0.C1", "Q0.C2"}
        )
        self.assertIn("[Source 1]", report["summary"])
        self.assertIn("Criterion (user)", report["analysis"])
        self.assertIn("not been benchmarked", report["analysis"])
        self.assertEqual(report["verification_scope"], "citation_targets_only")
        for format in ("md", "json"):
            await self.server.call_tool(
                "save_report", {"job_id": job["job_id"], "format": format}
            )
        self.assertEqual(
            json.loads(next(Path(self.directory.name).glob("*.json")).read_text())[
                "claims"
            ],
            report["claims"],
        )
        markdown = next(Path(self.directory.name).glob("*.md")).read_text()
        self.assertIn("recommendation (supported)", markdown)
        self.assertIn("Criterion (user)", markdown)
        with self.assertRaisesRegex(ToolError, "immutable"):
            await self.assess(full, self.facts(full))

    async def test_unknown_cannot_be_hidden_by_selecting_only_recommendation(self):
        job = await self.retrieve(await self.start())
        unknown = {
            "claim_id": "Q0.C3",
            "kind": "fact",
            "status": "unresolved",
            "text": "Current price is unknown.",
            "rationale": "No dated pricing evidence was found.",
        }
        job = await self.assess(job, self.facts(job) + [self.recommendation(), unknown])
        self.assertFalse(job["research_plan"]["ready_to_complete"])
        with self.assertRaisesRegex(ToolError, "gaps remain"):
            await self.call(
                "complete_research",
                job_id=job["job_id"],
                revision=job["revision"],
                claim_ids=["Q0.C2"],
            )
        await self.call(
            "complete_research",
            job_id=job["job_id"],
            revision=job["revision"],
            claim_ids=["Q0.C2"],
            allow_incomplete=True,
        )
        full = await self.call("get_research", job_id=job["job_id"], view="full")
        report = ResearchReport.from_dict(full["report"])
        self.assertEqual(report.status, "partial")
        self.assertIn("Current price is unknown", _report_to_markdown(report))
        self.assertEqual(report.claims[-1]["status"], "unresolved")

    async def test_explicit_abstention_can_export_without_sources(self):
        job = await self.start()
        job = await self.assess(
            job,
            [
                {
                    "claim_id": "Q0.C1",
                    "kind": "fact",
                    "status": "unresolved",
                    "text": "The requested capability could not be established.",
                    "rationale": "No usable original evidence is available.",
                }
            ],
            status="unresolved",
        )
        await self.call(
            "complete_research",
            job_id=job["job_id"],
            revision=job["revision"],
            allow_incomplete=True,
        )
        full = await self.call("get_research", job_id=job["job_id"], view="full")
        self.assertEqual(full["report"]["sources"], [])
        self.assertEqual(full["report"]["status"], "partial")
        self.assertNotIn("[Source", full["report"]["summary"])

    async def test_arbitrary_prose_cannot_bypass_the_claim_report(self):
        job = await self.retrieve(await self.start())
        job = await self.assess(job, self.facts(job))
        for arguments in (
            {"summary": "Invented price [Source 1]"},
            {"key_findings": ["Invented precision [Source 1]"]},
            {"analysis": "Unsupported new assertion"},
        ):
            with (
                self.subTest(arguments=arguments),
                self.assertRaisesRegex(ToolError, "omit summary"),
            ):
                await self.call(
                    "complete_research",
                    job_id=job["job_id"],
                    revision=job["revision"],
                    **arguments,
                )

    async def test_invalid_claims_do_not_mutate_checkpoint(self):
        job = await self.retrieve(await self.start())
        fact = self.facts(job)[0]
        variants = [
            [{**fact, "supporting_passage_ids": []}],
            [{**fact, "supporting_passage_ids": ["missing"]}],
            [{**fact, "claim_id": "Q1.C1"}],
            [fact, fact],
            [{**fact, "kind": "inference"}],
            [
                {
                    **fact,
                    "kind": "inference",
                    "rationale": "Reason",
                    "basis_claim_ids": ["Q0.C99"],
                }
            ],
            [{**fact, "basis_claim_ids": ["Q0.C2"]}],
            [{**self.recommendation(), "criteria": []}],
            [
                {
                    **self.recommendation(),
                    "criteria": [
                        {"name": "Price", "preference": "Cheap", "origin": "source"}
                    ],
                }
            ],
            [{**fact, "status": "unresolved", "rationale": ""}],
            [{**fact, "status": "disputed"}],
            [{**fact, "challenging_passage_ids": fact["supporting_passage_ids"]}],
            [{**fact, "extra": "not allowed"}],
            [{**fact, "text": "Untracked reference [Source 2]"}],
        ]
        before = self.runtime.research_jobs.store.get(job["job_id"])
        for claims in variants:
            with self.subTest(claims=claims), self.assertRaises(ToolError):
                await self.assess(job, claims)
            self.assertEqual(
                self.runtime.research_jobs.store.get(job["job_id"]), before
            )

    async def test_dependency_cycles_are_rejected(self):
        job = await self.retrieve(await self.start())
        claims = [
            {
                "claim_id": "Q0.C1",
                "kind": "inference",
                "status": "supported",
                "text": "One",
                "rationale": "Reason",
                "basis_claim_ids": ["Q0.C2"],
            },
            {
                "claim_id": "Q0.C2",
                "kind": "inference",
                "status": "supported",
                "text": "Two",
                "rationale": "Reason",
                "basis_claim_ids": ["Q0.C1"],
            },
        ]
        with self.assertRaisesRegex(ToolError, "cycles"):
            await self.assess(job, claims)

    async def test_cross_question_dependency_changes_require_reassessment(self):
        job = await self.retrieve(
            await self.start(["What is documented?", "Which option fits?"]), "Q1"
        )
        facts = self.facts(job, "Q1")
        job = await self.assess(job, facts, "Q1")
        job = await self.retrieve(job, "Q2")
        recommendation = self.recommendation("Q2", "Q1.C1")
        job = await self.assess(job, [recommendation], "Q2")
        self.assertTrue(job["research_plan"]["ready_to_complete"])
        facts[0]["text"] = "Updated factual scope changes the premise."
        job = await self.assess(job, facts, "Q1")
        self.assertFalse(job["research_plan"]["ready_to_complete"])
        self.assertIn("changed", " ".join(job["research_plan"]["questions"][1]["gaps"]))
        await self.call(
            "complete_research",
            job_id=job["job_id"],
            revision=job["revision"],
            claim_ids=["Q2.C2"],
            allow_incomplete=True,
        )
        full = await self.call("get_research", job_id=job["job_id"], view="full")
        self.assertFalse(full["report"]["claims"][-1]["review_current"])
        self.assertEqual(full["report"]["status"], "partial")
        self.assertIn("needs reassessment", full["report"]["summary"])

    async def test_new_evidence_invalidates_dependent_question(self):
        job = await self.retrieve(await self.start(["Premise", "Recommendation"]), "Q1")
        job = await self.assess(job, self.facts(job, "Q1"), "Q1")
        job = await self.retrieve(job, "Q2")
        job = await self.assess(job, [self.recommendation("Q2", "Q1.C1")], "Q2")
        job = await self.retrieve(job, "Q1")
        gaps = job["research_plan"]["questions"][1]["gaps"]
        self.assertTrue(any("basis Q1.C1 needs reassessment" in gap for gap in gaps))

    async def test_referenced_premise_cannot_be_deleted(self):
        job = await self.retrieve(await self.start(["Premise", "Recommendation"]), "Q1")
        job = await self.assess(job, self.facts(job, "Q1"), "Q1")
        job = await self.retrieve(job, "Q2")
        job = await self.assess(job, [self.recommendation("Q2", "Q1.C1")], "Q2")
        before = self.runtime.research_jobs.store.get(job["job_id"])
        with self.assertRaisesRegex(ToolError, "preserve claims"):
            await self.assess(job, [], "Q1", status="unresolved")
        self.assertEqual(self.runtime.research_jobs.store.get(job["job_id"]), before)

    async def test_old_checkpoint_keeps_prose_completion_without_upgrade(self):
        job = self.runtime.research_jobs.start(
            "Legacy question", [], "host", 2, "en", 3
        )
        job = await self.retrieve(job)
        saved = self.runtime.research_jobs.store.get(job["job_id"])
        saved["notebook"].pop("claim_contract_version")
        self.runtime.research_jobs.store.save(saved)
        await self.runtime.aclose()
        self.runtime = SibylRuntime(self.policy)
        self.server = create_server(self.runtime, agent_mode=True)
        restored = await self.call("get_research", job_id=job["job_id"])
        self.assertEqual(restored["claim_contract_version"], 1)
        await self.call(
            "complete_research",
            job_id=job["job_id"],
            revision=restored["revision"],
            summary="Legacy summary [Source 1]",
            key_findings=["Legacy finding [Source 1]"],
            allow_incomplete=True,
        )
        full = await self.call("get_research", job_id=job["job_id"], view="full")
        self.assertEqual(full["report"]["schema_version"], "1.0")
        self.assertEqual(full["report"]["claims"], [])

    async def test_missing_claims_keep_new_jobs_incomplete(self):
        job = await self.retrieve(await self.start())
        job = await self.assess(job, [], status="unresolved")
        self.assertFalse(job["research_plan"]["ready_to_complete"])
        self.assertTrue(
            any(
                "claim-level" in gap
                for gap in job["research_plan"]["questions"][0]["gaps"]
            )
        )

    async def test_derived_claims_cannot_upgrade_unresolved_premises(self):
        job = await self.retrieve(await self.start())
        fact = self.facts(job)[0]
        fact.update(
            status="unresolved",
            rationale="The source leaves a material condition unclear.",
        )
        with self.assertRaisesRegex(ToolError, "cannot depend"):
            await self.assess(job, [fact, self.recommendation()])
        recommendation = self.recommendation()
        recommendation.update(
            status="unresolved", rationale="The material premise is not established."
        )
        job = await self.assess(job, [fact, recommendation])
        gaps = job["research_plan"]["questions"][0]["gaps"]
        self.assertTrue(any("basis Q0.C1 is unresolved" in gap for gap in gaps))

    async def test_contradictory_sources_remain_disputed(self):
        job = await self.retrieve(await self.start())
        fact = self.facts(job)[0]
        fact.update(
            status="disputed",
            challenging_passage_ids=[job["sources"][1]["passage_ids"][0]],
        )
        job = await self.assess(job, [fact], status="disputed")
        self.assertFalse(job["research_plan"]["ready_to_complete"])
        await self.call(
            "complete_research",
            job_id=job["job_id"],
            revision=job["revision"],
            allow_incomplete=True,
        )
        full = await self.call("get_research", job_id=job["job_id"], view="full")
        self.assertIn("fact (disputed)", full["report"]["summary"])
        self.assertIn("[Source 2]", full["report"]["summary"])

    async def test_model_payloads_obey_the_combined_text_limit(self):
        claim = self.recommendation()
        claim.update(
            text="x" * 1000,
            rationale="x" * 1000,
            assumptions=["x" * 300] * 4,
            criteria=[{"name": "x" * 80, "preference": "x" * 300, "origin": "user"}]
            * 6,
        )
        inputs = [
            ResearchClaim.model_validate(
                {**copy.deepcopy(claim), "claim_id": f"Q0.C{i}"}
            )
            for i in range(1, 13)
        ]
        with self.assertRaisesRegex(ToolError, "Combined tool text"):
            _validate_payload([inputs])
        job = await self.start()
        with self.assertRaisesRegex(ToolError, "Combined tool text"):
            await self.assess(job, [item.model_dump() for item in inputs])
