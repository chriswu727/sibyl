"""Recovery and export contracts without a model or live retrieval."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult

from sibyl.async_cache import CapacityError
from sibyl.mcp_server import create_server
from sibyl.report_models import ResearchReport
from sibyl.runtime import RuntimePolicy, SibylRuntime
from sibyl.storage import ArtifactStore


class TestAgentRecovery(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.runtime = SibylRuntime(RuntimePolicy(output_root=self.root))
        self.server = create_server(self.runtime, agent_mode=True)

    async def asyncTearDown(self):
        await self.runtime.aclose()
        self.directory.cleanup()

    async def call(self, name, error=False, **arguments):
        response = await self.server.call_tool(name, arguments)
        self.assertIsInstance(response, CallToolResult)
        self.assertEqual(bool(response.isError), error, str(response))
        return response.structuredContent

    def review(self, job, claims):
        return dict(
            job_id=job["job_id"],
            revision=job["revision"],
            question_id="Q0",
            status="unresolved",
            conclusion="The outcome remains unknown.",
            counterevidence_note="No usable evidence establishes the requested future outcome.",
            claims=claims,
        )

    def unknown(self):
        return dict(
            claim_id="Q0.C1",
            kind="fact",
            status="unresolved",
            text="The requested future outcome is unknown.",
            rationale="No observed outcome is available.",
        )

    def verify_artifact(self, artifact):
        path = Path(artifact["path"])
        self.assertEqual(path.parent, self.root.resolve())
        self.assertTrue(path.is_file())
        self.assertEqual(artifact["size_bytes"], path.stat().st_size)
        self.assertEqual(
            artifact["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )
        return path

    async def test_rejected_basis_can_be_repaired_without_guessing_revision(self):
        job = await self.call(
            "start_research", question="What is the exact future outcome?"
        )
        before = self.runtime.research_jobs.store.get(job["job_id"])
        invalid = {**self.unknown(), "kind": "inference", "status": "supported"}
        rejected = await self.call(
            "assess_research", error=True, **self.review(job, [invalid])
        )
        self.assertEqual(rejected["error"]["code"], "invalid_claim")
        self.assertEqual(rejected["error"]["field"], "claims.basis_claim_ids")
        self.assertEqual(rejected["error"]["claim_id"], "Q0.C1")
        self.assertEqual(rejected["current_revision"], job["revision"])
        self.assertEqual(rejected["recovery"]["action"], "repair_assessment")
        self.assertEqual(self.runtime.research_jobs.store.get(job["job_id"]), before)
        read = rejected["recovery"]["read"]
        current = await self.call(read["tool"], **read["arguments"])
        job = await self.call(
            "assess_research", **self.review(current, [self.unknown()])
        )
        stale = await self.call(
            "complete_research",
            error=True,
            job_id=job["job_id"],
            revision=before["revision"],
            allow_incomplete=True,
        )
        self.assertEqual(stale["error"]["code"], "stale_revision")
        self.assertEqual(stale["current_revision"], job["revision"])
        self.assertEqual(stale["recovery"]["action"], "reload_research")
        read = stale["recovery"]["read"]
        job = await self.call(read["tool"], **read["arguments"])
        done = await self.call(
            "complete_research",
            job_id=job["job_id"],
            revision=job["revision"],
            allow_incomplete=True,
        )
        self.assertEqual(done["report"]["status"], "partial")
        paths = []
        for format in ("md", "json"):
            exported = await self.call(
                "save_report", job_id=job["job_id"], format=format
            )
            self.assertEqual(exported["view"], "export")
            self.assertEqual(exported["status"], "saved")
            self.assertEqual(exported["report_status"], "partial")
            self.assertEqual(exported["requested_formats"], [format])
            self.assertEqual(exported["failures"], [])
            paths.append(self.verify_artifact(exported["artifacts"][0]))
        self.assertNotEqual(paths[1], paths[0].with_suffix(".json"))
        report = json.loads(paths[1].read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["sources"], [])
        self.assertEqual(report["claims"][0]["status"], "unresolved")

    async def test_disputed_claim_points_to_missing_challenge_and_is_atomic(self):
        job = await self.call("start_research", question="Conflicting sources")
        before = self.runtime.research_jobs.store.get(job["job_id"])
        invalid = {**self.unknown(), "status": "disputed"}
        rejected = await self.call(
            "assess_research", error=True, **self.review(job, [invalid])
        )
        self.assertEqual(rejected["error"]["field"], "claims.challenging_passage_ids")
        self.assertEqual(rejected["error"]["claim_id"], "Q0.C1")
        self.assertEqual(self.runtime.research_jobs.store.get(job["job_id"]), before)

    async def test_schema_error_is_structured_without_echoing_input(self):
        result = await self.call(
            "start_research", error=True, question="q", source_policy="token=SECRET"
        )
        self.assertEqual(result["error"]["code"], "invalid_parameters")
        self.assertEqual(result["error"]["field"], "source_policy")
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertIsNone(self.runtime._research_jobs)
        self.assertIsNone(result["current_revision"])
        self.assertIsNone(result["recovery"]["read"])

    async def test_legacy_mode_does_not_mislead_host_into_revision_retries(self):
        job = await self.call("start_research", question="Historical job")
        saved = self.runtime.research_jobs.store.get(job["job_id"])
        saved["mode"] = "autonomous"
        self.runtime.research_jobs.store.save(saved)
        for name, arguments in (
            ("assess_research", self.review(saved, [self.unknown()])),
            (
                "complete_research",
                {"job_id": saved["job_id"], "revision": saved["revision"]},
            ),
        ):
            rejected = await self.call(name, error=True, **arguments)
            self.assertEqual(rejected["error"]["code"], "unsupported_mode")
            self.assertEqual(rejected["recovery"]["action"], "start_new_research")
            self.assertEqual(
                self.runtime.research_jobs.store.get(saved["job_id"]), saved
            )

    async def test_nested_schema_error_does_not_echo_claim_text_or_extra_keys(self):
        job = await self.call("start_research", question="q")
        invalid = {**self.unknown(), "token=SECRET": "secret value"}
        rejected = await self.call(
            "assess_research", error=True, **self.review(job, [invalid])
        )
        self.assertEqual(rejected["error"]["code"], "invalid_parameters")
        self.assertEqual(rejected["error"]["field"], "claims.0")
        self.assertNotIn("SECRET", json.dumps(rejected))
        self.assertNotIn("secret value", json.dumps(rejected))
        self.assertEqual(rejected["current_revision"], job["revision"])

    async def test_provider_error_is_sanitized_and_keeps_current_state(self):
        job = await self.call("start_research", question="q")
        before = self.runtime.research_jobs.store.get(job["job_id"])
        with patch.object(
            self.runtime,
            "execute",
            AsyncMock(side_effect=RuntimeError("https://host/?token=SECRET")),
        ):
            result = await self.call("get_research", error=True, job_id=job["job_id"])
        self.assertEqual(result["error"]["code"], "operation_failed")
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(self.runtime.research_jobs.store.get(job["job_id"]), before)

    async def test_unfinished_report_has_recovery_but_no_artifact_paths(self):
        job = await self.call("start_research", question="q")
        result = await self.call(
            "save_report", error=True, job_id=job["job_id"], format="json"
        )
        self.assertEqual(result["error"]["code"], "report_not_finalized")
        self.assertEqual(result["recovery"]["action"], "review_and_finalize")
        self.assertEqual(result["current_revision"], job["revision"])
        self.assertEqual(list(self.root.glob("sibyl_*")), [])
        self.assertNotIn("artifacts", result)

    async def test_pdf_failure_returns_failed_manifest_and_counts_failure(self):
        self.runtime.last_report = ResearchReport("q", "summary", [], [])
        with patch(
            "sibyl.reporter.generate_pdf", side_effect=RuntimeError("token=SECRET")
        ):
            result = await self.call("save_report", error=True, format="pdf")
        self.assertEqual(result["view"], "export")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["artifacts"], [])
        self.assertEqual(result["failures"][0]["format"], "pdf")
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(self.runtime.failed, 1)

    async def test_combined_export_retains_only_real_successes(self):
        self.runtime.last_report = ResearchReport(
            "q", "summary", [], [], status="partial"
        )
        with patch(
            "sibyl.reporter.generate_pdf", side_effect=ImportError("token=SECRET")
        ):
            result = await self.call("save_report", format="both")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["report_status"], "partial")
        self.assertEqual(result["requested_formats"], ["pdf", "md"])
        self.assertEqual([item["format"] for item in result["artifacts"]], ["md"])
        self.verify_artifact(result["artifacts"][0])
        self.assertFalse(list(self.root.glob("*.json")))
        self.assertFalse(list(self.root.glob("*.pdf")))
        self.assertNotIn("SECRET", json.dumps(result))

    async def test_missing_renderer_path_is_not_reported_as_saved(self):
        self.runtime.last_report = ResearchReport("q", "summary", [], [])
        with patch(
            "sibyl.reporter.generate_pdf", return_value=str(self.root / "missing.pdf")
        ):
            result = await self.call("save_report", error=True, format="pdf")
        self.assertEqual(result["artifacts"], [])
        self.assertEqual(result["status"], "failed")

    async def test_markdown_failure_retains_an_already_saved_pdf(self):
        self.runtime.last_report = ResearchReport("q", "summary", [], [])

        def render(report, directory):
            return str(
                ArtifactStore(directory).write(b"%PDF-1.4\n%%EOF", "report", ".pdf")
            )

        with (
            patch("sibyl.reporter.generate_pdf", render),
            patch(
                "sibyl.reporter._report_to_markdown",
                side_effect=RuntimeError("token=SECRET"),
            ),
        ):
            result = await self.call("save_report", format="both")
        self.assertEqual(result["status"], "partial")
        self.assertEqual([item["format"] for item in result["artifacts"]], ["pdf"])
        self.verify_artifact(result["artifacts"][0])
        self.assertEqual(result["failures"][0]["format"], "md")
        self.assertNotIn("SECRET", json.dumps(result))

    async def test_renderer_cannot_confirm_paths_outside_the_store_or_symlinks(self):
        self.runtime.last_report = ResearchReport("q", "summary", [], [])
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "external.pdf"
            outside.write_bytes(b"external data")
            link = self.root / "linked.pdf"
            link.symlink_to(outside)
            for path in (outside, link):
                with (
                    self.subTest(path=path),
                    patch("sibyl.reporter.generate_pdf", return_value=str(path)),
                ):
                    result = await self.call("save_report", error=True, format="pdf")
                self.assertEqual(result["artifacts"], [])
                self.assertNotIn(str(path), json.dumps(result))

    async def test_storage_capacity_is_explicit_without_fake_artifact(self):
        self.runtime.last_report = ResearchReport("q", "summary", [], [])
        with patch(
            "sibyl.storage.ArtifactStore.write", side_effect=CapacityError("full")
        ):
            result = await self.call("save_report", error=True, format="json")
        self.assertEqual(result["artifacts"], [])
        self.assertEqual(result["failures"][0]["code"], "capacity_exceeded")

    async def test_legacy_profile_keeps_text_results_and_raised_errors(self):
        self.runtime.last_report = ResearchReport("q", "summary", [], [])
        legacy = create_server(self.runtime)
        _, result = await legacy.call_tool("save_report", {"format": "md"})
        self.assertIn("Markdown saved: ", result["result"])
        self.assertTrue(
            Path(result["result"].removeprefix("Markdown saved: ")).is_file()
        )
        with self.assertRaises(ToolError):
            await legacy.call_tool(
                "start_research", {"question": "q", "source_policy": "invalid"}
            )
