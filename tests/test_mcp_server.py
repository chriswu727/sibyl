"""MCP report safety and request-isolation tests. No network."""
import asyncio
import os
import unittest
from unittest import mock

from mcp.server.fastmcp.exceptions import ToolError

import sibyl.mcp_server as mcp_server
from sibyl.config import Config, Provider
from sibyl.evidence import BundleDiagnostics, SourceBundle
from sibyl.mcp_server import _format_report, gather_bundle, gather_sources, research
from sibyl.researcher import ResearchReport
from sibyl.verifier import FindingVerification


class TestMcpFormatting(unittest.TestCase):
    def test_verification_verdicts_are_visible(self):
        report = ResearchReport(
            "q", "summary", ["Supported [Source 1]", "Unsupported [Source 1]"], [],
            finding_verifications=[
                FindingVerification(1, True, "high", [1], ""),
                FindingVerification(2, False, "low", [1], "not found"),
            ],
        )
        text = _format_report(report)
        self.assertIn("[verified: high]", text)
        self.assertIn("(unverified)", text)

    def test_insufficient_evidence_is_not_formatted_as_a_report(self):
        report = ResearchReport(
            "q", "", [], [], status="insufficient_evidence", error="No sources found"
        )
        text = _format_report(report)
        self.assertIn("Insufficient evidence", text)
        self.assertIn("No sources found", text)
        self.assertNotIn("## Summary", text)

    def test_missing_verification_is_disclosed(self):
        report = ResearchReport("q", "summary", ["Claim [Source 1]"], [])
        self.assertIn("verification was not performed", _format_report(report))


class TestMcpResearch(unittest.IsolatedAsyncioTestCase):
    async def test_missing_credentials_fails_before_network(self):
        cfg = Config(providers=[Provider(model="deepseek/deepseek-v4-flash")])
        mcp_server._last_report = ResearchReport("old", "stale", [], [])
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch("sibyl.mcp_server._get_config", return_value=cfg):
            with self.assertRaisesRegex(ToolError, "requires an LLM provider key"):
                await research("question")
        self.assertIsNone(mcp_server._last_report)

    async def test_per_call_flags_do_not_mutate_global_config(self):
        cfg = Config(
            providers=[Provider(model="deepseek/deepseek-v4-flash", api_key="k")],
            fast=False,
            verify_claims=True,
        )
        captured = {}

        class FakeResearcher:
            def __init__(self, config):
                captured["config"] = config

            async def research(self, query, **kwargs):
                return ResearchReport(query, "summary", [], [])

        with mock.patch("sibyl.mcp_server._get_config", return_value=cfg), \
             mock.patch("sibyl.researcher.Researcher", FakeResearcher):
            await research("question", fast=True, verify=False)

        self.assertFalse(cfg.fast)
        self.assertTrue(cfg.verify_claims)
        self.assertTrue(captured["config"].fast)
        self.assertFalse(captured["config"].verify_claims)


class TestMcpRetrieval(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bundle = SourceBundle(
            "1.6",
            "sb_test",
            "question",
            "insufficient_evidence",
            [],
            BundleDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 7000, 7000, 1),
            "No sources found for query: 'question'. Try a different phrasing.",
        )

    async def test_structured_tool_returns_source_bundle(self):
        gather = mock.AsyncMock(return_value=self.bundle)
        with mock.patch(
            "sibyl.mcp_server.gather_source_bundle",
            new=gather,
        ):
            result = await gather_bundle("question", ranker="none")

        self.assertIs(result, self.bundle)
        self.assertEqual(result.to_dict()["bundle_id"], "sb_test")
        gather.assert_awaited_once_with(
            "question", 10, 7000, ranker="none", render_thin_pages=False
        )

    async def test_stdio_defaults_to_warning_logs(self):
        self.assertEqual(mcp_server.mcp.settings.log_level, "WARNING")

    async def test_fastmcp_serializes_structured_content(self):
        with mock.patch(
            "sibyl.mcp_server.gather_source_bundle",
            new=mock.AsyncMock(return_value=self.bundle),
        ):
            _, structured = await mcp_server.mcp.call_tool(
                "gather_bundle", {"query": "question"}
            )

        self.assertEqual(structured["bundle_id"], "sb_test")
        self.assertEqual(structured["diagnostics"]["sources_returned"], 0)

    async def test_legacy_tool_renders_same_bundle(self):
        with mock.patch(
            "sibyl.mcp_server.gather_source_bundle",
            new=mock.AsyncMock(return_value=self.bundle),
        ):
            result = await gather_sources("question")

        self.assertEqual(result, self.bundle.error)

    async def test_matching_tools_share_inflight_retrieval(self):
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def retrieve(*args, **kwargs):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return self.bundle

        with mock.patch("sibyl.mcp_server.gather_source_bundle", new=retrieve):
            legacy = asyncio.create_task(gather_sources(" question "))
            await started.wait()
            structured = asyncio.create_task(gather_bundle("question"))
            await asyncio.sleep(0)
            release.set()
            legacy_result, structured_result = await asyncio.gather(
                legacy, structured
            )

        self.assertEqual(legacy_result, self.bundle.error)
        self.assertIs(structured_result, self.bundle)
        self.assertEqual(calls, 1)

    async def test_failed_retrieval_is_not_cached(self):
        failed = SourceBundle(
            "1.6",
            "sb_failed",
            "failure-case",
            "failed",
            [],
            self.bundle.diagnostics,
            "temporary failure",
        )
        recovered = SourceBundle(
            "1.6",
            "sb_recovered",
            "failure-case",
            "ok",
            [],
            self.bundle.diagnostics,
        )
        retrieve = mock.AsyncMock(side_effect=[failed, recovered])

        with mock.patch(
            "sibyl.mcp_server.gather_source_bundle",
            new=retrieve,
        ):
            first = await gather_bundle("failure-case")
            second = await gather_bundle("failure-case")

        self.assertIs(first, failed)
        self.assertIs(second, recovered)
        self.assertEqual(retrieve.await_count, 2)

    async def test_mcp_registers_thirteen_tools(self):
        tools = await mcp_server.mcp.list_tools()
        names = {tool.name for tool in tools}
        structured_tool = next(tool for tool in tools if tool.name == "gather_bundle")

        self.assertEqual(len(tools), 13)
        self.assertIn("gather_bundle", names)
        self.assertIn("gather_sources", names)
        self.assertEqual(
            set(structured_tool.outputSchema["properties"]["status"]["enum"]),
            {"ok", "insufficient_evidence", "invalid_request", "failed"},
        )
        self.assertEqual(
            set(structured_tool.inputSchema["properties"]["ranker"]["enum"]),
            {"lexical", "flashrank", "none"},
        )
        self.assertEqual(
            set(
                structured_tool.outputSchema["$defs"]["EvidenceSource"]
                ["properties"]["content_origin"]["enum"]
            ),
            {
                "direct_fetch",
                "jina_reader",
                "wikipedia_api",
                "search_snippet",
                "crossref_api",
            },
        )

    async def test_auto_profile_hides_tools_that_need_configuration(self):
        config = Config(providers=[Provider(model="deepseek/deepseek-v4-flash")])
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mcp_server._mcp_profile(config), "keyless")

    async def test_auto_profile_enables_report_surface_with_credentials(self):
        config = Config(
            providers=[Provider(model="deepseek/deepseek-v4-flash", api_key="k")]
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mcp_server._mcp_profile(config), "report")


if __name__ == "__main__":
    unittest.main()
