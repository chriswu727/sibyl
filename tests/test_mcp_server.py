"""MCP report safety and request-isolation tests. No network."""

import asyncio
import json
import unittest
from unittest import mock

import sibyl.mcp_server as mcp_server
from sibyl.evidence import BundleDiagnostics, SourceBundle
from sibyl.mcp_server import _format_report, gather_bundle
from sibyl.report_models import FindingVerification, ResearchReport


class TestMcpFormatting(unittest.TestCase):
    def test_verification_verdicts_are_visible(self):
        report = ResearchReport(
            "q",
            "summary",
            ["Supported [Source 1]", "Unsupported [Source 1]"],
            [],
            finding_verifications=[
                FindingVerification(1, True, "high", [1], ""),
                FindingVerification(2, False, "low", [1], "not found"),
            ],
        )
        text = _format_report(report)
        self.assertIn("[historical: supported; recorded confidence: high]", text)
        self.assertIn("historical: unverified", text)
        self.assertIn("not revalidated", text)
        self.assertNotIn("[verified:", text)

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
            "sibyl.mcp_tools.gather_source_bundle",
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
            "sibyl.mcp_tools.gather_source_bundle",
            new=mock.AsyncMock(return_value=self.bundle),
        ):
            result = await mcp_server.mcp.call_tool(
                "gather_bundle", {"query": "question"}
            )

        self.assertFalse(result.isError)
        structured = result.structuredContent
        self.assertEqual(structured, self.bundle.to_dict())
        self.assertEqual(structured["bundle_id"], "sb_test")
        self.assertEqual(structured["diagnostics"]["sources_returned"], 0)
        # Text-only hosts receive the same payload.
        self.assertEqual(json.loads(result.content[0].text), structured)

    async def test_equivalent_queries_share_inflight_retrieval(self):
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def retrieve(*args, **kwargs):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return self.bundle

        with mock.patch("sibyl.mcp_tools.gather_source_bundle", new=retrieve):
            padded = asyncio.create_task(gather_bundle(" question "))
            await started.wait()
            plain = asyncio.create_task(gather_bundle("question"))
            await asyncio.sleep(0)
            release.set()
            padded_result, plain_result = await asyncio.gather(padded, plain)

        self.assertIs(padded_result, self.bundle)
        self.assertIs(plain_result, self.bundle)
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
            "sibyl.mcp_tools.gather_source_bundle",
            new=retrieve,
        ):
            first = await gather_bundle("failure-case")
            second = await gather_bundle("failure-case")

        self.assertIs(first, failed)
        self.assertIs(second, recovered)
        self.assertEqual(retrieve.await_count, 2)

    async def test_mcp_registers_one_research_tool_set(self):
        tools = await mcp_server.mcp.list_tools()
        names = {tool.name for tool in tools}
        structured_tool = next(tool for tool in tools if tool.name == "gather_bundle")

        self.assertEqual(
            names,
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
        self.assertEqual(len(tools), 8)
        self.assertEqual(
            set(structured_tool.inputSchema["properties"]["ranker"]["enum"]),
            {"lexical", "flashrank", "none"},
        )
        # The SourceBundle schema is documented, not repeated in every tool list;
        # tests/test_source_bundle_contract.py checks its consumer enums.
        self.assertIsNone(structured_tool.outputSchema)

    async def test_removed_tools_are_unknown(self):
        for name in ("gather_sources", "gather_evidence", "quick_search", "read_url"):
            with self.subTest(name=name):
                result = await mcp_server.mcp.call_tool(name, {"query": "q"})
                self.assertTrue(result.isError)
                self.assertEqual(result.structuredContent["view"], "error")
                self.assertEqual(result.structuredContent["operation"], name)


if __name__ == "__main__":
    unittest.main()
