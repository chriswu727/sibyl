"""Adversarial model verdicts and report-wide disclosure regressions."""

import json
import unittest
from unittest import mock

from test_researcher import evidence_bundle

from sibyl.config import Provider
from sibyl.report_audit import audit_report
from sibyl.report_models import ResearchReport
from sibyl.research_state import ResearchNotebook
from sibyl.verifier import verify_findings


class TestReportAudit(unittest.IsolatedAsyncioTestCase):
    def response(self, verdicts):
        return mock.Mock(
            choices=[
                mock.Mock(message=mock.Mock(content=json.dumps({"verdicts": verdicts})))
            ]
        )

    async def test_model_cannot_approve_an_invalid_citation(self):
        completion = mock.AsyncMock(
            return_value=self.response(
                [
                    {"index": 1, "supported": True},
                    {"index": 2, "supported": True},
                ]
            )
        )
        notebook = ResearchNotebook("q", bundles=[evidence_bundle()])
        verdicts = await verify_findings(
            ["Claim [Source 999]", "Uncited claim"],
            notebook.pages(),
            Provider("test"),
            completion=completion,
        )
        self.assertTrue(all(not verdict.supported for verdict in verdicts))

    async def test_string_false_is_not_a_positive_verdict(self):
        completion = mock.AsyncMock(
            return_value=self.response([{"index": 1, "supported": "false"}])
        )
        notebook = ResearchNotebook("q", bundles=[evidence_bundle()])
        verdicts = await verify_findings(
            ["Claim [Source 1]"],
            notebook.pages(),
            Provider("test"),
            completion=completion,
        )
        self.assertFalse(verdicts[0].supported)

    async def test_summary_analysis_and_findings_are_all_audited(self):
        notebook = ResearchNotebook("q", bundles=[evidence_bundle()])
        report = ResearchReport(
            "q",
            "Summary [Source 1]",
            ["Finding [Source 1]"],
            notebook.report_sources(),
            analysis="Analysis [Source 1]",
        )
        completion = mock.AsyncMock(
            return_value=self.response(
                [
                    {"index": 1, "supported": False},
                    {"index": 2, "supported": False},
                    {"index": 3, "supported": True},
                ]
            )
        )
        await audit_report(
            report, notebook.pages(), Provider("test"), completion=completion
        )
        self.assertTrue(report.summary.startswith("**Unverified:**"))
        self.assertTrue(report.analysis.startswith("**Unverified:**"))
        self.assertFalse(report.key_findings[0].startswith("**Unverified:**"))
        self.assertEqual(report.verification_scope, "report_blocks_model_checked")

    async def test_malformed_verdicts_never_silently_pass(self):
        notebook = ResearchNotebook("q", bundles=[evidence_bundle()])
        report = ResearchReport(
            "q",
            notebook.pages()[0].text + " [Source 1]",
            ["Claim [Source 1]"],
            notebook.report_sources(),
        )
        completion = mock.AsyncMock(
            return_value=mock.Mock(
                choices=[mock.Mock(message=mock.Mock(content="not json"))]
            )
        )
        await audit_report(
            report, notebook.pages(), Provider("test"), completion=completion
        )
        self.assertIn("Unverified", report.summary)
        self.assertTrue(
            all(not verdict.supported for verdict in report.finding_verifications)
        )
