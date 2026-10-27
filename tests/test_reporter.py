"""Reporter smoke tests — PDF generates and markdown is preserved. No network.

Run: python -m unittest discover tests
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest import mock

from sibyl.report_exports import export_report
from sibyl.report_models import ResearchReport, Source
from sibyl.report_references import reference_groups
from sibyl.report_text import _format_report
from sibyl.reporter import _report_to_markdown, generate_pdf


def _report():
    return ResearchReport(
        query="Test report",
        summary="A **bold claim** [Source 1].\n\n## Sub-heading\n\nMore prose with **emphasis**.",
        key_findings=["**Finding one** with 92% [Source 1]", "Finding two"],
        sources=[Source(url="https://ex.com/1", title="Example", snippet="snippet")],
        analysis="### Header\n- bullet with **bold**\n- second bullet\n\nClosing paragraph.",
        predictions="",
        model_used="deepseek/deepseek-v4-flash",
    )


def _claim_report():
    report = _report()
    fact = "The documented mode permits reader and writer overlap."
    recommendation = "Consider this mode if only one writer is required."
    unknown = "The workload throughput is unknown."
    report.model_used = "host"
    report.summary = recommendation + " [Source 1]"
    report.key_findings = [report.summary]
    report.analysis = (
        f"Q0.C1 — fact (supported): {fact} [Source 1]\n\n"
        f"Q0.C2 — recommendation (supported): {recommendation} [Source 1]\n"
        "Basis: Q0.C1\nReasoning: The stated concurrency need fits this limit.\n"
        "Criterion (user): Concurrency — Reader/writer overlap\n"
        "Assumption: The workload has not been benchmarked.\n\n"
        f"Q0.C3 — fact (unresolved): {unknown}\n"
        "Reasoning: No representative workload measurements were available."
    )
    report.claims = [
        {"claim_id": identifier, "text": text}
        for identifier, text in (
            ("Q0.C1", fact),
            ("Q0.C2", recommendation),
            ("Q0.C3", unknown),
        )
    ]
    report.status = "partial"
    report.verification_scope = "citation_targets_only"
    report.limitations = ["No empirical workload benchmark."]
    return report


class TestMarkdownReport(unittest.TestCase):
    def test_reference_groups_preserve_citation_numbers_and_report(self):
        report = _report()
        report.sources = [
            Source(f"https://ex.com/{i}", f"Title {i}", "") for i in range(1, 5)
        ]
        report.summary = "Summary [Source 3]"
        report.key_findings = ["Finding [source 1]"]
        before = report.to_dict()
        for render in (_report_to_markdown, _format_report):
            text = render(report)
            cited, background = text.split("## Uncited Background (2)")
            self.assertIn("## Cited Sources (2)", cited)
            self.assertIn("[Source 3]", cited)
            self.assertNotIn("Title 2", cited)
            self.assertIn("[Source 2]", background)
            self.assertIn("[Source 4]", background)
            self.assertIn("does not establish", cited)
        self.assertEqual(report.to_dict(), before)
        self.assertEqual(ResearchReport.from_dict(before).to_dict(), before)

    def test_uncited_and_invalid_citations_do_not_create_support(self):
        report = _report()
        report.summary = "Unknown [Source 999]"
        report.key_findings = []
        self.assertEqual(
            [group[0] for group in reference_groups(report)], ["Uncited Background"]
        )
        report.sources = []
        self.assertEqual(reference_groups(report), [])

    def test_citations_in_other_sections_are_included(self):
        for name in (
            "analysis",
            "predictions",
            "cross_analysis",
            "market_data_summary",
        ):
            report = _report()
            report.summary = "Summary"
            report.key_findings = []
            setattr(report, name, "Evidence [Source 1]")
            self.assertEqual(reference_groups(report)[0][0], "Cited Sources")

    def test_partial_plain_text_report_keeps_content(self):
        report = _report()
        report.status = "partial"
        text = _format_report(report)
        self.assertIn("Partial report: unresolved research gaps remain.", text)
        self.assertIn(report.summary, text)
        self.assertNotIn("Research failed", text)

    def test_partial_report_displays_warning_without_losing_content(self):
        report = _report()
        report.status = "partial"
        report.limitations = ["Missing supplier disclosures."]
        restored = ResearchReport.from_dict(report.to_dict())
        md = _report_to_markdown(restored)
        self.assertIn("Partial report: unresolved research gaps remain.", md)
        self.assertIn(report.summary, md)
        self.assertIn(report.limitations[0], md)

    def test_markdown_preserves_formatting(self):
        md = _report_to_markdown(_report())
        self.assertIn("**bold claim**", md)  # markdown text keeps the syntax
        self.assertIn("## Summary", md)
        self.assertIn("Example", md)

    def test_failed_report_is_not_rendered_as_completed_research(self):
        rep = ResearchReport(
            "q", "", [], [], status="failed", error="provider unavailable"
        )
        md = _report_to_markdown(rep)
        self.assertIn("Research failed", md)
        self.assertIn("provider unavailable", md)
        self.assertNotIn("## Summary", md)


class TestVerificationRendering(unittest.TestCase):
    def test_unverified_marker_in_markdown(self):
        from sibyl.report_models import FindingVerification

        rep = _report()
        rep.finding_verifications = [
            FindingVerification(1, True, "high", [1], ""),
            FindingVerification(2, False, "low", [], "no support"),
        ]
        before = rep.to_dict()
        for render in (_report_to_markdown, _format_report):
            text = render(rep)
            self.assertIn("historical: supported; recorded confidence: high", text)
            self.assertIn("historical: unverified", text)
            self.assertIn("not revalidated on read or export", text)
            self.assertNotIn("[verified:", text)
        self.assertEqual(rep.to_dict(), before)
        self.assertEqual(ResearchReport.from_dict(before).to_dict(), before)

    @unittest.skipUnless(find_spec("pypdf"), "PDF reader extra is optional")
    def test_pdf_labels_old_verdicts_without_mutating_saved_report(self):
        from pypdf import PdfReader

        from sibyl.report_models import FindingVerification

        rep = _report()
        rep.finding_verifications = [FindingVerification(1, True, "high", [1], "")]
        before = rep.to_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = generate_pdf(rep, directory)
            text = "\n".join(page.extract_text() for page in PdfReader(path).pages)
        self.assertIn("Historical recorded verdicts", text)
        self.assertIn("not revalidated on read or export", text)
        self.assertIn("historical: supported; recorded confidence: high", text)
        self.assertNotIn("[verified:", text)
        self.assertEqual(rep.to_dict(), before)

    def test_empty_verifications_no_markers(self):
        rep = _report()
        rep.finding_verifications = []
        md = _report_to_markdown(rep)
        self.assertNotIn("(unverified)", md)
        self.assertNotIn("confidence:", md)

    def test_length_mismatch_no_crash(self):
        from sibyl.report_models import FindingVerification

        rep = _report()  # 2 findings
        rep.finding_verifications = [
            FindingVerification(1, True, "high", [1], "")
        ]  # only 1
        md = _report_to_markdown(rep)  # must not IndexError
        self.assertIn("confidence: high", md)


class TestPdfGeneration(unittest.TestCase):
    @unittest.skipUnless(find_spec("pypdf"), "PDF reader extra is optional")
    def test_claim_pdf_retains_each_statement_once_with_the_fallback_font(self):
        from pypdf import PdfReader

        report = _claim_report()
        before = report.to_dict()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("fpdf.FPDF.add_font", side_effect=OSError("font unavailable")),
        ):
            path = generate_pdf(report, directory)
            text = "\n".join(page.extract_text() for page in PdfReader(path).pages)
        for claim in report.claims:
            self.assertEqual(text.count(claim["text"]), 1)
        for expected in (
            "Basis: Q0.C1",
            "Criterion (user)",
            "not been benchmarked",
            "citation_targets_only",
            "Partial report",
            "[Source 1] Example",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("Key Findings", text)
        self.assertEqual(report.to_dict(), before)

    def test_unrenderable_text_fails_without_saving_an_incomplete_pdf(self):
        report = _claim_report()
        report.analysis += "\n缺少工作负载数据。"
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("fpdf.FPDF.add_font", side_effect=OSError("font unavailable")),
        ):
            with self.assertRaisesRegex(ValueError, "required report text"):
                generate_pdf(report, directory)
            self.assertEqual(list(Path(directory).glob("*.pdf")), [])

    def test_missing_unicode_glyph_fails_instead_of_exporting_blank_text(self):
        report = _claim_report()
        report.analysis += "\nUnsupported glyph: \U0010ffff"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "required report text"):
                generate_pdf(report, directory)
            self.assertEqual(list(Path(directory).glob("*.pdf")), [])

    def test_render_failure_manifest_retains_markdown_and_actionable_reason(self):
        report = _claim_report()
        report.analysis += "\n缺少工作负载数据。"
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("fpdf.FPDF.add_font", side_effect=OSError("font unavailable")),
        ):
            manifest = export_report(report, Path(directory), ".", "both", "test-job")
            self.assertEqual(manifest.status, "partial")
            self.assertEqual([a.format for a in manifest.artifacts], ["md"])
            self.assertEqual([f.format for f in manifest.failures], ["pdf"])
            self.assertIn("PDF font cannot render", manifest.failures[0].message)
            self.assertIn("Markdown/JSON", manifest.failures[0].message)
            self.assertIn(
                "缺少工作负载数据",
                Path(manifest.artifacts[0].path).read_text(encoding="utf-8"),
            )
            self.assertEqual(list(Path(directory).glob("*.pdf")), [])

    @unittest.skipUnless(
        shutil.which("pdftotext"), "Poppler text extractor is optional"
    )
    def test_pdf_separates_background_without_renumbering(self):
        report = _report()
        report.sources.insert(
            0, Source("https://ex.com/background", "Background only", "")
        )
        report.summary = "Summary [Source 2]"
        report.key_findings = ["Finding [Source 2]"]
        with tempfile.TemporaryDirectory() as directory:
            path = generate_pdf(report, directory)
            text = subprocess.run(
                ["pdftotext", path, "-"], capture_output=True, text=True, check=True
            ).stdout
        self.assertIn("Cited Sources (1)", text)
        self.assertIn("Uncited Background (1)", text)
        self.assertIn("[Source 2] Example", text)
        self.assertIn("[Source 1] Background only", text)

    @unittest.skipUnless(
        shutil.which("pdftotext"), "Poppler text extractor is optional"
    )
    def test_partial_pdf_keeps_warning_and_findings(self):
        report = _report()
        report.status = "partial"
        with tempfile.TemporaryDirectory() as directory:
            path = generate_pdf(report, directory)
            text = subprocess.run(
                ["pdftotext", path, "-"], capture_output=True, text=True, check=True
            ).stdout
        self.assertIn("Partial report: unresolved research gaps remain.", text)
        self.assertIn("Finding one", text)

    @unittest.skipUnless(
        shutil.which("pdftotext"), "Poppler text extractor is optional"
    )
    def test_consecutive_bullets_and_limitations_are_not_silently_lost(self):
        report = _report()
        report.limitations = [
            "First limitation retained.",
            "Second limitation retained.",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = generate_pdf(report, directory)
            extracted = subprocess.run(
                ["pdftotext", path, "-"], capture_output=True, text=True, check=True
            ).stdout
        for expected in ("second bullet", "Closing paragraph.", *report.limitations):
            self.assertIn(expected, extracted)

    def assert_valid_pdf(self, path):
        data = Path(path).read_bytes()
        self.assertGreater(len(data), 1000)
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertTrue(data.rstrip().endswith(b"%%EOF"))
        self.assertIn(b"/Type /Page", data)

    def test_pdf_generates_without_crashing(self):
        with tempfile.TemporaryDirectory() as d:
            path = generate_pdf(_report(), d)
            self.assertTrue(os.path.exists(path))
            self.assert_valid_pdf(path)

    def test_pdf_falls_back_when_system_font_is_unavailable(self):
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch("fpdf.FPDF.add_font", side_effect=OSError("font unavailable")),
        ):
            path = generate_pdf(_report(), d)
            self.assert_valid_pdf(path)

    def test_pdf_handles_empty_optional_sections(self):
        rep = _report()
        rep.analysis = ""
        rep.key_findings = []
        with tempfile.TemporaryDirectory() as d:
            path = generate_pdf(rep, d)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
