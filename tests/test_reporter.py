"""Reporter smoke tests — PDF generates and markdown is preserved. No network.

Run: python -m unittest discover tests
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sibyl.reporter import _report_to_markdown, generate_pdf
from sibyl.researcher import ResearchReport, Source


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


class TestMarkdownReport(unittest.TestCase):
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
        from sibyl.verifier import FindingVerification

        rep = _report()
        rep.finding_verifications = [
            FindingVerification(1, True, "high", [1], ""),
            FindingVerification(2, False, "low", [], "no support"),
        ]
        md = _report_to_markdown(rep)
        self.assertIn("confidence: high", md)
        self.assertIn("(unverified)", md)

    def test_empty_verifications_no_markers(self):
        rep = _report()
        rep.finding_verifications = []
        md = _report_to_markdown(rep)
        self.assertNotIn("(unverified)", md)
        self.assertNotIn("confidence:", md)

    def test_length_mismatch_no_crash(self):
        from sibyl.verifier import FindingVerification

        rep = _report()  # 2 findings
        rep.finding_verifications = [
            FindingVerification(1, True, "high", [1], "")
        ]  # only 1
        md = _report_to_markdown(rep)  # must not IndexError
        self.assertIn("confidence: high", md)


class TestPdfGeneration(unittest.TestCase):
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
