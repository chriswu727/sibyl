"""Reporter smoke tests — PDF generates and markdown is preserved. No network.

Run: python -m unittest discover tests
"""
import os
import tempfile
import unittest

from sibyl.researcher import ResearchReport, Source
from sibyl.reporter import generate_pdf, _report_to_markdown


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
    def test_markdown_preserves_formatting(self):
        md = _report_to_markdown(_report())
        self.assertIn("**bold claim**", md)   # markdown text keeps the syntax
        self.assertIn("## Summary", md)
        self.assertIn("Example", md)


class TestPdfGeneration(unittest.TestCase):
    def test_pdf_generates_without_crashing(self):
        with tempfile.TemporaryDirectory() as d:
            path = generate_pdf(_report(), d)
            self.assertTrue(os.path.exists(path))
            # A real multi-section report renders to a non-trivial file.
            self.assertGreater(os.path.getsize(path), 5000)

    def test_pdf_handles_empty_optional_sections(self):
        rep = _report()
        rep.analysis = ""
        rep.key_findings = []
        with tempfile.TemporaryDirectory() as d:
            path = generate_pdf(rep, d)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
