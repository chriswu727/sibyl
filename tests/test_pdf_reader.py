"""PDF extraction boundaries, provenance, and subprocess cleanup."""

import asyncio
import io
import os
import unittest
from importlib.util import find_spec
from unittest import mock

from sibyl.pdf_reader import read_pdf
from sibyl.research_sources import read_source_bundle
from sibyl.research_state import ResearchNotebook
from sibyl.scraper import WebPage, _ResponseSnapshot, scrape_url


def document(
    pages=1, text="Evidence from an original publication. " * 20, encrypted=False
):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=600, height=800)
        if text:
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 20 700 Td ({text}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata({"/Title": "Original publication"})
    if encrypted:
        writer.encrypt("password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@unittest.skipUnless(
    find_spec("pypdf") and os.name == "posix", "PDF extra and POSIX required"
)
class TestPDF(unittest.IsolatedAsyncioTestCase):
    async def test_real_worker_text_title_and_page_markers(self):
        result = await read_pdf(document(pages=2), 4000)
        self.assertNotIn("error", result)
        self.assertEqual(result["title"], "Original publication")
        self.assertIn("[PDF page 2]", result["text"])
        self.assertIn("original publication", result["text"])

    async def test_truncation_page_limit_encryption_and_blank(self):
        result = await read_pdf(document(pages=2), 200)
        self.assertEqual(len(result["text"]), 200)
        self.assertTrue(any("truncated" in item for item in result["warnings"]))
        result = await read_pdf(document(pages=41, text="x"), 60000)
        self.assertIn("[PDF page 40]", result["text"])
        self.assertNotIn("[PDF page 41]", result["text"])
        self.assertTrue(any("first 40" in item for item in result["warnings"]))
        self.assertIn(
            "Encrypted", (await read_pdf(document(encrypted=True), 4000))["error"]
        )
        self.assertIn("OCR", (await read_pdf(document(text=""), 4000))["error"])

    async def test_invalid_and_oversized_inputs(self):
        for data in (b"not pdf", b"%PDF-invalid", b"%PDF-" + b"x" * (2 * 1024 * 1024)):
            self.assertIn("error", await read_pdf(data, 1000))

    async def test_fetch_routes_pdf_magic_without_binary_html(self):
        response = _ResponseSnapshot(
            200, {"content-type": "application/octet-stream"}, "", document()
        )
        with (
            mock.patch(
                "sibyl.scraper.resolve_public_url",
                mock.AsyncMock(return_value=(None, ["8.8.8.8"])),
            ),
            mock.patch(
                "sibyl.scraper._bounded_get", mock.AsyncMock(return_value=response)
            ),
        ):
            page = await scrape_url("https://example.com/document", max_chars=200)
        self.assertEqual(page.source_type, "pdf")
        self.assertIsNone(page.error)
        self.assertTrue(page.extraction_warnings)


class TestPDFLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_kills_and_reaps_worker(self):
        process = mock.Mock(returncode=None)
        process.communicate = mock.AsyncMock(side_effect=asyncio.TimeoutError)
        process.wait = mock.AsyncMock(return_value=-9)
        with (
            mock.patch("sibyl.pdf_reader.os.name", "posix"),
            mock.patch("sibyl.pdf_reader.sys.platform", "linux"),
            mock.patch(
                "sibyl.pdf_reader.asyncio.create_subprocess_exec",
                mock.AsyncMock(return_value=process),
            ),
        ):
            self.assertIn("timed out", (await read_pdf(b"%PDF-test", 1000))["error"])
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()

    async def test_source_read_discloses_pdf_warnings_and_failures(self):
        import json
        import tempfile
        from pathlib import Path

        from sibyl.config import Config
        from sibyl.mcp_server import create_server
        from sibyl.runtime import RuntimePolicy, SibylRuntime

        with tempfile.TemporaryDirectory() as directory:
            runtime = SibylRuntime(RuntimePolicy(output_root=Path(directory)), Config())
            server = create_server(runtime)
            try:
                job = (
                    await server.call_tool("start_research", {"question": "q"})
                ).structuredContent
                page = WebPage(
                    "https://example.com/paper.pdf",
                    "Paper",
                    "evidence " * 100,
                    source_type="pdf",
                    extraction_warnings=("PDF text was truncated.",),
                )
                arguments = {
                    "job_id": job["job_id"],
                    "url": page.url,
                    "query": "evidence",
                }
                with mock.patch(
                    "sibyl.research_sources.scrape_url",
                    mock.AsyncMock(return_value=page),
                ):
                    result = await server.call_tool("continue_research", arguments)
                self.assertFalse(result.isError)
                self.assertEqual(
                    result.structuredContent["latest_attempt"]["status"], "ok"
                )
                self.assertIn("PDF text was truncated", result.content[0].text)
                full = await server.call_tool(
                    "get_research", {"job_id": job["job_id"], "view": "full"}
                )
                self.assertIn(
                    "PDF text was truncated", json.dumps(full.structuredContent)
                )
                page.error = "Encrypted PDF is not supported."
                with mock.patch(
                    "sibyl.research_sources.scrape_url",
                    mock.AsyncMock(return_value=page),
                ):
                    result = await server.call_tool("continue_research", arguments)
                attempt = result.structuredContent["latest_attempt"]
                self.assertNotEqual(attempt["status"], "ok")
                self.assertEqual(attempt["substantive_sources"], 0)
                self.assertIn("Encrypted PDF", attempt["error"])
            finally:
                await runtime.aclose()

    async def test_cancel_kills_and_reaps_worker_without_secrets(self):
        process = mock.Mock(returncode=None)
        entered = asyncio.Event()

        async def blocked(data):
            entered.set()
            await asyncio.Event().wait()

        process.communicate = blocked
        process.wait = mock.AsyncMock(return_value=-9)
        with (
            mock.patch("sibyl.pdf_reader.os.name", "posix"),
            mock.patch("sibyl.pdf_reader.sys.platform", "linux"),
            mock.patch(
                "sibyl.pdf_reader.asyncio.create_subprocess_exec",
                mock.AsyncMock(return_value=process),
            ) as spawn,
        ):
            task = asyncio.create_task(read_pdf(b"%PDF-test", 1000))
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()
        self.assertEqual(spawn.call_args.kwargs["env"], {})
        self.assertIn("-I", spawn.call_args.args)

    async def test_warnings_and_pdf_type_survive_evidence_and_report_limits(self):
        warning = "PDF text was truncated at the extraction character limit."
        page = WebPage(
            "https://example.com/a.pdf",
            "Paper",
            "[PDF page 1]\n" + "evidence " * 100,
            source_type="pdf",
            extraction_warnings=(warning,),
        )
        with mock.patch(
            "sibyl.research_sources.scrape_url", mock.AsyncMock(return_value=page)
        ):
            bundle = await read_source_bundle(page.url, "question")
        self.assertEqual(bundle.sources[0].source_type, "pdf")
        self.assertIn("[PDF page 1]", bundle.sources[0].evidence[0].text)
        self.assertIn(warning, ResearchNotebook("q", bundles=[bundle]).limitations())
