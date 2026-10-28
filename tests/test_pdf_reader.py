"""PDF extraction boundaries, provenance, and subprocess cleanup."""

import asyncio
import io
import os
import unittest
from importlib.util import find_spec
from unittest import mock

from sibyl.pdf_reader import read_pdf
from sibyl.scraper import _ResponseSnapshot, scrape_url


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
