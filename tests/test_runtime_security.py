"""Offline regression tests for isolation, admission, cancellation, and artifacts."""

import asyncio
import gzip
import stat
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import httpx
from mcp.server.fastmcp.exceptions import ToolError

from sibyl.async_cache import AsyncSingleFlightTTL, CapacityError
from sibyl.config import Config, Provider
from sibyl.mcp_server import create_server
from sibyl.privacy import display_url
from sibyl.report_models import ResearchReport
from sibyl.runtime import RuntimePolicy, SibylRuntime
from sibyl.scraper import ResponseTooLargeError, _bounded_get
from sibyl.storage import ArtifactStore


class Stream(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data

    async def __aiter__(self):
        yield self.data


class TestCompressedBodies(unittest.IsolatedAsyncioTestCase):
    async def fetch(self, body, encoding, limit):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-encoding": encoding}, stream=Stream(body)
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            return await _bounded_get(
                client,
                "https://example.com",
                headers={},
                timeout=1,
                max_response_bytes=limit,
            )

    async def test_compressed_bomb_is_rejected_with_bounded_allocation(self):
        import tracemalloc

        body = gzip.compress(b"x" * (8 * 1024 * 1024))
        tracemalloc.start()
        try:
            with self.assertRaises(ResponseTooLargeError):
                await self.fetch(body, "gzip", 4096)
            _, peak = tracemalloc.get_traced_memory()
            self.assertLess(peak, 1024 * 1024)
        finally:
            tracemalloc.stop()

    async def test_gzip_and_zlib_deflate_are_supported(self):
        for encoding, compress in (("gzip", gzip.compress), ("deflate", zlib.compress)):
            result = await self.fetch(compress(b"hello world"), encoding, 100)
            self.assertEqual(result.text, "hello world")

    async def test_truncated_and_stacked_encodings_fail_closed(self):
        with self.assertRaises(ValueError):
            await self.fetch(gzip.compress(b"hello")[:-4], "gzip", 100)
        with self.assertRaises(ValueError):
            await self.fetch(b"hello", "gzip, deflate", 100)


class TestRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_export_uses_runtime_root_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SibylRuntime(RuntimePolicy(output_root=Path(directory)))
            runtime.last_report = ResearchReport("question", "summary", [], [])
            server = create_server(runtime)
            await server.call_tool(
                "save_report", {"format": "md", "output_dir": "nested"}
            )
            files = list((Path(directory) / "nested").glob("*.md"))
            self.assertEqual(len(files), 1)
            self.assertIn("summary", files[0].read_text())
            with self.assertRaises(ToolError):
                await server.call_tool(
                    "save_report", {"format": "md", "output_dir": "../escape"}
                )
            await runtime.aclose()

    async def test_oversized_input_is_rejected_before_provider(self):
        server = create_server(SibylRuntime())
        with mock.patch("sibyl.search.search_web", new=mock.AsyncMock()) as search:
            with self.assertRaises(ToolError):
                await server.call_tool("quick_search", {"query": "x" * 10001})
            search.assert_not_awaited()

    async def test_read_url_redacts_metadata_without_changing_evidence(self):
        from sibyl.scraper import WebPage

        server = create_server(SibylRuntime())
        page = WebPage(
            "https://example.com/?token=SECRET", "title", "original evidence " * 20
        )
        with mock.patch(
            "sibyl.scraper.scrape_url", new=mock.AsyncMock(return_value=page)
        ):
            result = await server.call_tool("read_url", {"url": page.url})
        rendered = str(result)
        self.assertNotIn("SECRET", rendered)
        self.assertIn("original evidence", rendered)

    async def test_capacity_deadline_and_shutdown(self):
        runtime = SibylRuntime(RuntimePolicy(max_active=1, tool_timeout=0.05))
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.Event().wait()

        first = asyncio.create_task(runtime.execute("read_url", slow))
        await started.wait()
        with self.assertRaises(CapacityError):
            await runtime.execute("read_url", slow)
        with self.assertRaises(asyncio.TimeoutError):
            await first
        self.assertEqual(runtime.failed, 1)
        await runtime.aclose()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await runtime.execute("read_url", slow)

    async def test_servers_do_not_share_reports_or_loops(self):
        one, two = SibylRuntime(), SibylRuntime()
        one.last_report = ResearchReport("q", "summary", [], [])
        self.assertIsNone(two.last_report)
        self.assertIsNot(one.evidence_loops, two.evidence_loops)
        self.assertIsNot(one.bundle_cache, two.bundle_cache)
        server = create_server(two)
        with self.assertRaisesRegex(ToolError, "No research report"):
            await server.call_tool("save_report", {})
        await one.aclose()
        await two.aclose()

    async def test_provider_errors_do_not_leak_secrets(self):
        server = create_server(SibylRuntime())
        with mock.patch(
            "sibyl.search.search_web", side_effect=ValueError("token=SECRET")
        ):
            with self.assertRaises(ToolError) as raised:
                await server.call_tool("quick_search", {"query": "q"})
        self.assertNotIn("SECRET", str(raised.exception))

    async def test_report_operations_cannot_race(self):
        runtime = SibylRuntime()
        started = asyncio.Event()
        release = asyncio.Event()

        async def report():
            started.set()
            await release.wait()

        first = asyncio.create_task(runtime.execute("research", report))
        await started.wait()
        with self.assertRaises(CapacityError):
            await runtime.execute("save_report", report)
        release.set()
        await first
        await runtime.aclose()

    async def test_shutdown_cancels_active_operations(self):
        runtime = SibylRuntime()
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def operation():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        task = asyncio.create_task(runtime.execute("read_url", operation))
        await started.wait()
        await runtime.aclose()
        self.assertTrue(task.cancelled())
        self.assertTrue(stopped.is_set())

    async def test_cache_capacity_and_last_waiter_cancellation(self):
        cache = AsyncSingleFlightTTL(30, 2, max_inflight=1)
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def factory():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        task = asyncio.create_task(cache.get_or_create("one", factory))
        await started.wait()
        with self.assertRaises(CapacityError):
            await cache.get_or_create("two", factory)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stopped.is_set())
        self.assertEqual(
            await cache.get_or_create("two", mock.AsyncMock(return_value=42)), 42
        )
        await cache.aclose()


class TestArtifacts(unittest.TestCase):
    def test_private_unique_files_and_no_temporary_leftovers(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            first = store.write(b"one", "report", ".md")
            second = store.write(b"two", "report", ".md")
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b"one")
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o600)
            self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_escape_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ("../escape", "/tmp"):
                with self.assertRaises(ValueError):
                    ArtifactStore(directory, value)
            (Path(directory) / "link").symlink_to(Path(directory).parent)
            with self.assertRaises(ValueError):
                ArtifactStore(directory, "link/child")

    def test_staging_is_cleaned_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaises(RuntimeError):
                with store.staging(".md"):
                    raise RuntimeError("failed")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_credentials_are_hidden_in_configuration_repr(self):
        config = Config(
            providers=[Provider("model", api_key="SECRET", api_base="http://SECRET")]
        )
        self.assertNotIn("SECRET", repr(config))

    def test_url_metadata_redacts_credentials_but_preserves_query(self):
        result = display_url(
            "https://user:pass@example.com/p?q=hello&access_token=SECRET#SECRET"
        )
        self.assertNotIn("SECRET", result)
        self.assertNotIn("user:pass", result)
        self.assertIn("q=hello", result)
