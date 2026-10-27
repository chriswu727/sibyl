"""Offline regression tests for isolation, admission, cancellation, and artifacts."""

import asyncio
import gzip
import json
import os
import stat
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import httpx

from sibyl.async_cache import AsyncSingleFlightTTL, CapacityError
from sibyl.mcp_server import create_server
from sibyl.privacy import display_url
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
    def runtime(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        runtime = SibylRuntime(RuntimePolicy(output_root=Path(directory.name)))
        self.addAsyncCleanup(runtime.aclose)
        return runtime, Path(directory.name)

    async def call(self, server, name, error=False, **arguments):
        result = await server.call_tool(name, arguments)
        self.assertEqual(bool(result.isError), error, str(result))
        self.assertEqual(json.loads(result.content[0].text), result.structuredContent)
        return result.structuredContent

    async def finalized(self, server):
        """Finalize a report through the public tools only."""
        job = await self.call(server, "start_research", question="question")
        job = await self.call(
            server,
            "assess_research",
            job_id=job["job_id"],
            revision=job["revision"],
            question_id="Q0",
            status="unresolved",
            conclusion="Unknown.",
            counterevidence_note="No usable evidence was found.",
            claims=[
                {
                    "claim_id": "Q0.C1",
                    "kind": "fact",
                    "status": "unresolved",
                    "text": "The exported outcome is unknown.",
                    "rationale": "No observed outcome is available.",
                }
            ],
        )
        await self.call(
            server,
            "complete_research",
            job_id=job["job_id"],
            revision=job["revision"],
            allow_incomplete=True,
        )
        return job["job_id"]

    async def test_pdf_only_export_failure_is_a_structured_error(self):
        runtime, directory = self.runtime()
        server = create_server(runtime)
        job_id = await self.finalized(server)
        with mock.patch(
            "sibyl.reporter.generate_pdf",
            side_effect=RuntimeError("token=SECRET"),
        ):
            result = await self.call(
                server, "save_report", error=True, job_id=job_id, format="pdf"
            )
        self.assertEqual(result["view"], "export")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["artifacts"], [])
        self.assertEqual([item["format"] for item in result["failures"]], ["pdf"])
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertFalse(list(directory.glob("*.pdf")))
        self.assertEqual(runtime.failed, 1)

    async def test_markdown_and_json_need_separate_exports_and_return_real_paths(self):
        runtime, directory = self.runtime()
        server = create_server(runtime)
        job_id = await self.finalized(server)
        markdown = await self.call(server, "save_report", job_id=job_id, format="md")
        self.assertEqual(markdown["requested_formats"], ["md"])
        self.assertEqual([item["format"] for item in markdown["artifacts"]], ["md"])
        self.assertFalse(list(directory.glob("*.json")))
        md_path = Path(markdown["artifacts"][0]["path"])
        self.assertTrue(md_path.is_file())
        exported = await self.call(server, "save_report", job_id=job_id, format="json")
        json_path = Path(exported["artifacts"][0]["path"])
        self.assertTrue(json_path.is_file())
        self.assertNotEqual(json_path, md_path.with_suffix(".json"))
        self.assertEqual(
            json.loads(json_path.read_text(encoding="utf-8"))["claims"][0]["text"],
            "The exported outcome is unknown.",
        )

    async def test_export_uses_runtime_root_and_rejects_escape(self):
        runtime, directory = self.runtime()
        server = create_server(runtime)
        job_id = await self.finalized(server)
        await self.call(
            server, "save_report", job_id=job_id, format="md", output_dir="nested"
        )
        files = list((directory / "nested").glob("*.md"))
        self.assertEqual(len(files), 1)
        self.assertIn(
            "The exported outcome is unknown.", files[0].read_text(encoding="utf-8")
        )
        before = sorted(directory.parent.iterdir())
        escaped = await self.call(
            server,
            "save_report",
            error=True,
            job_id=job_id,
            format="md",
            output_dir="../escape",
        )
        self.assertNotIn("artifacts", escaped)
        self.assertFalse((directory.parent / "escape").exists())
        self.assertEqual(sorted(directory.parent.iterdir()), before)

    async def test_oversized_input_is_rejected_before_provider(self):
        runtime, _ = self.runtime()
        server = create_server(runtime)
        job = await self.call(server, "start_research", question="q")
        search = mock.AsyncMock()
        read = mock.AsyncMock()
        with (
            mock.patch("sibyl.mcp_tools.gather_source_bundle", new=search),
            mock.patch("sibyl.research_jobs.gather_source_bundle", new=search),
            mock.patch("sibyl.research_jobs.read_sources", new=read),
        ):
            for name, arguments in (
                ("gather_bundle", {"query": "x" * 10001}),
                ("continue_research", {"job_id": job["job_id"], "query": "x" * 10001}),
                (
                    "continue_research",
                    {
                        "job_id": job["job_id"],
                        "query": "q",
                        "url": "https://example.com/" + "x" * 10001,
                    },
                ),
            ):
                with self.subTest(name=name, fields=sorted(arguments)):
                    result = await self.call(server, name, error=True, **arguments)
                    self.assertIn("10000 characters", result["error"]["message"])
                    self.assertNotIn("x" * 100, json.dumps(result))
        search.assert_not_awaited()
        read.assert_not_awaited()
        current = runtime.research_jobs.get(job["job_id"])
        self.assertEqual(current["attempts"], [])
        self.assertEqual(current["revision"], job["revision"])

    async def test_source_read_redacts_metadata_without_changing_evidence(self):
        from sibyl.scraper import WebPage

        runtime, _ = self.runtime()
        server = create_server(runtime)
        job = await self.call(server, "start_research", question="q")
        page = WebPage(
            "https://example.com/?token=SECRET", "title", "original evidence " * 20
        )
        with mock.patch(
            "sibyl.research_sources.scrape_url", new=mock.AsyncMock(return_value=page)
        ):
            brief = await self.call(
                server,
                "continue_research",
                job_id=job["job_id"],
                url=page.url,
                query="original evidence",
            )
        full = await self.call(
            server, "get_research", job_id=job["job_id"], view="full"
        )
        passage_id = full["source_catalog"][0]["passage_ids"][0]
        passage = await self.call(
            server, "get_research", job_id=job["job_id"], passage_id=passage_id
        )
        for result in (brief, full, passage):
            self.assertNotIn("SECRET", json.dumps(result))
        self.assertIn("token=REDACTED", full["source_catalog"][0]["url"])
        self.assertTrue(passage["passage"]["text"].startswith("original evidence"))

    async def test_capacity_deadline_and_shutdown(self):
        runtime = SibylRuntime(RuntimePolicy(max_active=1, tool_timeout=0.05))
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.Event().wait()

        first = asyncio.create_task(runtime.execute("gather_bundle", slow))
        await started.wait()
        with self.assertRaises(CapacityError):
            await runtime.execute("gather_bundle", slow)
        with self.assertRaises(asyncio.TimeoutError):
            await first
        self.assertEqual(runtime.failed, 1)
        await runtime.aclose()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await runtime.execute("gather_bundle", slow)

    async def test_servers_do_not_share_reports_or_caches(self):
        (one, _), (two, root) = self.runtime(), self.runtime()
        self.assertIsNot(one.bundle_cache, two.bundle_cache)
        job_id = await self.finalized(create_server(one))
        server = create_server(two)
        missing = await self.call(server, "save_report", error=True, format="md")
        self.assertEqual(missing["error"]["code"], "invalid_parameters")
        self.assertEqual(missing["error"]["field"], "job_id")
        other = await self.call(
            server, "save_report", error=True, job_id=job_id, format="md"
        )
        self.assertNotIn("artifacts", other)
        self.assertEqual(list(root.glob("sibyl_*")), [])

    async def test_provider_errors_do_not_leak_secrets(self):
        runtime, _ = self.runtime()
        server = create_server(runtime)
        job = await self.call(server, "start_research", question="q")
        for exception, code in (
            (ValueError, "invalid_request"),
            (RuntimeError, "operation_failed"),
        ):
            failure = mock.AsyncMock(side_effect=exception("https://h/?token=SECRET"))
            with (
                self.subTest(exception=exception.__name__),
                mock.patch("sibyl.mcp_tools.gather_source_bundle", new=failure),
                mock.patch("sibyl.research_jobs.gather_source_bundle", new=failure),
            ):
                bundle = await self.call(server, "gather_bundle", error=True, query="q")
                attempt = await self.call(
                    server, "continue_research", job_id=job["job_id"], query="q"
                )
                self.assertEqual(bundle["error"]["code"], code)
                self.assertEqual(attempt["attempt"]["status"], "failed")
                for result in (bundle, attempt):
                    self.assertNotIn("SECRET", json.dumps(result))

    async def test_report_operations_cannot_race(self):
        runtime = SibylRuntime()
        started = asyncio.Event()
        release = asyncio.Event()

        async def report():
            started.set()
            await release.wait()

        first = asyncio.create_task(runtime.execute("save_report", report))
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

        task = asyncio.create_task(runtime.execute("gather_bundle", operation))
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
    def test_quotas_reject_without_removing_existing_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            store.MAX_ARTIFACT_BYTES = 4
            with self.assertRaises(CapacityError):
                store.write(b"large", "report", ".md")
            first = store.write(b"one", "report", ".md")
            store.MAX_DIRECTORY_BYTES = 4
            with self.assertRaises(CapacityError):
                store.write(b"two", "report", ".md")
            store.MAX_DIRECTORY_BYTES = 100
            store.MAX_ARTIFACTS = 1
            with self.assertRaises(CapacityError):
                store.write(b"two", "report", ".md")
            self.assertEqual(first.read_bytes(), b"one")
            self.assertFalse(list(Path(directory).glob(".sibyl-*")))

    def test_symlinked_staging_and_lock_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            target = Path(directory) / "original"
            target.write_bytes(b"original")
            link = Path(directory) / "link"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                store.publish(link, "report", ".md")
            (Path(directory) / ".artifact.lock").symlink_to(target)
            with self.assertRaises(ValueError):
                store.write(b"new", "report", ".md")
            self.assertEqual(target.read_bytes(), b"original")

    def test_private_unique_files_and_no_temporary_leftovers(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            first = store.write(b"one", "report", ".md")
            second = store.write(b"two", "report", ".md")
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b"one")
            self.assertEqual(len(list(Path(directory).glob("sibyl_*"))), 2)
            self.assertFalse(list(Path(directory).glob(".sibyl-*")))
            if os.name == "posix":  # Windows has no POSIX modes; profile ACLs apply.
                lock = Path(directory) / ".artifact.lock"
                self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(lock.stat().st_mode), 0o600)

    def test_escape_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ("../escape", "/tmp", "\\tmp", "C:escape", "C:/tmp"):
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

    def test_url_metadata_redacts_credentials_but_preserves_query(self):
        result = display_url(
            "https://user:pass@example.com/p?q=hello&access_token=SECRET#SECRET"
        )
        self.assertNotIn("SECRET", result)
        self.assertNotIn("user:pass", result)
        self.assertIn("q=hello", result)
