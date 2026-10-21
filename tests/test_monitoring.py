"""Authenticated, content-free monitoring over ASGI and a real loopback socket."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from sibyl.async_cache import CapacityError
from sibyl.config import Config
from sibyl.monitoring import MonitorApp
from sibyl.runtime import RuntimePolicy, SibylRuntime

TOKEN = "test-monitor-token-with-at-least-32-characters"


class TestMonitoring(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.runtime = SibylRuntime(RuntimePolicy(output_root=Path(self.directory.name), max_active=1), Config())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=MonitorApp(self.runtime, 8766, TOKEN)),
                                       base_url="http://127.0.0.1:8766", headers={"Authorization": f"Bearer {TOKEN}"})

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.runtime.aclose()
        self.directory.cleanup()

    async def test_authorization_origin_host_method_and_query(self):
        for headers, status in [({"Authorization": ""}, 401), ({"Authorization": "Bearer wrong"}, 401),
                                ({"Origin": "https://evil.example"}, 400), ({"Host": "evil.example"}, 400)]:
            result = await self.client.get("/status", headers=headers)
            self.assertEqual(result.status_code, status)
        self.assertEqual((await self.client.post("/status")).status_code, 405)
        self.assertEqual((await self.client.get("/status?token=secret")).status_code, 400)
        self.assertEqual((await self.client.get("/unknown")).status_code, 404)
        self.assertIsNone(self.runtime._research_jobs)
        self.assertFalse(list(Path(self.directory.name).iterdir()))

    async def test_status_is_content_free_and_reports_store_failure(self):
        self.runtime.research_jobs.start("PRIVATE question", [], "host", 1, "en", 2)
        result = await self.client.get("/status")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["jobs"]["states"], {"awaiting_input": 1})
        self.assertNotIn("PRIVATE", result.text)
        self.assertNotIn(TOKEN, result.text)
        self.assertNotIn(self.directory.name, result.text)
        with mock.patch.object(self.runtime.research_jobs.store, "summary", side_effect=OSError("PRIVATE")):
            self.assertEqual((await self.client.get("/readyz")).status_code, 503)
            self.assertEqual((await self.client.get("/healthz")).status_code, 200)
            self.assertEqual((await self.client.get("/status")).status_code, 503)
            self.assertIn("sibyl_store_healthy 0", (await self.client.get("/metrics")).text)

    async def test_busy_readiness_metrics_and_cancellation(self):
        entered = asyncio.Event()

        async def wait():
            entered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(self.runtime.execute("gather_bundle", wait))
        await entered.wait()
        self.assertEqual((await self.client.get("/readyz")).status_code, 503)
        with self.assertRaises(CapacityError):
            await self.runtime.execute("gather_bundle", wait)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await self.runtime.execute("get_research", mock.AsyncMock(return_value={"secret": "PRIVATE"}))
        metrics = (await self.client.get("/metrics")).text
        self.assertIn('outcome="rejected"} 1', metrics)
        self.assertIn('outcome="cancelled"} 1', metrics)
        self.assertIn('sibyl_tool_duration_seconds_count{tool="get_research"} 1', metrics)
        self.assertNotIn("PRIVATE", json.dumps(self.runtime.snapshot()))
        self.assertEqual((await self.client.get("/readyz")).status_code, 200)

    async def test_real_listener_binds_loopback_and_closes(self):
        runtime = SibylRuntime(RuntimePolicy(output_root=Path(self.directory.name), monitor_port=0), Config())
        with mock.patch.dict(os.environ, {"SIBYL_MONITOR_TOKEN": TOKEN}):
            await asyncio.gather(runtime.start(), runtime.start())
        self.addAsyncCleanup(runtime.aclose)
        port = runtime.monitor.port
        self.assertEqual(runtime.monitor.socket.getsockname()[0], "127.0.0.1")
        async with httpx.AsyncClient(trust_env=False) as client:
            url = f"http://127.0.0.1:{port}/healthz"
            self.assertEqual((await client.get(url)).status_code, 401)
            response = await client.get(url, headers={"Authorization": f"Bearer {TOKEN}"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["cache-control"], "no-store")
            await runtime.aclose()
            with self.assertRaises(httpx.ConnectError):
                await client.get(url)

    async def test_monitor_is_opt_in_and_rejects_weak_token(self):
        await self.runtime.start()
        self.assertIsNone(self.runtime.monitor)
        for token in ("", "short", "x" * 257, " " * 32, "x" * 32 + "\x00"):
            with self.assertRaises(ValueError):
                MonitorApp(self.runtime, 8766, token)
