"""Opt-in, loopback-only, read-only ASGI monitoring surface."""

from __future__ import annotations

import asyncio
import hmac
import json
import socket
from contextlib import contextmanager


class MonitorApp:
    def __init__(self, runtime, port: int, token: str):
        if not 32 <= len(token) <= 256 or any(not 33 <= ord(c) <= 126 for c in token):
            raise ValueError(
                "SIBYL_MONITOR_TOKEN must be 32-256 non-whitespace ASCII characters."
            )
        self.runtime = runtime
        self.host = f"127.0.0.1:{port}".encode()
        self.authorization = b"Bearer " + token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        headers = scope.get("headers", [])
        values = dict(headers)
        status, body, content_type = 200, b"", b"application/json"
        if (
            len(values) != len(headers)
            or values.get(b"host") != self.host
            or b"origin" in values
            or b"transfer-encoding" in values
            or values.get(b"content-length", b"0") != b"0"
            or scope.get("query_string")
        ):
            status, body = 400, b'{"error":"invalid request"}'
        elif scope["method"] != "GET":
            status, body = 405, b'{"error":"read-only endpoint"}'
        elif not hmac.compare_digest(
            values.get(b"authorization", b""), self.authorization
        ):
            status, body = 401, b'{"error":"authentication required"}'
        elif scope["path"] == "/healthz":
            status = 503 if self.runtime.closed else 200
            body = json.dumps(
                {"status": "stopped" if self.runtime.closed else "alive"}
            ).encode()
        elif scope["path"] not in {"/readyz", "/status", "/metrics"}:
            status, body = 404, b'{"error":"not found"}'
        else:
            snapshot = self.runtime.snapshot()
            path = scope["path"]
            if path == "/readyz":
                ready = (
                    not snapshot["closed"]
                    and snapshot["jobs"]["healthy"]
                    and snapshot["active_calls"] < snapshot["capacity"]
                )
                status, body = (
                    (200 if ready else 503),
                    json.dumps({"ready": ready}).encode(),
                )
            elif path == "/status":
                status = 200 if snapshot["jobs"]["healthy"] else 503
                body = json.dumps(snapshot, allow_nan=False).encode()
            elif path == "/metrics":
                content_type = b"text/plain; version=0.0.4; charset=utf-8"
                body = self.runtime.metrics.prometheus(snapshot).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", content_type),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"content-security-policy", b"default-src 'none'"),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class MonitorServer:
    def __init__(self, runtime, port: int, token: str):
        if not 0 <= port <= 65535:
            raise ValueError("Monitor port must be between 0 and 65535.")
        self.runtime, self.port, self.token = runtime, port, token
        self.server = self.task = self.socket = None
        MonitorApp(runtime, port, token)

    async def start(self):
        import uvicorn

        class Server(uvicorn.Server):
            @contextmanager
            def capture_signals(self):
                # MCP owns process signals; the auxiliary listener must not replace them.
                yield

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.bind(("127.0.0.1", self.port))
            self.socket.listen(16)
            self.socket.setblocking(False)
            self.port = self.socket.getsockname()[1]
            config = uvicorn.Config(
                MonitorApp(self.runtime, self.port, self.token),
                host="127.0.0.1",
                port=self.port,
                access_log=False,
                log_config=None,
                log_level="critical",
                lifespan="off",
                ws="none",
                proxy_headers=False,
                http="h11",
                limit_concurrency=16,
                timeout_keep_alive=2,
                timeout_graceful_shutdown=2,
                h11_max_incomplete_event_size=8192,
            )
            self.server = Server(config)
            self.task = asyncio.create_task(self.server.serve(sockets=[self.socket]))
            async with asyncio.timeout(5):
                while not self.server.started:
                    if self.task.done():
                        await self.task
                        raise RuntimeError("Monitoring listener failed to start.")
                    await asyncio.sleep(0.01)
        except BaseException:
            await self.close()
            raise

    async def close(self):
        if self.server:
            self.server.should_exit = True
        if self.task:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), 3)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        if self.socket:
            self.socket.close()
