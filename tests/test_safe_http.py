"""Pinned DNS transport tests. No network."""
import ssl
import unittest
from unittest import mock

import httpcore

from sibyl.safe_http import PinnedDNSAsyncClient, PinnedDNSNetworkBackend
from sibyl.scraper import scrape_url


class _RecordingStream(httpcore.AsyncMockStream):
    def __init__(self, response, sni_hostnames, writes):
        super().__init__([response])
        self._sni_hostnames = sni_hostnames
        self._writes = writes

    async def write(self, buffer, timeout=None):
        self._writes.append(buffer)

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname=None,
        timeout=None,
    ):
        self._sni_hostnames.append(server_hostname)
        return self


class _ReadTimeoutStream(_RecordingStream):
    async def read(self, max_bytes, timeout=None):
        if self._buffer:
            return await super().read(max_bytes, timeout=timeout)
        raise httpcore.ReadTimeout("body stalled")


class _RecordingBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, responses):
        self.responses = responses
        self.connected_hosts = []
        self.sni_hostnames = []
        self.writes = []

    async def connect_tcp(
        self,
        host,
        port,
        timeout=None,
        local_address=None,
        socket_options=None,
    ):
        self.connected_hosts.append(host)
        response = self.responses[host]
        stream_type = (
            _ReadTimeoutStream if isinstance(response, tuple) else _RecordingStream
        )
        payload = response[0] if isinstance(response, tuple) else response
        return stream_type(payload, self.sni_hostnames, self.writes)

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise AssertionError("unexpected Unix socket")

    async def sleep(self, seconds):
        pass


def _response(status, body=b"", **headers):
    reason = {200: "OK", 302: "Found"}[status]
    fields = {
        "Content-Length": str(len(body)),
        "Connection": "close",
        **headers,
    }
    encoded_headers = "".join(f"{key}: {value}\r\n" for key, value in fields.items())
    return f"HTTP/1.1 {status} {reason}\r\n{encoded_headers}\r\n".encode() + body


class TestPinnedDNSNetworkBackend(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_unpinned_hostname(self):
        backend = PinnedDNSNetworkBackend(_RecordingBackend({}))

        with self.assertRaisesRegex(httpcore.ConnectError, "not been validated"):
            await backend.connect_tcp("example.com", 443)

    async def test_rejects_non_public_pin(self):
        backend = PinnedDNSNetworkBackend(_RecordingBackend({}))

        with self.assertRaisesRegex(ValueError, "non-public"):
            backend.pin("example.com", ["127.0.0.1"])


class TestPinnedScrape(unittest.IsolatedAsyncioTestCase):
    async def test_connects_to_validated_ip_with_original_host_and_sni(self):
        html = (
            b"<html><head><title>Pinned</title></head>"
            b"<body><article>validated public response body</article></body></html>"
        )
        backend = _RecordingBackend(
            {"93.184.216.34": _response(200, html, **{"Content-Type": "text/html"})}
        )
        resolver = mock.Mock(return_value=["93.184.216.34"])

        async with PinnedDNSAsyncClient(network_backend=backend) as client:
            with mock.patch("sibyl.url_safety._resolve_hostname", resolver):
                page = await scrape_url("https://example.com/article", client=client)

        self.assertEqual(page.title, "Pinned")
        self.assertEqual(backend.connected_hosts, ["93.184.216.34"])
        self.assertEqual(backend.sni_hostnames, ["example.com"])
        self.assertIn(b"Host: example.com", b"".join(backend.writes))
        resolver.assert_called_once_with("example.com", 443)

    async def test_pins_each_cross_domain_redirect(self):
        html = (
            b"<html><head><title>Final</title></head>"
            b"<body><article>redirected public response body</article></body></html>"
        )
        backend = _RecordingBackend(
            {
                "93.184.216.34": _response(
                    302,
                    Location="https://final.example/article",
                ),
                "1.1.1.1": _response(200, html, **{"Content-Type": "text/html"}),
            }
        )

        def resolve(hostname, port):
            return {
                "start.example": ["93.184.216.34"],
                "final.example": ["1.1.1.1"],
            }[hostname]

        async with PinnedDNSAsyncClient(network_backend=backend) as client:
            with mock.patch(
                "sibyl.url_safety._resolve_hostname",
                side_effect=resolve,
            ) as resolver:
                page = await scrape_url("https://start.example/", client=client)

        self.assertEqual(page.url, "https://final.example/article")
        self.assertEqual(backend.connected_hosts, ["93.184.216.34", "1.1.1.1"])
        self.assertEqual(
            backend.sni_hostnames,
            ["start.example", "final.example"],
        )
        self.assertEqual(resolver.call_count, 2)

    async def test_maps_stream_timeout_to_scraper_timeout(self):
        headers_only = _response(
            200,
            **{"Content-Type": "text/html", "Content-Length": "10"},
        )
        backend = _RecordingBackend({"93.184.216.34": (headers_only,)})

        async with PinnedDNSAsyncClient(network_backend=backend) as client:
            with mock.patch(
                "sibyl.url_safety._resolve_hostname",
                return_value=["93.184.216.34"],
            ):
                page = await scrape_url("https://example.com/slow", client=client)

        self.assertEqual(page.error, "timeout")


if __name__ == "__main__":
    unittest.main()
