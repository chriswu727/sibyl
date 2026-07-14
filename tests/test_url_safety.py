"""URL safety and redirect tests. No network."""
import unittest
from unittest import mock

import httpx

from sibyl.scraper import ResponseTooLargeError, _bounded_get, scrape_url
from sibyl.url_safety import unsafe_url_reason, validate_public_url


class _ChunkedStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"123"
        yield b"456"


def _response(status, text="", location=""):
    response = mock.Mock()
    response.status_code = status
    response.text = text
    response.headers = {
        "content-type": "text/html",
        **({"location": location} if location else {}),
    }
    return response


class TestUnsafeUrlReason(unittest.TestCase):
    def test_allows_public_web_urls(self):
        self.assertEqual(unsafe_url_reason("https://example.com/article?q=1"), "")
        self.assertEqual(unsafe_url_reason("http://8.8.8.8/"), "")

    def test_rejects_local_and_non_global_destinations(self):
        blocked = [
            "http://localhost/admin",
            "http://service.internal/admin",
            "http://127.0.0.1/admin",
            "http://2130706433/admin",
            "http://10.0.0.1/admin",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/admin",
            "http://224.0.0.1/",
        ]

        for url in blocked:
            with self.subTest(url=url):
                self.assertNotEqual(unsafe_url_reason(url), "")

    def test_rejects_credentials_and_non_web_ports(self):
        self.assertIn("credentials", unsafe_url_reason("https://user:pass@example.com/"))
        self.assertIn("ports 80 and 443", unsafe_url_reason("https://example.com:22/"))


class TestSafeRedirects(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        resolver_patcher = mock.patch(
            "sibyl.url_safety._resolve_hostname", return_value=["93.184.216.34"]
        )
        resolver_patcher.start()
        self.addCleanup(resolver_patcher.stop)
        fetch_patcher = mock.patch("sibyl.scraper._bounded_get", new=mock.AsyncMock())
        self.fetch = fetch_patcher.start()
        self.addCleanup(fetch_patcher.stop)

    async def test_unsafe_initial_url_never_reaches_client(self):
        client = mock.Mock()

        page = await scrape_url("http://127.0.0.1/admin", client=client)

        self.assertIn("Unsafe URL", page.error)
        self.fetch.assert_not_awaited()

    async def test_redirect_to_private_ip_is_blocked_before_second_request(self):
        client = mock.Mock()
        self.fetch.return_value = _response(
            302, location="http://169.254.169.254/metadata"
        )

        page = await scrape_url("https://example.com/start", client=client)

        self.assertIn("Unsafe redirect URL", page.error)
        self.assertEqual(self.fetch.await_count, 1)

    async def test_redirect_without_location_fails_without_looping(self):
        client = mock.Mock()
        self.fetch.return_value = _response(302)

        page = await scrape_url("https://example.com/start", client=client)

        self.assertIn("missing a Location", page.error)
        self.assertEqual(self.fetch.await_count, 1)

    async def test_safe_redirect_returns_final_url(self):
        html = "<html><head><title>Final</title></head><body><article>substantive final page text</article></body></html>"
        client = mock.Mock()
        self.fetch.side_effect = [
            _response(302, location="/final"),
            _response(200, text=html),
        ]

        page = await scrape_url("https://example.com/start", client=client)

        self.assertEqual(page.url, "https://example.com/final")
        self.assertEqual(page.title, "Final")
        self.assertEqual(self.fetch.await_count, 2)

    async def test_hostname_resolving_to_private_ip_is_blocked(self):
        client = mock.Mock()

        with mock.patch(
            "sibyl.url_safety._resolve_hostname", return_value=["127.0.0.1"]
        ):
            page = await scrape_url("https://example.com/start", client=client)

        self.assertIn("Private", page.error)
        self.fetch.assert_not_awaited()

    async def test_unresolvable_hostname_is_rejected(self):
        with mock.patch(
            "sibyl.url_safety._resolve_hostname", side_effect=OSError("dns failed")
        ):
            reason = await validate_public_url("https://missing.example.com/")

        self.assertIn("could not be resolved", reason)


class TestBoundedGet(unittest.IsolatedAsyncioTestCase):
    async def test_reads_a_response_under_the_limit(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"hello", request=request)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            response = await _bounded_get(
                client,
                "https://example.com/",
                headers={},
                timeout=1.0,
                max_response_bytes=5,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "hello")

    async def test_rejects_content_length_over_the_limit(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"too large", request=request)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with self.assertRaisesRegex(ResponseTooLargeError, "exceeds 4 bytes"):
                await _bounded_get(
                    client,
                    "https://example.com/",
                    headers={},
                    timeout=1.0,
                    max_response_bytes=4,
                )

    async def test_stops_streaming_when_body_crosses_the_limit(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                stream=_ChunkedStream(),
                request=request,
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with self.assertRaisesRegex(ResponseTooLargeError, "exceeds 5 bytes"):
                await _bounded_get(
                    client,
                    "https://example.com/",
                    headers={},
                    timeout=1.0,
                    max_response_bytes=5,
                )

    async def test_does_not_auto_follow_redirects(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(
                302,
                headers={"location": "https://example.com/final"},
                request=request,
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, follow_redirects=True
        ) as client:
            response = await _bounded_get(
                client,
                "https://example.com/start",
                headers={},
                timeout=1.0,
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
