"""URL safety and redirect tests. No network."""
import unittest
from unittest import mock

from sibyl.scraper import scrape_url
from sibyl.url_safety import unsafe_url_reason, validate_public_url


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
        patcher = mock.patch(
            "sibyl.url_safety._resolve_hostname", return_value=["93.184.216.34"]
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_unsafe_initial_url_never_reaches_client(self):
        client = mock.Mock()
        client.get = mock.AsyncMock()

        page = await scrape_url("http://127.0.0.1/admin", client=client)

        self.assertIn("Unsafe URL", page.error)
        client.get.assert_not_awaited()

    async def test_redirect_to_private_ip_is_blocked_before_second_request(self):
        client = mock.Mock()
        client.get = mock.AsyncMock(
            return_value=_response(302, location="http://169.254.169.254/metadata")
        )

        page = await scrape_url("https://example.com/start", client=client)

        self.assertIn("Unsafe redirect URL", page.error)
        self.assertEqual(client.get.await_count, 1)
        self.assertFalse(client.get.await_args.kwargs["follow_redirects"])

    async def test_redirect_without_location_fails_without_looping(self):
        client = mock.Mock()
        client.get = mock.AsyncMock(return_value=_response(302))

        page = await scrape_url("https://example.com/start", client=client)

        self.assertIn("missing a Location", page.error)
        self.assertEqual(client.get.await_count, 1)

    async def test_safe_redirect_returns_final_url(self):
        html = "<html><head><title>Final</title></head><body><article>substantive final page text</article></body></html>"
        client = mock.Mock()
        client.get = mock.AsyncMock(
            side_effect=[
                _response(302, location="/final"),
                _response(200, text=html),
            ]
        )

        page = await scrape_url("https://example.com/start", client=client)

        self.assertEqual(page.url, "https://example.com/final")
        self.assertEqual(page.title, "Final")
        self.assertEqual(client.get.await_count, 2)

    async def test_hostname_resolving_to_private_ip_is_blocked(self):
        client = mock.Mock()
        client.get = mock.AsyncMock()

        with mock.patch(
            "sibyl.url_safety._resolve_hostname", return_value=["127.0.0.1"]
        ):
            page = await scrape_url("https://example.com/start", client=client)

        self.assertIn("Private", page.error)
        client.get.assert_not_awaited()

    async def test_unresolvable_hostname_is_rejected(self):
        with mock.patch(
            "sibyl.url_safety._resolve_hostname", side_effect=OSError("dns failed")
        ):
            reason = await validate_public_url("https://missing.example.com/")

        self.assertIn("could not be resolved", reason)


if __name__ == "__main__":
    unittest.main()
