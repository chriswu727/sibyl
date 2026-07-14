"""Jina Reader fallback — opt-in, triggered on a hard block. Mocked, no network.

Run: python -m unittest discover tests
"""
import os
import unittest
from unittest import mock

from sibyl.scraper import scrape_url, _try_jina


def _resp(status, text=""):
    r = mock.Mock()
    r.status_code = status
    r.text = text
    return r


class TestJinaFallback(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        resolver_patcher = mock.patch(
            "sibyl.url_safety._resolve_hostname", return_value=["93.184.216.34"]
        )
        resolver_patcher.start()
        self.addCleanup(resolver_patcher.stop)
        fetch_patcher = mock.patch("sibyl.scraper._bounded_get", new=mock.AsyncMock())
        self.fetch = fetch_patcher.start()
        self.addCleanup(fetch_patcher.stop)

    async def test_not_triggered_by_default(self):
        # 403 twice, jina_fallback off (default) → error, Jina never called.
        client = mock.Mock()
        self.fetch.return_value = _resp(403)
        with mock.patch("sibyl.scraper._try_jina") as jina:
            page = await scrape_url("https://blocked.com/x", client=client)
        jina.assert_not_called()
        self.assertTrue(page.error and "403" in page.error)

    async def test_triggered_on_451_first_attempt(self):
        # 451 doesn't retry (no UA-swap continue) — the fallback must still fire.
        client = mock.Mock()
        self.fetch.return_value = _resp(451)
        from sibyl.scraper import WebPage

        gate = mock.Mock()
        gate.render = mock.AsyncMock(
            return_value=WebPage(
                url="https://legal-block.com/x",
                title="Recovered",
                text="clean content " * 20,
            )
        )

        with mock.patch("sibyl.scraper._get_jina_gate", return_value=gate):
            page = await scrape_url("https://legal-block.com/x", client=client, jina_fallback=True)
        self.assertEqual(page.title, "Recovered")
        gate.render.assert_awaited_once_with(
            "https://legal-block.com/x", 6000, client
        )

    async def test_triggered_on_block_when_enabled(self):
        client = mock.Mock()
        self.fetch.return_value = _resp(403)
        from sibyl.scraper import WebPage

        async def fake_jina(url, max_chars, c=None):
            return WebPage(url=url, title="Recovered", text="clean markdown content " * 20)

        with mock.patch("sibyl.scraper._try_jina", fake_jina):
            page = await scrape_url("https://blocked.com/x", client=client, jina_fallback=True)
        self.assertEqual(page.title, "Recovered")
        self.assertIn("clean markdown", page.text)

    async def test_jina_parses_markdown_and_uses_key(self):
        captured = {}

        async def fake_get(client, url, headers=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            body = (
                "Title: My Page\nPublished Time: 2026-07-12T08:45:00Z\n\n"
                + (
                    "This is the clean body content returned by Jina reader "
                    "for the page. " * 4
                )
            )
            return _resp(200, body)

        client = mock.Mock()
        self.fetch.side_effect = fake_get
        with mock.patch.dict(os.environ, {"JINA_API_KEY": "jina-key"}, clear=False):
            page = await _try_jina("https://ex.com/a", 6000, client)
        self.assertIsNotNone(page)
        self.assertEqual(page.title, "My Page")
        self.assertIn("clean body content", page.text)
        self.assertEqual(page.content_origin, "jina_reader")
        self.assertEqual(page.published_at, "2026-07-12T08:45:00+00:00")
        self.assertEqual(page.published_at_method, "jina_published_time")
        self.assertTrue(captured["url"].startswith("https://r.jina.ai/https://ex.com/a"))
        self.assertEqual(captured["headers"]["X-Return-Format"], "markdown")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer jina-key")


if __name__ == "__main__":
    unittest.main()
