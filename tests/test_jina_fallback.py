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
    async def test_not_triggered_by_default(self):
        # 403 twice, jina_fallback off (default) → error, Jina never called.
        client = mock.Mock()
        client.get = mock.AsyncMock(return_value=_resp(403))
        with mock.patch("sibyl.scraper._try_jina") as jina:
            page = await scrape_url("https://blocked.com/x", client=client)
        jina.assert_not_called()
        self.assertTrue(page.error and "403" in page.error)

    async def test_triggered_on_block_when_enabled(self):
        client = mock.Mock()
        client.get = mock.AsyncMock(return_value=_resp(403))
        from sibyl.scraper import WebPage

        async def fake_jina(url, max_chars, c=None):
            return WebPage(url=url, title="Recovered", text="clean markdown content " * 20)

        with mock.patch("sibyl.scraper._try_jina", fake_jina):
            page = await scrape_url("https://blocked.com/x", client=client, jina_fallback=True)
        self.assertEqual(page.title, "Recovered")
        self.assertIn("clean markdown", page.text)

    async def test_jina_parses_markdown_and_uses_key(self):
        captured = {}

        async def fake_get(url, headers=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            body = "Title: My Page\n\n" + ("This is the clean body content returned by Jina reader for the page. " * 4)
            return _resp(200, body)

        client = mock.Mock()
        client.get = fake_get
        with mock.patch.dict(os.environ, {"JINA_API_KEY": "jina-key"}, clear=False):
            page = await _try_jina("https://ex.com/a", 6000, client)
        self.assertIsNotNone(page)
        self.assertEqual(page.title, "My Page")
        self.assertIn("clean body content", page.text)
        self.assertTrue(captured["url"].startswith("https://r.jina.ai/https://ex.com/a"))
        self.assertEqual(captured["headers"]["X-Return-Format"], "markdown")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer jina-key")


if __name__ == "__main__":
    unittest.main()
