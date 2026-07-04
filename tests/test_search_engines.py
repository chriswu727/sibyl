"""Search-engine parser tests — the most external-format-fragile code.

Each engine scrapes a live HTML/JSON format that can change silently and
degrade results without raising. These pin the parsing against canned responses
using an injected mock client (no network).

Run: python -m unittest discover tests
"""
import json
import unittest
from unittest import mock

from sibyl.search import (
    search_duckduckgo, search_reddit, search_wikipedia, search_google_news,
    search_mojeek, _search_general_web, SearchResult,
)


def _client_returning(text=None, payload=None, status=200):
    """A stand-in httpx.AsyncClient whose .get() returns a canned response."""
    resp = mock.Mock()
    resp.status_code = status
    resp.text = text or ""
    resp.json = mock.Mock(return_value=payload or {})
    client = mock.Mock()
    client.get = mock.AsyncMock(return_value=resp)
    return client


class TestDuckDuckGo(unittest.IsolatedAsyncioTestCase):
    async def test_parses_results(self):
        html = """
        <table>
          <tr><td><a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.com">Result A</a></td></tr>
          <tr><td class="result-snippet">Snippet for A</td></tr>
          <tr><td><a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fb.com">Result B</a></td></tr>
          <tr><td class="result-snippet">Snippet for B</td></tr>
        </table>
        """
        res = await search_duckduckgo("q", max_results=5, client=_client_returning(text=html))
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].url, "https://a.com")
        self.assertEqual(res[0].title, "Result A")
        self.assertEqual(res[0].snippet, "Snippet for A")
        self.assertEqual(res[0].source, "web")

    async def test_non_200_returns_empty(self):
        res = await search_duckduckgo("q", client=_client_returning(status=503))
        self.assertEqual(res, [])


class TestReddit(unittest.IsolatedAsyncioTestCase):
    async def test_parses_json(self):
        payload = {"data": {"children": [
            {"data": {"title": "Post one", "permalink": "/r/x/1", "selftext": "body", "subreddit": "x", "score": 42}},
        ]}}
        res = await search_reddit("q", max_results=5, client=_client_returning(payload=payload))
        self.assertEqual(len(res), 1)
        self.assertIn("r/x", res[0].title)
        self.assertIn("42 upvotes", res[0].title)
        self.assertEqual(res[0].url, "https://reddit.com/r/x/1")
        self.assertEqual(res[0].source, "reddit")

    async def test_bad_status_swallowed(self):
        res = await search_reddit("q", client=_client_returning(status=429))
        self.assertEqual(res, [])


class TestWikipedia(unittest.IsolatedAsyncioTestCase):
    async def test_parses_and_builds_url(self):
        payload = {"query": {"search": [
            {"title": "Machine learning", "snippet": "the <b>field</b> of study"},
        ]}}
        res = await search_wikipedia("q", client=_client_returning(payload=payload))
        self.assertEqual(len(res), 1)
        self.assertIn("Wikipedia", res[0].title)
        self.assertIn("Machine_learning", res[0].url)
        self.assertNotIn("<b>", res[0].snippet)  # HTML stripped from snippet


class TestMojeek(unittest.IsolatedAsyncioTestCase):
    async def test_parses_results(self):
        html = """
        <ul class="results-standard">
          <li><h2><a href="https://a.com/x">Title A</a></h2><p class="i">breadcrumb</p><p class="s">Snippet A here</p></li>
          <li><h2><a href="https://b.com/y">Title B</a></h2><p class="s">Snippet B here</p></li>
        </ul>
        """
        res = await search_mojeek("q", max_results=5, client=_client_returning(text=html))
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].title, "Title A")
        self.assertEqual(res[0].url, "https://a.com/x")
        self.assertEqual(res[0].snippet, "Snippet A here")
        self.assertEqual(res[0].source, "web")

    async def test_non_200_returns_empty(self):
        res = await search_mojeek("q", client=_client_returning(status=503))
        self.assertEqual(res, [])


class TestGeneralWebFailover(unittest.IsolatedAsyncioTestCase):
    async def test_ddg_results_skip_mojeek(self):
        ddg = mock.AsyncMock(return_value=[SearchResult("T", "https://x.com", "s", "web")])
        moj = mock.AsyncMock(return_value=[])
        with mock.patch("sibyl.search.search_duckduckgo", ddg), mock.patch("sibyl.search.search_mojeek", moj):
            res = await _search_general_web("q", 5)
        self.assertEqual(len(res), 1)
        moj.assert_not_called()  # DDG worked → no failover

    async def test_ddg_empty_fails_over_to_mojeek(self):
        ddg = mock.AsyncMock(return_value=[])
        moj = mock.AsyncMock(return_value=[SearchResult("M", "https://m.com", "s", "web")])
        with mock.patch("sibyl.search.search_duckduckgo", ddg), mock.patch("sibyl.search.search_mojeek", moj):
            res = await _search_general_web("q", 5)
        moj.assert_called_once()
        self.assertEqual(res[0].url, "https://m.com")


class TestGoogleNews(unittest.IsolatedAsyncioTestCase):
    async def test_parses_rss(self):
        rss = """<?xml version="1.0"?><rss><channel>
          <item><title>Headline one</title><link>https://n.com/1</link><description>desc one</description></item>
          <item><title>Headline two</title><link>https://n.com/2</link><description>desc two</description></item>
        </channel></rss>"""
        res = await search_google_news("q", max_results=5, client=_client_returning(text=rss))
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].title, "Headline one")
        self.assertEqual(res[0].url, "https://n.com/1")
        self.assertEqual(res[0].source, "news")


if __name__ == "__main__":
    unittest.main()
