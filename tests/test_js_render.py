"""JS-render trigger on thin 200 pages. Mocked, no network.

Run: python -m unittest discover tests
"""
import unittest
from unittest import mock

from sibyl.scraper import scrape_url, WebPage


def _resp(status=200, text="", ct="text/html"):
    r = mock.Mock()
    r.status_code = status
    r.text = text
    r.headers = {"content-type": ct}
    return r


class TestJsRenderTrigger(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch(
            "sibyl.url_safety._resolve_hostname", return_value=["93.184.216.34"]
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_thin_page_triggers_render_and_keeps_longer(self):
        # a 200 that extracts to a thin shell → render via Jina, keep the longer text
        client = mock.Mock()
        client.get = mock.AsyncMock(return_value=_resp(200, "<html><body><div>hi</div></body></html>"))

        async def fake_jina(url, max_chars, c=None):
            return WebPage(url=url, title="Rendered", text="full rendered article content " * 30)

        with mock.patch("sibyl.scraper._try_jina", fake_jina):
            page = await scrape_url("https://spa.com/x", client=client, js_render=True, js_render_threshold=500)
        self.assertEqual(page.title, "Rendered")
        self.assertIn("rendered article", page.text)

    async def test_thick_page_does_not_render(self):
        big = "<html><body><article>" + ("real content sentence here. " * 100) + "</article></body></html>"
        client = mock.Mock()
        client.get = mock.AsyncMock(return_value=_resp(200, big))
        with mock.patch("sibyl.scraper._try_jina") as jina:
            page = await scrape_url("https://ok.com/x", client=client, js_render=True, js_render_threshold=500)
        jina.assert_not_called()
        self.assertIn("real content", page.text)

    async def test_disabled_by_default(self):
        client = mock.Mock()
        client.get = mock.AsyncMock(return_value=_resp(200, "<html><body>hi</body></html>"))
        with mock.patch("sibyl.scraper._try_jina") as jina:
            await scrape_url("https://spa.com/x", client=client)  # js_render defaults False at this layer
        jina.assert_not_called()


if __name__ == "__main__":
    unittest.main()
