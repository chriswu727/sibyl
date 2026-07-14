"""JS-render trigger on thin 200 pages. Mocked, no network.

Run: python -m unittest discover tests
"""
import asyncio
import unittest
from unittest import mock

from sibyl.scraper import _get_jina_gate, _JinaGate, scrape_url, WebPage


def _resp(status=200, text="", ct="text/html"):
    r = mock.Mock()
    r.status_code = status
    r.text = text
    r.headers = {"content-type": ct}
    return r


class TestJsRenderTrigger(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        resolver_patcher = mock.patch(
            "sibyl.url_safety._resolve_hostname", return_value=["93.184.216.34"]
        )
        resolver_patcher.start()
        self.addCleanup(resolver_patcher.stop)
        fetch_patcher = mock.patch("sibyl.scraper._bounded_get", new=mock.AsyncMock())
        self.fetch = fetch_patcher.start()
        self.addCleanup(fetch_patcher.stop)

    async def test_thin_page_triggers_render_and_keeps_longer(self):
        # a 200 that extracts to a thin shell → render via Jina, keep the longer text
        client = mock.Mock()
        self.fetch.return_value = _resp(
            200,
            """<html><head>
            <meta property="article:published_time" content="2026-07-10">
            </head><body><div>hi</div></body></html>""",
        )

        async def fake_jina(url, max_chars, c=None):
            return WebPage(url=url, title="Rendered", text="full rendered article content " * 30)

        with mock.patch("sibyl.scraper._try_jina", fake_jina):
            page = await scrape_url("https://spa.com/x", client=client, js_render=True, js_render_threshold=500)
        self.assertEqual(page.title, "Rendered")
        self.assertIn("rendered article", page.text)
        self.assertEqual(page.published_at, "2026-07-10")
        self.assertEqual(page.published_at_method, "meta_article_published_time")

    async def test_thick_page_does_not_render(self):
        big = "<html><body><article>" + ("real content sentence here. " * 100) + "</article></body></html>"
        client = mock.Mock()
        self.fetch.return_value = _resp(200, big)
        with mock.patch("sibyl.scraper._try_jina") as jina:
            page = await scrape_url("https://ok.com/x", client=client, js_render=True, js_render_threshold=500)
        jina.assert_not_called()
        self.assertIn("real content", page.text)

    async def test_disabled_by_default(self):
        client = mock.Mock()
        self.fetch.return_value = _resp(200, "<html><body>hi</body></html>")
        with mock.patch("sibyl.scraper._try_jina") as jina:
            await scrape_url("https://spa.com/x", client=client)
        jina.assert_not_called()


class TestJinaGate(unittest.IsolatedAsyncioTestCase):
    async def test_returns_one_shared_gate_per_event_loop(self):
        self.assertIs(_get_jina_gate(), _get_jina_gate())

    async def test_allows_at_most_two_in_flight_calls(self):
        gate = _JinaGate()
        release = asyncio.Event()
        two_started = asyncio.Event()
        active = 0
        max_active = 0

        async def fake_jina(url, max_chars, client):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                two_started.set()
            await release.wait()
            active -= 1

        with mock.patch.dict("os.environ", {"JINA_API_KEY": "test"}), mock.patch(
            "sibyl.scraper._try_jina", fake_jina
        ):
            tasks = [
                asyncio.create_task(gate.render(f"https://example.com/{i}", 100, object()))
                for i in range(4)
            ]
            await asyncio.wait_for(two_started.wait(), timeout=1.0)
            self.assertEqual(active, 2)
            release.set()
            await asyncio.gather(*tasks)

        self.assertEqual(max_active, 2)

    async def test_keyless_calls_wait_between_start_times(self):
        gate = _JinaGate()
        sleep = mock.AsyncMock()

        with mock.patch.dict("os.environ", {}, clear=True), mock.patch(
            "sibyl.scraper._try_jina", new=mock.AsyncMock(return_value=None)
        ), mock.patch("sibyl.scraper.time.monotonic", side_effect=[10.0, 11.0, 13.0]), mock.patch(
            "sibyl.scraper.asyncio.sleep", sleep
        ):
            await gate.render("https://example.com/1", 100, object())
            await gate.render("https://example.com/2", 100, object())

        sleep.assert_awaited_once_with(2.0)


if __name__ == "__main__":
    unittest.main()
