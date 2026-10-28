"""A refused page stays refused: one request, honest identity. Mocked, no network.

Run: python -m unittest discover tests
"""

import unittest
from unittest import mock

from sibyl.scraper import scrape_url


def _resp(status, text=""):
    r = mock.Mock()
    r.status_code = status
    r.text = text
    return r


class TestRefusals(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        resolver_patcher = mock.patch(
            "sibyl.url_safety._resolve_hostname", return_value=["93.184.216.34"]
        )
        resolver_patcher.start()
        self.addCleanup(resolver_patcher.stop)
        fetch_patcher = mock.patch("sibyl.scraper._bounded_get", new=mock.AsyncMock())
        self.fetch = fetch_patcher.start()
        self.addCleanup(fetch_patcher.stop)

    async def test_refusal_is_fetched_once_with_sibyl_identity(self):
        # Sibyl does not imitate browsers or retry a refusal under another
        # identity.
        self.fetch.return_value = _resp(403)
        page = await scrape_url("https://blocked.com/x", client=mock.Mock())
        self.fetch.assert_awaited_once()
        agent = self.fetch.await_args.kwargs["headers"]["User-Agent"]
        self.assertTrue(agent.startswith("Sibyl/"), agent)
        self.assertNotIn("Mozilla", agent)
        self.assertEqual(page.error, "HTTP 403")


if __name__ == "__main__":
    unittest.main()
