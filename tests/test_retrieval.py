"""Structured keyless retrieval tests. No network."""
import hashlib
import unittest
from unittest import mock

from sibyl.retrieval import gather_source_bundle, render_source_bundle
from sibyl.scraper import WebPage
from sibyl.search import SearchResult


class TestGatherSourceBundle(unittest.IsolatedAsyncioTestCase):
    async def test_builds_stable_structured_evidence(self):
        results = [
            SearchResult("Alpha", "https://example.com/a", "", "web"),
            SearchResult("News", "https://example.com/b", "", "news"),
            SearchResult("Paper", "https://example.com/c", "", "academic"),
        ]
        pages = [
            WebPage(result.url, result.title, f"alpha {result.title} " + "x" * 600)
            for result in results
        ]
        pages[1].title = ""
        client = object()

        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=results)), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=pages)), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])) as wiki:
            first = await gather_source_bundle(" alpha ", max_sources=2, client=client)
            second = await gather_source_bundle("alpha", max_sources=2, client=client)

        self.assertEqual(first.status, "ok")
        self.assertEqual(first.bundle_id, second.bundle_id)
        self.assertRegex(first.bundle_id, r"^sb_[0-9a-f]{16}$")
        self.assertEqual([source.source_id for source in first.sources], ["S1", "S2"])
        self.assertEqual(first.sources[1].source_type, "news")
        self.assertEqual(first.sources[1].title, "News")
        self.assertEqual(first.sources[0].evidence[0].passage_id, "P1")
        self.assertEqual(
            first.sources[0].evidence[0].citation_id,
            f"{first.bundle_id}/S1/P1",
        )
        self.assertEqual(
            first.sources[0].content_hash,
            hashlib.sha256(pages[0].text.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(len(first.sources[0].content_hash), 64)
        self.assertIsNone(first.sources[0].relevance_score)
        self.assertIsNone(first.sources[0].quality_score)
        self.assertIsNone(first.sources[0].evidence[0].score)
        self.assertEqual(first.diagnostics.search_results, 3)
        self.assertEqual(first.diagnostics.urls_attempted, 3)
        self.assertEqual(first.diagnostics.sources_returned, 2)
        self.assertEqual(first.diagnostics.effective_max_sources, 2)
        self.assertEqual(first.diagnostics.effective_chars_per_source, 7000)
        wiki.assert_not_awaited()

    async def test_bounds_request_parameters(self):
        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=[])), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=[])), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])):
            bundle = await gather_source_bundle(
                "query", max_sources=50, chars_per_source=100, client=object()
            )

        self.assertEqual(bundle.diagnostics.requested_max_sources, 50)
        self.assertEqual(bundle.diagnostics.effective_max_sources, 20)
        self.assertEqual(bundle.diagnostics.requested_chars_per_source, 100)
        self.assertEqual(bundle.diagnostics.effective_chars_per_source, 500)

    async def test_empty_query_is_rejected_before_search(self):
        search = mock.AsyncMock()
        with mock.patch("sibyl.retrieval.search_web", new=search):
            bundle = await gather_source_bundle("   ", client=object())

        self.assertEqual(bundle.status, "invalid_request")
        self.assertIn("must not be empty", bundle.error)
        search.assert_not_awaited()

    async def test_no_sources_is_explicit_insufficient_evidence(self):
        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=[])), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=[])), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])):
            bundle = await gather_source_bundle("missing", client=object())

        self.assertEqual(bundle.status, "insufficient_evidence")
        self.assertEqual(bundle.sources, [])
        self.assertEqual(bundle.diagnostics.sources_returned, 0)
        self.assertEqual(
            render_source_bundle(bundle),
            "No sources found for query: 'missing'. Try a different phrasing.",
        )

        invalid = await gather_source_bundle("", client=object())
        self.assertNotEqual(bundle.bundle_id, invalid.bundle_id)

    async def test_retrieval_failure_is_returned_as_failure(self):
        with mock.patch(
            "sibyl.retrieval.search_web",
            new=mock.AsyncMock(side_effect=RuntimeError("engine unavailable")),
        ):
            bundle = await gather_source_bundle("query", client=object())

        self.assertEqual(bundle.status, "failed")
        self.assertIn("engine unavailable", bundle.error)
        self.assertEqual(render_source_bundle(bundle), bundle.error)

        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=[])), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=[])), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])):
            insufficient = await gather_source_bundle("query", client=object())
        self.assertNotEqual(bundle.bundle_id, insufficient.bundle_id)


class TestRenderSourceBundle(unittest.TestCase):
    def test_legacy_renderer_keeps_source_blocks(self):
        diagnostics = mock.Mock()
        passage = mock.Mock(text="Evidence text")
        source = mock.Mock(title="Title", url="https://example.com", evidence=[passage])
        bundle = mock.Mock(status="ok", sources=[source], query="question", diagnostics=diagnostics)

        text = render_source_bundle(bundle)

        self.assertIn("Retrieved 1 sources", text)
        self.assertIn("[Source 1: Title]", text)
        self.assertIn("URL: https://example.com", text)
        self.assertIn("Evidence text", text)


if __name__ == "__main__":
    unittest.main()
