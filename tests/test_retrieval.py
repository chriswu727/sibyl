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
        self.assertEqual(first.schema_version, "1.5")
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
        self.assertIsNotNone(first.sources[0].relevance_score)
        self.assertIsNotNone(first.sources[0].evidence[0].score)
        self.assertIsNone(first.sources[0].quality_score)
        self.assertEqual(first.diagnostics.search_results, 3)
        self.assertEqual(first.diagnostics.urls_attempted, 3)
        self.assertEqual(first.diagnostics.sources_returned, 2)
        self.assertEqual(first.diagnostics.effective_max_sources, 2)
        self.assertEqual(first.diagnostics.effective_chars_per_source, 7000)
        self.assertEqual(first.diagnostics.ranking_method, "lexical_v1")
        self.assertEqual(first.diagnostics.requested_ranking_method, "lexical")
        self.assertEqual(first.diagnostics.ranking_warning, "")
        self.assertEqual(first.diagnostics.candidates_ranked, 3)
        self.assertEqual(first.diagnostics.passages_returned, 2)
        self.assertEqual(first.diagnostics.coverage_method, "lexical_query_terms_v1")
        self.assertGreater(first.diagnostics.query_term_coverage, 0)
        self.assertEqual(first.diagnostics.unique_domains, 1)
        self.assertEqual(first.diagnostics.substantive_sources, 2)
        self.assertGreater(first.diagnostics.evidence_chars, 0)
        self.assertEqual(first.diagnostics.evidence_sufficiency, "limited")
        self.assertEqual(first.diagnostics.sufficiency_reasons, ["single_domain"])
        wiki.assert_not_awaited()

    async def test_marks_diverse_relevant_evidence_sufficient(self):
        results = [
            SearchResult("Alpha evidence", "https://a.example/report", "", "web"),
            SearchResult("Alpha findings", "https://b.example/study", "", "academic"),
        ]
        pages = [
            WebPage(result.url, result.title, "alpha evidence findings " * 30)
            for result in results
        ]

        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=results)), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=pages)), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])):
            bundle = await gather_source_bundle(
                "alpha evidence", max_sources=2, client=object()
            )

        self.assertEqual(bundle.status, "ok")
        self.assertEqual(bundle.diagnostics.evidence_sufficiency, "sufficient")
        self.assertEqual(bundle.diagnostics.sufficiency_reasons, [])
        self.assertEqual(bundle.diagnostics.unique_domains, 2)
        self.assertEqual(bundle.diagnostics.substantive_sources, 2)

    async def test_marks_irrelevant_full_text_as_insufficient(self):
        results = [
            SearchResult("Cooking", "https://a.example/recipe", "", "web"),
            SearchResult("Travel", "https://b.example/hotels", "", "web"),
        ]
        pages = [
            WebPage(results[0].url, results[0].title, "bread recipe kitchen " * 30),
            WebPage(results[1].url, results[1].title, "hotel travel guide " * 30),
        ]

        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=results)), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=pages)), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])):
            bundle = await gather_source_bundle(
                "quantum battery breakthrough", max_sources=2, client=object()
            )

        self.assertEqual(bundle.status, "insufficient_evidence")
        self.assertEqual(bundle.diagnostics.evidence_sufficiency, "insufficient")
        self.assertEqual(
            bundle.diagnostics.sufficiency_reasons,
            ["low_query_term_coverage"],
        )
        self.assertEqual(len(bundle.sources), 2)
        self.assertIn("low query-term coverage", bundle.error)
        self.assertIn("Evidence warning", render_source_bundle(bundle))

    async def test_marks_thin_only_sources_as_insufficient(self):
        result = SearchResult("Alpha evidence", "https://a.example/short", "", "web")
        page = WebPage(
            result.url,
            result.title,
            ("alpha evidence " * 20)[:180],
        )

        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=[result])), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=[page])), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])):
            bundle = await gather_source_bundle("alpha evidence", client=object())

        self.assertEqual(bundle.status, "insufficient_evidence")
        self.assertEqual(bundle.diagnostics.substantive_sources, 0)
        self.assertIn(
            "no_substantive_sources",
            bundle.diagnostics.sufficiency_reasons,
        )

    async def test_reranks_candidates_before_applying_source_limit(self):
        results = [
            SearchResult("Cooking", "https://example.com/cooking", "", "web"),
            SearchResult("Travel", "https://example.com/travel", "", "web"),
            SearchResult("Madrid Open", "https://example.com/tennis", "", "web"),
        ]
        pages = [
            WebPage(results[0].url, results[0].title, "bread recipe " * 30),
            WebPage(results[1].url, results[1].title, "hotel guide " * 30),
            WebPage(
                results[2].url,
                results[2].title,
                "Dušan Lajović was the Serbian quarterfinalist at the 2018 Madrid Open. " * 8,
            ),
        ]

        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=results)), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=pages)), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])):
            bundle = await gather_source_bundle(
                "Serbian quarterfinalist 2018 Madrid Open",
                max_sources=1,
                client=object(),
            )

        self.assertEqual(bundle.sources[0].url, "https://example.com/tennis")
        self.assertGreater(bundle.sources[0].relevance_score, 0.5)
        self.assertEqual(bundle.diagnostics.candidates_ranked, 3)

    async def test_flashrank_backend_records_actual_method(self):
        results = [
            SearchResult("Noise", "https://example.com/noise", "", "web"),
            SearchResult("Target", "https://example.com/target", "", "web"),
            SearchResult("Other", "https://example.com/other", "", "web"),
        ]
        pages = [
            WebPage(result.url, result.title, f"{result.title} body " * 30)
            for result in results
        ]

        def fake_scores(query, documents):
            return [0.95 if title == "Target" else 0.05 for title, _ in documents]

        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=results)), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=pages)), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])), \
             mock.patch("sibyl.retrieval.flashrank_relevance_scores", side_effect=fake_scores):
            bundle = await gather_source_bundle(
                "target", max_sources=1, client=object(), ranker="flashrank"
            )

        self.assertEqual(bundle.sources[0].url, "https://example.com/target")
        self.assertEqual(bundle.diagnostics.requested_ranking_method, "flashrank")
        self.assertEqual(bundle.diagnostics.ranking_method, "flashrank")
        self.assertEqual(bundle.diagnostics.ranking_warning, "")

    async def test_flashrank_failure_falls_back_with_diagnostic(self):
        results = [
            SearchResult("Noise", "https://example.com/noise", "", "web"),
            SearchResult("Alpha beta", "https://example.com/match", "", "web"),
            SearchResult("Other", "https://example.com/other", "", "web"),
        ]
        pages = [
            WebPage(result.url, result.title, f"{result.title} body " * 30)
            for result in results
        ]

        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=results)), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=pages)), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])), \
             mock.patch(
                 "sibyl.retrieval.flashrank_relevance_scores",
                 side_effect=ImportError("optional dependency missing"),
             ):
            bundle = await gather_source_bundle(
                "alpha beta", max_sources=1, client=object(), ranker="flashrank"
            )

        self.assertEqual(bundle.sources[0].url, "https://example.com/match")
        self.assertEqual(bundle.diagnostics.ranking_method, "lexical_v1")
        self.assertIn("ImportError", bundle.diagnostics.ranking_warning)
        self.assertIn("fell back", bundle.diagnostics.ranking_warning)

    async def test_none_ranker_preserves_order_and_null_scores(self):
        results = [
            SearchResult("First", "https://example.com/first", "", "web"),
            SearchResult("Target", "https://example.com/target", "", "web"),
            SearchResult("Third", "https://example.com/third", "", "web"),
        ]
        pages = [
            WebPage(result.url, result.title, f"{result.title} body " * 30)
            for result in results
        ]

        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=results)), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=pages)), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])):
            bundle = await gather_source_bundle(
                "target", max_sources=2, client=object(), ranker="none"
            )

        self.assertEqual(
            [source.url for source in bundle.sources],
            ["https://example.com/first", "https://example.com/target"],
        )
        self.assertTrue(all(source.relevance_score is None for source in bundle.sources))
        self.assertTrue(
            all(
                passage.score is None
                for source in bundle.sources
                for passage in source.evidence
            )
        )
        self.assertEqual(bundle.diagnostics.ranking_method, "none")
        self.assertEqual(bundle.diagnostics.candidates_ranked, 0)
        self.assertEqual(bundle.diagnostics.chunks_ranked, 0)

    async def test_returns_ranked_passages_with_offsets_within_source_budget(self):
        results = [
            SearchResult("Alpha", "https://example.com/alpha", "", "web"),
            SearchResult("Beta", "https://example.com/beta", "", "web"),
            SearchResult("Gamma", "https://example.com/gamma", "", "web"),
        ]
        target_text = (
            "background material " * 80
            + "critical alpha evidence first section. " * 20
            + "unrelated bridge " * 80
            + "alpha evidence second section with details. " * 20
        )
        pages = [
            WebPage(results[0].url, results[0].title, target_text),
            WebPage(results[1].url, results[1].title, "beta material " * 200),
            WebPage(results[2].url, results[2].title, "gamma material " * 200),
        ]

        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=results)), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=pages)), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])):
            bundle = await gather_source_bundle(
                "alpha evidence", max_sources=1, chars_per_source=3000, client=object()
            )

        source = bundle.sources[0]
        self.assertGreaterEqual(len(source.evidence), 2)
        self.assertLessEqual(sum(len(p.text) for p in source.evidence), 3000)
        self.assertEqual(
            [passage.passage_id for passage in source.evidence],
            [f"P{index}" for index in range(1, len(source.evidence) + 1)],
        )
        self.assertEqual(
            [passage.score for passage in source.evidence],
            sorted((passage.score for passage in source.evidence), reverse=True),
        )
        for passage in source.evidence:
            self.assertEqual(
                target_text[passage.start_char:passage.end_char], passage.text
            )
            self.assertIn(f"/{source.source_id}/{passage.passage_id}", passage.citation_id)
        self.assertEqual(bundle.diagnostics.passages_returned, len(source.evidence))
        self.assertEqual(bundle.diagnostics.chunks_ranked, len(source.evidence))

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

    async def test_invalid_ranker_is_rejected_before_search(self):
        search = mock.AsyncMock()
        with mock.patch("sibyl.retrieval.search_web", new=search):
            bundle = await gather_source_bundle(
                "query", client=object(), ranker="remote"
            )

        self.assertEqual(bundle.status, "invalid_request")
        self.assertIn("lexical, flashrank, none", bundle.error)
        self.assertEqual(bundle.diagnostics.requested_ranking_method, "remote")
        search.assert_not_awaited()

    async def test_no_sources_is_explicit_insufficient_evidence(self):
        with mock.patch("sibyl.retrieval.search_web", new=mock.AsyncMock(return_value=[])), \
             mock.patch("sibyl.retrieval.scrape_urls", new=mock.AsyncMock(return_value=[])), \
             mock.patch("sibyl.retrieval.wikipedia_lookup", new=mock.AsyncMock(return_value=[])):
            bundle = await gather_source_bundle("missing", client=object())

        self.assertEqual(bundle.status, "insufficient_evidence")
        self.assertEqual(bundle.sources, [])
        self.assertEqual(bundle.diagnostics.sources_returned, 0)
        self.assertEqual(bundle.diagnostics.ranking_method, "not_run")
        self.assertIsNone(bundle.diagnostics.query_term_coverage)
        self.assertEqual(bundle.diagnostics.evidence_sufficiency, "insufficient")
        self.assertEqual(bundle.diagnostics.sufficiency_reasons, ["no_sources"])
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
