"""Pure-function tests — HTML extraction, URL decoding, formatting. No network.

Run: python -m unittest discover tests
"""
import unittest

from sibyl.scraper import _extract_content, scrape_url
from sibyl.search import _extract_ddg_url
from sibyl.analyzer import CrossAnalysis, format_cross_analysis


class TestExtractContent(unittest.TestCase):
    def test_strips_noise_and_extracts_article(self):
        html = """
        <html><head><title>My Title</title></head>
        <body>
          <nav>menu home about links</nav>
          <script>var x = 1;</script>
          <article>
            <p>This is the real article body with meaningful content here.</p>
            <p>A second paragraph that also carries substantive information.</p>
          </article>
          <footer>copyright 2026 all rights reserved</footer>
        </body></html>
        """
        page = _extract_content(html, "https://ex.com/a", max_chars=6000)
        self.assertEqual(page.title, "My Title")
        self.assertIn("real article body", page.text)
        self.assertNotIn("var x", page.text)          # script dropped
        self.assertNotIn("copyright 2026", page.text)  # footer dropped

    def test_respects_max_chars(self):
        html = "<html><body><article>" + ("word longenough " * 500) + "</article></body></html>"
        page = _extract_content(html, "https://ex.com/b", max_chars=200)
        self.assertLessEqual(len(page.text), 200)

    def test_extracts_and_normalizes_article_publication_time(self):
        html = """
        <html><head>
          <meta property="article:published_time" content="2026-07-13T18:30:00Z">
          <script type="application/ld+json">
            {"datePublished": "2025-01-01"}
          </script>
        </head><body><article>Substantive article body text.</article></body></html>
        """

        page = _extract_content(html, "https://ex.com/published", 6000)

        self.assertEqual(page.published_at, "2026-07-13T18:30:00+00:00")
        self.assertEqual(
            page.published_at_method,
            "meta_article_published_time",
        )

    def test_extracts_json_ld_publication_date(self):
        html = """
        <html><head><script type="application/ld+json">
          {"@graph": [{"@type": "NewsArticle", "datePublished": "20260712"}]}
        </script></head><body><article>Substantive article body text.</article></body></html>
        """

        page = _extract_content(html, "https://ex.com/json-ld", 6000)

        self.assertEqual(page.published_at, "2026-07-12")
        self.assertEqual(page.published_at_method, "json_ld_date_published")

    def test_extracts_explicit_published_time_element(self):
        html = """
        <html><body><article>
          <time itemprop="datePublished" datetime="2026/07/11">July 11</time>
          Substantive article body text.
        </article></body></html>
        """

        page = _extract_content(html, "https://ex.com/time", 6000)

        self.assertEqual(page.published_at, "2026-07-11")
        self.assertEqual(page.published_at_method, "time_date_published")

    def test_ignores_modified_and_invalid_future_dates(self):
        html = """
        <html><head>
          <meta property="article:modified_time" content="2026-07-13T18:30:00Z">
          <meta property="article:published_time" content="2099-01-01T00:00:00Z">
        </head><body><article>Substantive article body text.</article></body></html>
        """

        page = _extract_content(html, "https://ex.com/invalid", 6000)

        self.assertIsNone(page.published_at)
        self.assertEqual(page.published_at_method, "")


class TestExtractorChoice(unittest.TestCase):
    def test_bs4_default_unchanged(self):
        html = "<html><head><title>T</title></head><body><article><p>the real body content here for the test</p></article></body></html>"
        page = _extract_content(html, "https://ex.com/a", 6000)  # default extractor
        self.assertIn("real body content", page.text)

    def test_trafilatura_length_compare_never_shrinks(self):
        # A page where BS4 would extract more than a thin trafilatura result:
        # the length-comparison fallback must keep the longer (bs4) text.
        body = " ".join(f"sentence number {i} with enough words to count" for i in range(40))
        html = f"<html><head><title>T</title></head><body><article><p>{body}</p></article></body></html>"
        traf = _extract_content(html, "https://ex.com/b", 6000, extractor="trafilatura")
        bs4 = _extract_content(html, "https://ex.com/b", 6000, extractor="bs4")
        self.assertGreaterEqual(len(traf.text), len(bs4.text) - 5)
        self.assertIn("sentence number 20", traf.text)


class TestScrapeUrlGuards(unittest.IsolatedAsyncioTestCase):
    async def test_non_http_url_returns_error_without_network(self):
        page = await scrape_url("ftp://example.com/file")
        self.assertIn("Only http and https", page.error)
        self.assertEqual(page.text, "")


class TestDdgUrl(unittest.TestCase):
    def test_decodes_uddg_param(self):
        raw = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage%3Fa%3D1&rut=x"
        self.assertEqual(_extract_ddg_url(raw), "https://example.com/page?a=1")

    def test_passthrough_when_no_uddg(self):
        self.assertEqual(_extract_ddg_url("https://plain.com/x"), "https://plain.com/x")


class TestFormatCrossAnalysis(unittest.TestCase):
    def test_renders_sections(self):
        ca = CrossAnalysis(
            consensus_points=["everyone agrees on A"],
            disagreement_points=["X vs Y"],
            unique_insights=["only source 3 notes Z"],
            overall_sentiment="positive",
            sentiment_breakdown={"positive": 3, "negative": 1, "neutral": 0},
        )
        out = format_cross_analysis(ca)
        self.assertIn("POSITIVE", out)
        self.assertIn("everyone agrees on A", out)
        self.assertIn("X vs Y", out)
        self.assertIn("only source 3 notes Z", out)


if __name__ == "__main__":
    unittest.main()
