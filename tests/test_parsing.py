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


class TestScrapeUrlGuards(unittest.IsolatedAsyncioTestCase):
    async def test_non_http_url_returns_error_without_network(self):
        page = await scrape_url("ftp://example.com/file")
        self.assertEqual(page.error, "Invalid URL")
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
