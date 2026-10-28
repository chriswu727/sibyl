"""Pure-function tests — HTML extraction. No network.

Run: python -m unittest discover tests
"""

import unittest

from sibyl.scraper import _extract_content, scrape_url


class TestExtractContent(unittest.TestCase):
    def test_whitespace_cleanup_does_not_expose_html_comments(self):
        page = _extract_content(
            "<main><p>Visible evidence.</p><!-- hidden   instructions --></main>",
            "https://example.com",
            6000,
        )
        self.assertIn("Visible evidence.", page.text)
        self.assertNotIn("hidden", page.text)

    def test_placeholder_titles_fall_back_to_heading(self):
        page = _extract_content(
            '<title>references-details-empty</title><meta property="og:title" '
            'content="references-details-empty"><main><h1>Retry safety</h1>'
            "<p>Request tokens have retention limits.</p></main>",
            "https://example.com",
            6000,
        )
        self.assertEqual(page.title, "Retry safety")

    def test_metadata_title_precedes_heading_when_document_title_is_missing(self):
        page = _extract_content(
            '<meta name="twitter:title" content="  Research  paper "><h1>Overview</h1>',
            "https://example.com",
            6000,
        )
        self.assertEqual(page.title, "Research paper")

    def test_whitespace_cleanup_leaves_room_for_later_evidence(self):
        html = (
            "<main><p>Retries"
            + " " * 500
            + "need tokens.</p><pre>…"
            + " " * 4000
            + '"ClientToken": "abc", …</pre>'
            + "<p>Expired tokens permit a new request.</p></main>"
        )
        page = _extract_content(html, "https://example.com", 200)
        self.assertIn("Expired tokens permit a new request.", page.text)
        self.assertIn('"ClientToken": "abc"', page.text)
        self.assertTrue(
            any("preformatted" in warning for warning in page.extraction_warnings)
        )
        self.assertFalse(
            any("truncated" in warning for warning in page.extraction_warnings)
        )

    def test_normal_code_spacing_is_preserved(self):
        page = _extract_content(
            '<main><pre><code>value = "a  b"\nif value:\n    print(value)</code></pre></main>',
            "https://example.com",
            6000,
        )
        self.assertIn('value = "a  b"', page.text)
        self.assertFalse(page.extraction_warnings)

    def test_html_truncation_is_disclosed(self):
        page = _extract_content(
            "<main>" + "word " * 100 + "</main>", "https://example.com", 100
        )
        self.assertEqual(len(page.text), 100)
        self.assertIn(
            "HTML text truncated to 100 characters.", page.extraction_warnings
        )

    def test_article_survives_navigation_wrapper(self):
        html = (
            "<body><nav><a>Menu</a><main><p>"
            + ("Real financial disclosure. " * 20)
            + "</p></main></nav><footer>Contact only</footer></body>"
        )
        page = _extract_content(html, "https://example.com", 6000)
        self.assertIn("Real financial disclosure", page.text)
        self.assertNotIn("Contact only", page.text)
        self.assertNotIn("Menu", page.text)

    def test_page_wide_form_keeps_article_but_not_controls(self):
        html = (
            '<body><form><input value="hidden-token"><main><p>'
            + ("Quarterly earnings disclosure. " * 20)
            + "</p><form><label>Email</label><input></form></main></form></body>"
        )
        page = _extract_content(html, "https://example.com", 6000)
        self.assertIn("Quarterly earnings disclosure", page.text)
        self.assertNotIn("hidden-token", page.text)
        self.assertNotIn("Email", page.text)

    def test_empty_main_does_not_hide_substantive_article(self):
        html = (
            "<body><main>Loading</main><article>"
            + ("Full disclosure. " * 30)
            + "</article></body>"
        )
        page = _extract_content(html, "https://example.com", 6000)
        self.assertIn("Full disclosure", page.text)

    def test_empty_content_widget_does_not_hide_later_content(self):
        html = (
            '<body><div class="content">Stock quote</div><div class="news-content">'
            + ("Real news. " * 40)
            + "</div></body>"
        )
        page = _extract_content(html, "https://example.com", 6000)
        self.assertIn("Real news", page.text)
        self.assertNotIn("Stock quote", page.text)

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
        self.assertNotIn("var x", page.text)  # script dropped
        self.assertNotIn("copyright 2026", page.text)  # footer dropped

    def test_respects_max_chars(self):
        html = (
            "<html><body><article>"
            + ("word longenough " * 500)
            + "</article></body></html>"
        )
        page = _extract_content(html, "https://ex.com/b", max_chars=200)
        self.assertLessEqual(len(page.text), 200)


class TestBodyExtraction(unittest.TestCase):
    def test_bs4_default_unchanged(self):
        html = "<html><head><title>T</title></head><body><article><p>the real body content here for the test</p></article></body></html>"
        page = _extract_content(html, "https://ex.com/a", 6000)
        self.assertIn("real body content", page.text)


class TestScrapeUrlGuards(unittest.IsolatedAsyncioTestCase):
    async def test_non_http_url_returns_error_without_network(self):
        page = await scrape_url("ftp://example.com/file")
        self.assertIn("Only http and https", page.error)
        self.assertEqual(page.text, "")


if __name__ == "__main__":
    unittest.main()
