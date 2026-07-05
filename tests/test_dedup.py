"""Canonical-URL dedup. Pure functions, no network.

Run: python -m unittest discover tests
"""
import unittest

from sibyl.dedup import canonical_url, dedup_pages
from sibyl.scraper import WebPage


class TestCanonicalUrl(unittest.TestCase):
    def test_scheme_www_slash_fragment(self):
        a = canonical_url("http://www.example.com/page/")
        b = canonical_url("https://example.com/page#section")
        self.assertEqual(a, b)

    def test_strips_tracking_params(self):
        a = canonical_url("https://ex.com/p?utm_source=x&id=5&gclid=abc")
        b = canonical_url("https://ex.com/p?id=5")
        self.assertEqual(a, b)

    def test_distinct_paths_differ(self):
        self.assertNotEqual(canonical_url("https://ex.com/a"), canonical_url("https://ex.com/b"))


class TestDedupPages(unittest.TestCase):
    def test_keeps_first_position_longest_text(self):
        pages = [
            WebPage(url="https://www.ex.com/a/", title="A", text="short"),
            WebPage(url="https://ex.com/b", title="B", text="other page"),
            WebPage(url="http://ex.com/a", title="A2", text="a much longer body of text here"),
        ]
        out = dedup_pages(pages)
        self.assertEqual(len(out), 2)                      # a-variants collapsed
        self.assertEqual(out[0].url, "https://www.ex.com/a/")  # first position preserved
        self.assertEqual(out[0].text, "a much longer body of text here")  # longer text won

    def test_no_duplicates_unchanged(self):
        pages = [WebPage(url=f"https://ex.com/{i}", title=str(i), text="t") for i in range(4)]
        self.assertEqual(len(dedup_pages(pages)), 4)


if __name__ == "__main__":
    unittest.main()
