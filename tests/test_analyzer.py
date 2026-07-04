"""Cross-analysis JSON parsing + graceful fallback. LLM mocked, no network.

Run: python -m unittest discover tests
"""
import json
import unittest
from unittest import mock

from sibyl.analyzer import analyze_sources
from sibyl.config import Provider
from sibyl.scraper import WebPage


def _pages():
    return [WebPage(url=f"u{i}", title=f"T{i}", text="content " * 40) for i in range(3)]


def _resp(content):
    m = mock.Mock()
    m.choices = [mock.Mock(message=mock.Mock(content=content))]
    return m


class TestAnalyzeSources(unittest.IsolatedAsyncioTestCase):
    async def test_parses_json(self):
        payload = json.dumps({
            "overall_sentiment": "positive",
            "sentiment_breakdown": {"positive": 3, "negative": 1, "neutral": 0},
            "consensus": ["they agree on A", "and on B"],
            "disagreements": ["X vs Y"],
            "unique": ["Source 2: a unique point"],
        })

        async def fake(**kwargs):
            return _resp(payload)

        with mock.patch("sibyl.analyzer.litellm.acompletion", fake):
            ca = await analyze_sources(_pages(), "q", Provider(model="deepseek/deepseek-v4-flash", api_key="k"))
        self.assertEqual(ca.overall_sentiment, "positive")
        self.assertEqual(ca.sentiment_breakdown["positive"], 3)
        self.assertEqual(len(ca.consensus_points), 2)
        self.assertEqual(ca.disagreement_points, ["X vs Y"])
        self.assertEqual(len(ca.unique_insights), 1)

    async def test_empty_content_degrades_gracefully(self):
        # Empty content every attempt → defaults, no crash (json.loads("") would raise)
        async def fake(**kwargs):
            return _resp("")

        with mock.patch("sibyl.analyzer.litellm.acompletion", fake):
            ca = await analyze_sources(_pages(), "q", Provider(model="deepseek/deepseek-v4-flash", api_key="k"))
        self.assertEqual(ca.overall_sentiment, "mixed")
        self.assertEqual(ca.consensus_points, [])
        self.assertEqual(ca.sentiment_breakdown, {"positive": 0, "negative": 0, "neutral": 0})

    async def test_non_json_degrades_gracefully(self):
        async def fake(**kwargs):
            return _resp("this is not json at all")

        with mock.patch("sibyl.analyzer.litellm.acompletion", fake):
            ca = await analyze_sources(_pages(), "q", Provider(model="deepseek/deepseek-v4-flash", api_key="k"))
        self.assertEqual(ca.overall_sentiment, "mixed")
        self.assertEqual(ca.consensus_points, [])

    async def test_requests_json_object_format(self):
        captured = {}

        async def fake(**kwargs):
            captured.update(kwargs)
            return _resp(json.dumps({"overall_sentiment": "neutral"}))

        with mock.patch("sibyl.analyzer.litellm.acompletion", fake):
            await analyze_sources(_pages(), "q", Provider(model="deepseek/deepseek-v4-flash", api_key="k"))
        self.assertEqual(captured.get("response_format"), {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
