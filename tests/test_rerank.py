"""Ranked relevance rerank in _filter_sources. LLM mocked, no network.

Run: python -m unittest discover tests
"""
import json
import unittest
from unittest import mock

from sibyl.config import Config, Provider
from sibyl.researcher import Researcher
from sibyl.scraper import WebPage


def _cfg(**kw):
    return Config(providers=[Provider(model="deepseek/deepseek-v4-flash", api_key="k", role="general")], **kw)


class TestParseScores(unittest.TestCase):
    def test_parses_object(self):
        s = Researcher._parse_scores('{"scores":[{"id":1,"score":9},{"id":2,"score":3}]}', 2)
        self.assertEqual(s, {1: 9.0, 2: 3.0})

    def test_bare_list_and_truncated(self):
        self.assertEqual(Researcher._parse_scores('[{"id":1,"score":5}]', 1), {1: 5.0})
        self.assertEqual(Researcher._parse_scores('{"scores":[{"id":1', 1), {})  # truncated → empty


class TestFilterRerank(unittest.IsolatedAsyncioTestCase):
    async def test_keeps_top_n_by_score(self):
        cfg = _cfg(rerank_top_n=2, reranker="llm")
        r = Researcher(cfg)
        pages = [WebPage(url=f"u{i}", title=str(i), text="body " * 30) for i in range(5)]

        async def fake_llm(provider, prompt, max_tokens=1500, thinking=False, json_mode=False):
            # rank source 3 and 5 highest
            return json.dumps({"scores": [{"id": 1, "score": 2}, {"id": 2, "score": 1},
                                          {"id": 3, "score": 10}, {"id": 4, "score": 4},
                                          {"id": 5, "score": 9}]})
        r._llm_call = fake_llm
        kept = await r._filter_sources("q", pages)
        self.assertEqual(len(kept), 2)
        self.assertEqual({p.title for p in kept}, {"2", "4"})  # ids 3 and 5 → indices 2 and 4

    async def test_empty_scores_falls_back(self):
        cfg = _cfg(rerank_top_n=3, reranker="llm")
        r = Researcher(cfg)
        pages = [WebPage(url=f"u{i}", title=str(i), text="body " * 30) for i in range(5)]

        async def fake_llm(provider, prompt, max_tokens=1500, thinking=False, json_mode=False):
            return "not json"
        r._llm_call = fake_llm
        kept = await r._filter_sources("q", pages)
        self.assertEqual(len(kept), 3)  # falls back to pages[:top_n]

    async def test_reranker_none_skips_llm(self):
        cfg = _cfg(rerank_top_n=2, reranker="none")
        r = Researcher(cfg)
        called = []
        async def fake_llm(*a, **k):
            called.append(1); return ""
        r._llm_call = fake_llm
        pages = [WebPage(url=f"u{i}", title=str(i), text="t") for i in range(5)]
        kept = await r._filter_sources("q", pages)
        self.assertEqual(len(kept), 2)
        self.assertEqual(called, [])  # no LLM call when reranker=none

    async def test_default_lexical_reranker_skips_llm_and_reorders(self):
        cfg = _cfg(rerank_top_n=1)
        researcher = Researcher(cfg)
        pages = [
            WebPage(url="cooking", title="Cooking", text="bread recipe " * 30),
            WebPage(
                url="tennis",
                title="2018 Madrid Open men's singles",
                text="A Serbian quarterfinalist reached this stage. " * 20,
            ),
        ]

        async def fail_llm(*args, **kwargs):
            raise AssertionError("lexical reranking must not call the LLM")

        researcher._llm_call = fail_llm
        kept = await researcher._filter_sources(
            "Serbian quarterfinalist 2018 Madrid Open men's singles", pages
        )

        self.assertEqual([page.url for page in kept], ["tennis"])

    async def test_flashrank_failure_falls_back_to_local_reranker(self):
        cfg = _cfg(rerank_top_n=1, reranker="flashrank")
        researcher = Researcher(cfg)
        pages = [
            WebPage(url="noise", title="Noise", text="unrelated text " * 30),
            WebPage(url="match", title="Alpha beta", text="alpha beta " * 30),
        ]

        with mock.patch.object(
            researcher, "_flashrank_rerank", side_effect=ImportError("not installed")
        ):
            kept = await researcher._filter_sources("alpha beta", pages)

        self.assertEqual([page.url for page in kept], ["match"])


if __name__ == "__main__":
    unittest.main()
