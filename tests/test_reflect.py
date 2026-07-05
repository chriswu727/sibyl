"""Reflection loop — opt-in, bounded, safe-stop. Mocked, no network.

Run: python -m unittest discover tests
"""
import json
import unittest
from unittest import mock

from sibyl.config import Config, Provider
from sibyl.researcher import Researcher


def _cfg(**kw):
    return Config(providers=[Provider(model="deepseek/deepseek-v4-flash", api_key="k", role="general")], **kw)


class TestReflect(unittest.IsolatedAsyncioTestCase):
    async def test_parses_and_stops_on_sufficient(self):
        r = Researcher(_cfg())

        async def fake_llm(provider, prompt, max_tokens=1500, thinking=False, json_mode=False):
            return json.dumps({"sufficient": True, "gaps": [], "queries": []})
        r._llm_call = fake_llm
        rep = mock.Mock(summary="s", key_findings=["f"])
        qs, suff = await r._reflect("q", rep)
        self.assertTrue(suff)
        self.assertEqual(qs, [])

    async def test_returns_queries_when_gaps(self):
        r = Researcher(_cfg())

        async def fake_llm(provider, prompt, max_tokens=1500, thinking=False, json_mode=False):
            return json.dumps({"sufficient": False, "gaps": ["missing X"], "queries": ["query one", "query two"]})
        r._llm_call = fake_llm
        rep = mock.Mock(summary="s", key_findings=["f"])
        qs, suff = await r._reflect("q", rep)
        self.assertFalse(suff)
        self.assertEqual(qs, ["query one", "query two"])

    async def test_garbage_stops_safely(self):
        r = Researcher(_cfg())

        async def fake_llm(provider, prompt, max_tokens=1500, thinking=False, json_mode=False):
            return "not json"
        r._llm_call = fake_llm
        rep = mock.Mock(summary="s", key_findings=["f"])
        qs, suff = await r._reflect("q", rep)
        self.assertEqual((qs, suff), ([], True))


if __name__ == "__main__":
    unittest.main()
