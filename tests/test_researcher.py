"""Researcher tests — guard the performance-critical behaviors.

These run fully offline: the LLM and network calls are mocked, so they're fast
and deterministic and assert the *structure* of the pipeline (thinking disabled,
fast mode skips review, sub-analysis is depth-3 only, synthesis batch shape,
density directive present).

Run: python -m unittest discover tests
"""
import unittest
from unittest import mock

from sibyl.config import Config, Provider
from sibyl.researcher import Researcher
from sibyl.search import SearchResult
from sibyl.scraper import WebPage
from sibyl.analyzer import CrossAnalysis


def _cfg(fast=False):
    return Config(
        providers=[Provider(model="deepseek/deepseek-v4-flash", api_key="sk", role="general"),
                   Provider(model="deepseek/deepseek-v4-flash", api_key="sk", role="analysis")],
        fast=fast,
    )


def _canned_llm(prompts_seen):
    """A stand-in for _llm_call that records prompts and returns parseable text."""
    async def fake(provider, prompt, max_tokens=1500, thinking=False):
        prompts_seen.append(prompt)
        p = prompt.lower()
        if "sub-questions" in p and "break this" in p:
            return "What is the state?\nWhat drives it?\nWhat is the outlook?\nWhat are risks?\nWhat is next?"
        if "search queries" in p:
            return "alpha query here\nbeta query here\ngamma query here"
        if "missing" in p or "knowledge gaps" in p:
            return "gap query one here\ngap query two here"
        if "relevance" in p:
            return "1,2,3,4,5"
        if "key findings" in p:
            return "\n".join(f"{i}. Finding {i} with datum {i}0% [Source {i}]." for i in range(1, 11))
        # summary / analysis / predictions / review → substantial prose
        return ("This is a substantial analytical paragraph with [Source 1] data. " * 20)
    return fake


def _patch_io():
    """Patch network + cross-analysis + warm-cache so research() runs offline."""
    async def fake_search(*a, **k):
        return [SearchResult(title=f"T{i}", url=f"https://ex.com/{i}", snippet="snippet text " * 10, source="web")
                for i in range(6)]

    async def fake_scrape(urls, *a, **k):
        return [WebPage(url=u, title=f"Title {u}", text="Full page content. " * 60) for u in urls]

    async def fake_cross(pages, query, provider):
        return CrossAnalysis(["agree A", "agree B"], ["disagree X"], ["unique Y"], "positive", {"positive": 2})

    async def fake_acompletion(**kwargs):  # _warm_cache
        return mock.Mock()

    return (mock.patch("sibyl.researcher.search_web", fake_search),
            mock.patch("sibyl.researcher.scrape_urls", fake_scrape),
            mock.patch("sibyl.analyzer.analyze_sources", fake_cross),
            mock.patch("sibyl.researcher.litellm.acompletion", fake_acompletion))


class TestLlmCallThinking(unittest.IsolatedAsyncioTestCase):
    async def test_thinking_disabled_adds_extra_body_for_deepseek(self):
        r = Researcher(_cfg())
        captured = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            m = mock.Mock()
            m.choices = [mock.Mock(message=mock.Mock(content="ok"))]
            return m

        prov = Provider(model="deepseek/deepseek-v4-flash", api_key="sk")
        with mock.patch("sibyl.researcher.litellm.acompletion", fake_acompletion):
            await r._llm_call(prov, "hello", thinking=False)
        self.assertEqual(captured.get("extra_body"), {"thinking": {"type": "disabled"}})

    async def test_thinking_true_no_extra_body(self):
        r = Researcher(_cfg())
        captured = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            m = mock.Mock()
            m.choices = [mock.Mock(message=mock.Mock(content="ok"))]
            return m

        prov = Provider(model="deepseek/deepseek-v4-flash", api_key="sk")
        with mock.patch("sibyl.researcher.litellm.acompletion", fake_acompletion):
            await r._llm_call(prov, "hello", thinking=True)
        self.assertNotIn("extra_body", captured)

    async def test_non_deepseek_never_gets_thinking_toggle(self):
        r = Researcher(_cfg())
        captured = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            m = mock.Mock()
            m.choices = [mock.Mock(message=mock.Mock(content="ok"))]
            return m

        prov = Provider(model="gpt-4o-mini", api_key="sk")
        with mock.patch("sibyl.researcher.litellm.acompletion", fake_acompletion):
            await r._llm_call(prov, "hello", thinking=False)
        self.assertNotIn("extra_body", captured)


class TestPipelineStructure(unittest.IsolatedAsyncioTestCase):
    async def _run(self, depth, fast=False):
        r = Researcher(_cfg(fast=fast))
        prompts = []
        r._llm_call = _canned_llm(prompts)
        review_spy = mock.AsyncMock(side_effect=lambda report, pages, language="auto": report)
        r._review_and_refine = review_spy
        subq_spy = mock.AsyncMock(return_value="sub answer [Source 1]")
        r._analyze_sub_question = subq_spy
        patches = _patch_io()
        for p in patches:
            p.start()
        try:
            report = await r.research("Test question about markets?", depth=depth)
        finally:
            for p in patches:
                p.stop()
        return report, prompts, review_spy, subq_spy

    async def test_fast_mode_skips_review(self):
        _, _, review_spy, _ = await self._run(depth=2, fast=True)
        review_spy.assert_not_called()

    async def test_default_runs_review(self):
        _, _, review_spy, _ = await self._run(depth=2, fast=False)
        review_spy.assert_called_once()

    async def test_subanalysis_is_depth3_only(self):
        _, _, _, subq_spy_d2 = await self._run(depth=2)
        self.assertEqual(subq_spy_d2.call_count, 0)
        _, _, _, subq_spy_d3 = await self._run(depth=3)
        self.assertGreater(subq_spy_d3.call_count, 0)

    async def test_depth2_has_analysis_no_predictions(self):
        report, _, _, _ = await self._run(depth=2)
        self.assertTrue(report.analysis)
        self.assertFalse(report.predictions)

    async def test_depth3_has_predictions(self):
        report, _, _, _ = await self._run(depth=3)
        self.assertTrue(report.analysis)
        self.assertTrue(report.predictions)

    async def test_density_directive_present_in_synthesis_prompts(self):
        _, prompts, _, _ = await self._run(depth=2)
        joined = "\n".join(prompts)
        self.assertIn("information-dense", joined)

    async def test_report_is_wellformed(self):
        report, _, _, _ = await self._run(depth=2)
        self.assertTrue(report.summary)
        self.assertGreaterEqual(len(report.key_findings), 3)
        self.assertTrue(report.sources)
        self.assertIn("positive", report.cross_analysis.lower())


if __name__ == "__main__":
    unittest.main()
