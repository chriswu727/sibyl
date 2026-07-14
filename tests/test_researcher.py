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
from sibyl.researcher import LLMCallError, ResearchReport, Researcher
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
    async def fake(provider, prompt, max_tokens=1500, thinking=False, json_mode=False):
        prompts_seen.append(prompt)
        p = prompt.lower()
        if "sub-questions" in p and "break this" in p:
            return "What is the state?\nWhat drives it?\nWhat is the outlook?\nWhat are risks?\nWhat is next?"
        if "search queries" in p:
            return "alpha query here\nbeta query here\ngamma query here"
        if "missing" in p or "knowledge gaps" in p:
            return "gap query one here\ngap query two here"
        if "relevance" in p:
            import json as _json
            return _json.dumps({"scores": [{"id": i, "score": 21 - i} for i in range(1, 21)]})
        if "key findings" in p:
            # findings now run in json_mode — return the JSON object shape
            import json as _json
            return _json.dumps({"findings": [f"Finding {i} with datum {i}0% [Source {i}]." for i in range(1, 11)]})
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

    async def fake_verify(findings, pages, provider):
        from sibyl.verifier import FindingVerification
        return [FindingVerification(i + 1, True, "high", [1], "") for i in range(len(findings))]

    return (mock.patch("sibyl.researcher.search_web", fake_search),
            mock.patch("sibyl.researcher.scrape_urls", fake_scrape),
            mock.patch("sibyl.analyzer.analyze_sources", fake_cross),
            mock.patch("sibyl.verifier.verify_findings", fake_verify),
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

    async def test_repeated_failure_raises_instead_of_returning_report_text(self):
        r = Researcher(_cfg())
        with mock.patch("sibyl.researcher.litellm.acompletion", mock.AsyncMock(side_effect=RuntimeError("down"))), \
             mock.patch("sibyl.researcher.asyncio.sleep", mock.AsyncMock()):
            with self.assertRaises(LLMCallError):
                await r._llm_call(_cfg().providers[0], "hello")


class TestParseFindings(unittest.TestCase):
    def test_parses_json_object(self):
        out = Researcher._parse_findings('{"findings": ["A [Source 1]", "B [Source 2]"]}')
        self.assertEqual(out, ["A [Source 1]", "B [Source 2]"])

    def test_parses_bare_json_array(self):
        out = Researcher._parse_findings('["one finding", "two finding"]')
        self.assertEqual(out, ["one finding", "two finding"])

    def test_falls_back_to_line_split(self):
        text = "1. First finding with enough length here\n2. Second finding also long enough"
        out = Researcher._parse_findings(text)
        self.assertEqual(len(out), 2)
        self.assertIn("First finding", out[0])

    def test_empty_and_garbage(self):
        self.assertEqual(Researcher._parse_findings(""), [])
        self.assertEqual(Researcher._parse_findings('{"findings": []}'), [])

    def test_strips_leading_enumerator(self):
        # A pre-numbered finding must not survive as "1. ..." (renderer adds its own)
        out = Researcher._parse_findings('{"findings": ["1. Revenue grew 40% [Source 1]", "2) Margins fell"]}')
        self.assertEqual(out[0], "Revenue grew 40% [Source 1]")
        self.assertEqual(out[1], "Margins fell")

    def test_malformed_json_falls_back(self):
        # truncated JSON → fall back to line parse of whatever is there
        out = Researcher._parse_findings('{"findings": ["good finding that is long enough to survive"')
        self.assertTrue(any("good finding" in x for x in out) or out == [])


class TestJsonMode(unittest.IsolatedAsyncioTestCase):
    async def test_json_mode_sets_response_format(self):
        r = Researcher(_cfg())
        captured = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            m = mock.Mock()
            m.choices = [mock.Mock(message=mock.Mock(content='{"findings": []}'))]
            return m

        prov = Provider(model="deepseek/deepseek-v4-flash", api_key="sk")
        with mock.patch("sibyl.researcher.litellm.acompletion", fake_acompletion):
            await r._llm_call(prov, "give me json", json_mode=True)
        self.assertEqual(captured.get("response_format"), {"type": "json_object"})

    async def test_no_json_mode_no_response_format(self):
        r = Researcher(_cfg())
        captured = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            m = mock.Mock()
            m.choices = [mock.Mock(message=mock.Mock(content="prose"))]
            return m

        prov = Provider(model="deepseek/deepseek-v4-flash", api_key="sk")
        with mock.patch("sibyl.researcher.litellm.acompletion", fake_acompletion):
            await r._llm_call(prov, "write prose")
        self.assertNotIn("response_format", captured)


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


class TestEvidenceSafety(unittest.IsolatedAsyncioTestCase):
    async def test_no_sources_abstains_without_synthesis(self):
        r = Researcher(_cfg())
        r._llm_call = _canned_llm([])
        r._synthesize = mock.AsyncMock()

        async def no_results(*args, **kwargs):
            return []

        with mock.patch("sibyl.researcher.search_web", no_results), \
             mock.patch("sibyl.researcher.scrape_urls", no_results):
            report = await r.research("obscure question", depth=1)

        self.assertEqual(report.status, "insufficient_evidence")
        self.assertFalse(report.summary)
        self.assertFalse(report.sources)
        r._synthesize.assert_not_called()

    async def test_thin_snippets_are_not_treated_as_sufficient_evidence(self):
        r = Researcher(_cfg())
        r._llm_call = _canned_llm([])
        r._synthesize = mock.AsyncMock()

        async def thin_results(*args, **kwargs):
            return [SearchResult("Thin", "https://ex.com/thin", "brief snippet " * 5)]

        async def failed_scrape(*args, **kwargs):
            return []

        with mock.patch("sibyl.researcher.search_web", thin_results), \
             mock.patch("sibyl.researcher.scrape_urls", failed_scrape):
            report = await r.research("question", depth=1)

        self.assertEqual(report.status, "insufficient_evidence")
        r._synthesize.assert_not_called()

    async def test_llm_failure_becomes_failed_report(self):
        r = Researcher(_cfg())
        r._decompose_question = mock.AsyncMock(side_effect=LLMCallError("backend unavailable"))
        report = await r.research("question", depth=2)
        self.assertEqual(report.status, "failed")
        self.assertIn("backend unavailable", report.error)
        self.assertFalse(report.summary)

    async def test_review_receives_citation_ordered_source_evidence(self):
        r = Researcher(_cfg())
        prompts = []

        async def fake(provider, prompt, max_tokens=1500, thinking=False, json_mode=False):
            prompts.append(prompt)
            if json_mode:
                return '{"findings": ["Revenue was $4B [Source 1]"]}'
            return "Revenue was $4B [Source 1]. " * 20

        r._llm_call = fake
        report = ResearchReport("q", "draft", ["draft finding"], [])
        pages = [WebPage("https://ex.com", "Evidence", "Audited revenue was $4B in 2025.")]
        await r._review_and_refine(report, pages)

        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertIn("SOURCE EVIDENCE", prompt)
            self.assertIn("Audited revenue was $4B in 2025", prompt)
            self.assertIn("Never", prompt)
            self.assertNotIn("add if missing", prompt)


if __name__ == "__main__":
    unittest.main()
