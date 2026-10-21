"""Researcher tests — guard the performance-critical behaviors.

These run fully offline: the LLM and network calls are mocked, so they're fast
and deterministic and assert the *structure* of the pipeline (thinking disabled,
fast mode skips review, sub-analysis is depth-3 only, synthesis batch shape,
density directive present).

Run: python -m unittest discover tests
"""

import unittest
from unittest import mock

from sibyl.analyzer import CrossAnalysis
from sibyl.config import Config, Provider
from sibyl.evidence import (
    BundleDiagnostics,
    EvidencePassage,
    EvidenceSource,
    SourceBundle,
)
from sibyl.researcher import LLMCallError, Researcher, ResearchReport
from sibyl.scraper import WebPage


def evidence_bundle(query="question", count=6, origin="direct_fetch"):
    sources = []
    for index in range(count):
        text = f"Evidence {query} {index}. " * 40
        sources.append(
            EvidenceSource(
                f"{query}-{index}",
                f"https://example.com/{query}/{index}",
                f"Title {index}",
                "2026-10-20T00:00:00Z",
                f"hash-{query}-{index}",
                "web",
                len(text),
                [
                    EvidencePassage(
                        "P1", f"{query}/{index}/P1", text, f"passage-{query}-{index}"
                    )
                ],
                content_origin=origin,
            )
        )
    return SourceBundle(
        "1.6",
        f"bundle-{query}",
        query,
        "ok" if count else "insufficient_evidence",
        sources,
        BundleDiagnostics(
            count,
            count,
            count,
            count,
            0,
            0,
            0,
            count,
            8,
            8,
            4000,
            4000,
            1,
            evidence_sufficiency="sufficient" if count else "insufficient",
            recommended_action="synthesize" if count else "refine_query",
        ),
    )


def _cfg(fast=False):
    return Config(
        providers=[
            Provider(model="deepseek/deepseek-v4-flash", api_key="sk", role="general"),
            Provider(model="deepseek/deepseek-v4-flash", api_key="sk", role="analysis"),
        ],
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

            return _json.dumps(
                {"scores": [{"id": i, "score": 21 - i} for i in range(1, 21)]}
            )
        if "key findings" in p:
            # findings now run in json_mode — return the JSON object shape
            import json as _json

            return _json.dumps(
                {
                    "findings": [
                        f"Finding {i} with datum {i}0% [Source {i}]."
                        for i in range(1, 11)
                    ]
                }
            )
        # summary / analysis / predictions / review → substantial prose
        return "This is a substantial analytical paragraph with [Source 1] data. " * 20

    return fake


def _patch_io():
    """Patch network + cross-analysis + warm-cache so research() runs offline."""

    async def fake_cross(pages, query, provider):
        return CrossAnalysis(
            ["agree A", "agree B"],
            ["disagree X"],
            ["unique Y"],
            "positive",
            {"positive": 2},
        )

    async def fake_acompletion(**kwargs):  # _warm_cache
        return mock.Mock()

    async def fake_verify(findings, pages, provider, **kwargs):
        from sibyl.verifier import FindingVerification

        return [
            FindingVerification(i + 1, True, "high", [1], "")
            for i in range(len(findings))
        ]

    async def fake_bundle(query, **kwargs):
        return evidence_bundle(query)

    return (
        mock.patch("sibyl.retrieval.gather_source_bundle", fake_bundle),
        mock.patch("sibyl.analyzer.analyze_sources", fake_cross),
        mock.patch("sibyl.verifier.verify_findings", fake_verify),
        mock.patch("sibyl.researcher.litellm.acompletion", fake_acompletion),
    )


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
        with (
            mock.patch(
                "sibyl.researcher.litellm.acompletion",
                mock.AsyncMock(side_effect=RuntimeError("down")),
            ),
            mock.patch("sibyl.researcher.asyncio.sleep", mock.AsyncMock()),
        ):
            with self.assertRaises(LLMCallError):
                await r._llm_call(_cfg().providers[0], "hello")


class TestParseFindings(unittest.TestCase):
    def test_parses_json_object(self):
        out = Researcher._parse_findings(
            '{"findings": ["A [Source 1]", "B [Source 2]"]}'
        )
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
        out = Researcher._parse_findings(
            '{"findings": ["1. Revenue grew 40% [Source 1]", "2) Margins fell"]}'
        )
        self.assertEqual(out[0], "Revenue grew 40% [Source 1]")
        self.assertEqual(out[1], "Margins fell")

    def test_malformed_json_falls_back(self):
        # truncated JSON → fall back to line parse of whatever is there
        out = Researcher._parse_findings(
            '{"findings": ["good finding that is long enough to survive"'
        )
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
        review_spy = mock.AsyncMock(
            side_effect=lambda report, pages, language="auto": report
        )
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
        self.assertEqual(report.verification_scope, "report_blocks_model_checked")
        self.assertTrue(report.sources[0].passage_ids)


class TestEvidenceSafety(unittest.IsolatedAsyncioTestCase):
    async def test_reflection_retrieves_new_query_then_resynthesizes(self):
        config = _cfg()
        config.reflect_rounds = 2
        config.verify_claims = False
        researcher = Researcher(config)
        researcher._decompose_question = mock.AsyncMock(return_value=[])
        researcher._reflect = mock.AsyncMock(
            side_effect=[(["replication limits"], False), ([], True)]
        )
        researcher._synthesize = mock.AsyncMock(
            side_effect=[
                ResearchReport(
                    "question", "Initial [Source 1]", ["Initial [Source 1]"], []
                ),
                ResearchReport(
                    "question", "Revised [Source 1]", ["Revised [Source 1]"], []
                ),
            ]
        )
        researcher._review_and_refine = mock.AsyncMock(
            side_effect=lambda report, *args: report
        )
        with mock.patch(
            "sibyl.retrieval.gather_source_bundle",
            mock.AsyncMock(
                side_effect=lambda query, **kwargs: evidence_bundle(query, 1)
            ),
        ) as retrieve:
            report = await researcher.research("question", depth=2)
        self.assertEqual(
            [call.args[0] for call in retrieve.call_args_list],
            ["question", "replication limits"],
        )
        self.assertEqual(researcher._synthesize.await_count, 2)
        self.assertEqual(report.summary, "Revised [Source 1]")
        self.assertEqual(report.research_stats["reflection_rounds"], 2)
        self.assertEqual(
            [step["round"] for step in report.research_stats["steps"]], [1, 2]
        )

    async def test_reflection_stops_when_search_adds_no_usable_evidence(self):
        config = _cfg()
        config.reflect_rounds = 2
        config.verify_claims = False
        researcher = Researcher(config)
        researcher._decompose_question = mock.AsyncMock(return_value=[])
        researcher._reflect = mock.AsyncMock(
            return_value=(["replication limits"], False)
        )
        researcher._synthesize = mock.AsyncMock(
            return_value=ResearchReport("q", "S [Source 1]", ["F [Source 1]"], [])
        )
        researcher._review_and_refine = mock.AsyncMock(
            side_effect=lambda report, *args: report
        )
        with mock.patch(
            "sibyl.retrieval.gather_source_bundle",
            mock.AsyncMock(return_value=evidence_bundle("same", 1)),
        ):
            report = await researcher.research("q", depth=2)
        self.assertEqual(researcher._reflect.await_count, 1)
        self.assertEqual(researcher._synthesize.await_count, 1)
        self.assertIn(
            "Follow-up research did not add usable evidence.", report.limitations
        )

    async def test_failed_parallel_model_work_cancels_siblings(self):
        import asyncio

        from sibyl.async_utils import gather_owned

        started = asyncio.Event()
        stopped = asyncio.Event()

        async def pending():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def failing():
            await started.wait()
            raise LLMCallError("failed")

        with self.assertRaises(LLMCallError):
            await gather_owned(pending(), failing())
        self.assertTrue(stopped.is_set())

    async def test_short_supported_report_is_not_regenerated_to_fill_a_template(self):
        r = Researcher(_cfg())
        r._llm_call = mock.AsyncMock(
            side_effect=[
                "A short answer [Source 1].",
                '{"findings": ["One finding [Source 1]."]}',
            ]
        )
        report = await r._synthesize(
            "q",
            [],
            [WebPage("https://example.com", "Evidence", "Substantive evidence. " * 40)],
            1,
        )
        self.assertEqual(r._llm_call.await_count, 2)
        self.assertEqual(len(report.key_findings), 1)
        prompts = " ".join(call.args[1] for call in r._llm_call.call_args_list)
        self.assertNotIn("10-15", prompts)
        self.assertNotIn("at least one number", prompts)

    async def test_model_budget_is_reserved_before_each_provider_attempt(self):
        from sibyl.research_state import ResearchBudgetExceeded, ResearchNotebook

        notebook = ResearchNotebook("q", max_model_calls=1)
        r = Researcher(_cfg(), notebook=notebook)
        with (
            mock.patch(
                "sibyl.researcher.litellm.acompletion",
                mock.AsyncMock(side_effect=RuntimeError("failed")),
            ) as provider,
            mock.patch("sibyl.researcher.asyncio.sleep", mock.AsyncMock()),
        ):
            with self.assertRaises(ResearchBudgetExceeded):
                await r._llm_call(_cfg().providers[0], "q")
        self.assertEqual(provider.await_count, 1)
        self.assertEqual(notebook.model_calls, 1)

    async def test_compaction_preserves_source_numbering_when_a_source_is_irrelevant(
        self,
    ):
        r = Researcher(_cfg())
        r._llm_call = mock.AsyncMock(side_effect=["IRRELEVANT", "A compacted passage"])
        pages = [
            WebPage("https://one.example", "one", "first original"),
            WebPage("https://two.example", "two", "second original"),
        ]
        compacted = await r._compact_sources("q", pages)
        self.assertEqual([page.url for page in compacted], [page.url for page in pages])
        self.assertEqual(compacted[0].text, "first original")

    async def test_insufficient_diagnostics_block_synthesis_even_with_long_pages(self):
        from dataclasses import replace

        bundle = evidence_bundle()
        bundle = replace(
            bundle,
            status="insufficient_evidence",
            diagnostics=replace(bundle.diagnostics, recommended_action="refine_query"),
        )
        r = Researcher(_cfg())
        r._synthesize = mock.AsyncMock()
        with mock.patch(
            "sibyl.retrieval.gather_source_bundle", mock.AsyncMock(return_value=bundle)
        ):
            report = await r.research("q", depth=1)
        self.assertEqual(report.status, "insufficient_evidence")
        r._synthesize.assert_not_awaited()

    async def test_no_sources_abstains_without_synthesis(self):
        r = Researcher(_cfg())
        r._llm_call = _canned_llm([])
        r._synthesize = mock.AsyncMock()

        with (
            mock.patch(
                "sibyl.retrieval.gather_source_bundle",
                mock.AsyncMock(return_value=evidence_bundle(count=0)),
            ),
        ):
            report = await r.research("obscure question", depth=1)

        self.assertEqual(report.status, "insufficient_evidence")
        self.assertFalse(report.summary)
        self.assertFalse(report.sources)
        r._synthesize.assert_not_called()

    async def test_thin_snippets_are_not_treated_as_sufficient_evidence(self):
        r = Researcher(_cfg())
        r._llm_call = _canned_llm([])
        r._synthesize = mock.AsyncMock()

        with (
            mock.patch(
                "sibyl.retrieval.gather_source_bundle",
                mock.AsyncMock(return_value=evidence_bundle(origin="search_snippet")),
            ),
        ):
            report = await r.research("question", depth=1)

        self.assertEqual(report.status, "insufficient_evidence")
        r._synthesize.assert_not_called()

    async def test_llm_failure_becomes_failed_report(self):
        r = Researcher(_cfg())
        r._decompose_question = mock.AsyncMock(
            side_effect=LLMCallError("backend unavailable")
        )
        report = await r.research("question", depth=2)
        self.assertEqual(report.status, "failed")
        self.assertIn("backend unavailable", report.error)
        self.assertFalse(report.summary)

    async def test_review_receives_citation_ordered_source_evidence(self):
        r = Researcher(_cfg())
        prompts = []

        async def fake(
            provider, prompt, max_tokens=1500, thinking=False, json_mode=False
        ):
            prompts.append(prompt)
            if json_mode:
                return '{"findings": ["Revenue was $4B [Source 1]"]}'
            return "Revenue was $4B [Source 1]. " * 20

        r._llm_call = fake
        report = ResearchReport("q", "draft", ["draft finding"], [])
        pages = [
            WebPage("https://ex.com", "Evidence", "Audited revenue was $4B in 2025.")
        ]
        await r._review_and_refine(report, pages)

        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertIn("SOURCE EVIDENCE", prompt)
            self.assertIn("Audited revenue was $4B in 2025", prompt)
            self.assertIn("Never", prompt)
            self.assertNotIn("add if missing", prompt)


if __name__ == "__main__":
    unittest.main()
