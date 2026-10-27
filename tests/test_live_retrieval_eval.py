"""Model-free live retrieval metric tests. No network."""

import asyncio
import unittest

from sibyl.evidence import (
    BundleDiagnostics,
    EvidencePassage,
    EvidenceSource,
    SourceBundle,
)
from sibyl.live_retrieval_eval import LiveRetrievalCase, evaluate_live_retrieval


def bundle(
    text,
    status="ok",
    sufficiency="sufficient",
    query_complexity="single_step",
    recommended_action="synthesize",
):
    diagnostics = BundleDiagnostics(
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        10,
        10,
        7000,
        7000,
        125,
        evidence_sufficiency=sufficiency,
        search_providers=["tavily"],
        max_source_query_term_coverage=0.75,
        metadata_fallbacks=1,
        query_complexity=query_complexity,
        recommended_action=recommended_action,
        refinement_searches=1,
        substantive_sources=1,
        substantive_domains=1,
    )
    passage = EvidencePassage("P1", "sb/S1/P1", text, "hash")
    source = EvidenceSource(
        "S1",
        "https://example.com",
        "Source",
        "now",
        "hash",
        "web",
        len(text),
        [passage],
    )
    return SourceBundle("1.6", "sb", "q", status, [source], diagnostics)


class TestLiveRetrievalEval(unittest.IsolatedAsyncioTestCase):
    async def test_measures_answer_coverage_trap_safety_and_latency(self):
        cases = [
            LiveRetrievalCase("answer", "Who?", "Dušan Lajović", []),
            LiveRetrievalCase(
                "trap", "Who walked on Mars?", "NO_ANSWER", ["no human has"]
            ),
        ]

        async def gather(query, **kwargs):
            if "Mars" in query:
                return bundle("No human has walked on Mars.")
            return bundle("The Serbian player was Dusan Lajovic.")

        result = await evaluate_live_retrieval(cases, repeats=2, gather=gather)

        self.assertEqual(result.answer_coverage, 1.0)
        self.assertEqual(result.trap_safe_rate, 1.0)
        self.assertEqual(result.stable_case_rate, 1.0)
        self.assertEqual(result.status_ok_rate, 1.0)
        self.assertEqual(result.answerable_ready_rate, 1.0)
        self.assertEqual(result.ready_answer_precision, 1.0)
        self.assertEqual(result.p95_latency_ms, 125)
        self.assertEqual(result.cases[0].runs[0].search_providers, ["tavily"])
        self.assertEqual(result.cases[0].runs[0].metadata_fallbacks, 1)
        self.assertEqual(
            result.cases[0].runs[0].max_source_query_term_coverage,
            0.75,
        )
        self.assertEqual(result.cases[0].runs[0].query_complexity, "single_step")
        self.assertEqual(result.cases[0].runs[0].recommended_action, "synthesize")
        self.assertEqual(result.cases[0].runs[0].refinement_searches, 1)
        self.assertEqual(result.cases[0].runs[0].refinement_failures, 0)
        self.assertEqual(result.cases[0].runs[0].substantive_sources, 1)
        self.assertEqual(result.cases[0].runs[0].substantive_domains, 1)

    async def test_safe_traps_do_not_distort_answer_readiness(self):
        cases = [
            LiveRetrievalCase("answer", "Who?", "correct answer", []),
            LiveRetrievalCase("trap", "Impossible?", "NO_ANSWER", []),
        ]

        async def gather(query, **kwargs):
            if "Impossible" in query:
                return bundle(
                    "No supporting evidence.",
                    status="insufficient_evidence",
                    sufficiency="insufficient",
                )
            return bundle("Wrong material with no expected answer.")

        result = await evaluate_live_retrieval(cases, gather=gather)

        self.assertEqual(result.trap_safe_rate, 1.0)
        self.assertEqual(result.status_ok_rate, 0.5)
        self.assertEqual(result.answerable_ready_rate, 0.0)
        self.assertEqual(result.ready_answer_precision, 0.0)

    async def test_requires_cases_and_positive_repeats(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            await evaluate_live_retrieval([])
        with self.assertRaisesRegex(ValueError, "repeats"):
            await evaluate_live_retrieval(
                [LiveRetrievalCase("a", "q", "a", [])], repeats=0
            )

    async def test_records_retrieval_errors_and_reports_progress(self):
        cases = [LiveRetrievalCase("broken", "Who?", "answer", [])]
        progress = []

        async def gather(query, **kwargs):
            raise RuntimeError("network unavailable")

        result = await evaluate_live_retrieval(
            cases,
            gather=gather,
            progress=lambda done, total, case: progress.append(
                (done, total, case.case_id)
            ),
        )

        self.assertEqual(result.status_ok_rate, 0.0)
        self.assertEqual(result.answerable_ready_rate, 0.0)
        self.assertEqual(result.ready_answer_precision, 1.0)
        self.assertEqual(result.cases[0].runs[0].status, "failed")
        self.assertEqual(
            result.cases[0].runs[0].error,
            "RuntimeError: network unavailable",
        )
        self.assertEqual(progress, [(1, 1, "broken")])

    async def test_records_provider_request_outcomes_per_run(self):
        from unittest import mock

        from sibyl.search import search_mojeek, search_yahoo

        client = mock.Mock()
        client.get = mock.AsyncMock(return_value=mock.Mock(status_code=403, text=""))

        async def gather(query, **kwargs):
            await asyncio.gather(
                search_mojeek(query, client=client),
                search_yahoo(query, client=client),
            )
            await search_mojeek(query, client=client)
            return bundle("answer")

        cases = [LiveRetrievalCase("a", "Who?", "answer", [])]
        result = await evaluate_live_retrieval(cases, gather=gather)
        self.assertEqual(
            result.cases[0].runs[0].search_attempts,
            {"mojeek": {"403": 2}, "yahoo": {"403": 1}},
        )


if __name__ == "__main__":
    unittest.main()


class TestHostSourceRetrievalEval(unittest.IsolatedAsyncioTestCase):
    async def test_reads_use_frozen_urls_and_the_sufficiency_readiness_rule(self):
        calls = []

        async def read(urls, question):
            calls.append((urls, question))
            # A read is "ok" whenever a page was readable; readiness still
            # needs sufficient evidence.
            return bundle(
                "The answer is Dimorphos.",
                status="ok",
                sufficiency="sufficient" if "1" in urls[0] else "partial",
            )

        cases = [
            LiveRetrievalCase(
                "a", "Which moonlet?", "Dimorphos", [], urls=["https://e/1"]
            ),
            LiveRetrievalCase(
                "b", "Which moonlet again?", "Dimorphos", [], urls=["https://e/2"]
            ),
        ]
        result = await evaluate_live_retrieval(cases, read=read)
        self.assertEqual(calls[0], (["https://e/1"], "Which moonlet?"))
        runs = [case.runs[0] for case in result.cases]
        self.assertEqual([run.status for run in runs], ["ok", "insufficient_evidence"])
        self.assertEqual([run.read_status for run in runs], ["ok", "ok"])
        self.assertEqual(result.answer_coverage, 1.0)
        self.assertEqual(result.answerable_ready_rate, 0.5)
        self.assertEqual(result.ready_answer_precision, 1.0)

    async def test_trap_read_is_safe_only_without_sufficient_unrefuted_evidence(self):
        async def read(urls, question):
            return bundle("Einstein won in 1921.", sufficiency="sufficient")

        trap = LiveRetrievalCase(
            "t", "Relativity Nobel year?", "NO_ANSWER", ["photoelectric"], urls=["u"]
        )
        result = await evaluate_live_retrieval([trap], read=read)
        self.assertEqual(result.trap_safe_rate, 0.0)

    async def test_case_without_urls_fails_safely(self):
        from sibyl.research_sources import read_sources

        case = LiveRetrievalCase("e", "q", "x", [], urls=[])
        result = await evaluate_live_retrieval([case], read=read_sources)
        self.assertEqual(result.cases[0].runs[0].status, "failed")
        self.assertEqual(result.answer_coverage, 0.0)
