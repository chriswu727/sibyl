"""Fixed-plan evidence-loop evaluator tests. No network."""
import unittest

from sibyl.evidence import (
    BundleDiagnostics,
    EvidencePassage,
    EvidenceSource,
    SourceBundle,
)
from sibyl.evidence_loop_eval import EvidenceLoopEvalCase, evaluate_evidence_loops


def bundle(query: str, *, action: str, text: str) -> SourceBundle:
    status = "ok" if action == "synthesize" else "insufficient_evidence"
    sufficiency = "sufficient" if action == "synthesize" else "insufficient"
    passage = EvidencePassage("P1", f"sb/S1/P1", text, "hash")
    source = EvidenceSource(
        "S1",
        "https://example.com",
        "Example",
        "2026-07-21T00:00:00+00:00",
        "hash",
        "web",
        len(text),
        [passage],
    )
    return SourceBundle(
        "1.6",
        f"sb_{query.replace(' ', '_')}",
        query,
        status,
        [source],
        BundleDiagnostics(
            2,
            2,
            2,
            2,
            0,
            0,
            0,
            1,
            10,
            10,
            7000,
            7000,
            2,
            evidence_sufficiency=sufficiency,
            recommended_action=action,
            query_complexity="single_step",
        ),
    )


class TestEvidenceLoopEval(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_plan_passes_only_with_ready_answer_evidence(self):
        case = EvidenceLoopEvalCase(
            "cuda",
            "In what year was the company that created CUDA founded?",
            "1993",
            [],
            ["Who created CUDA?", "When was NVIDIA founded?"],
        )

        async def gather(query, **kwargs):
            text = "NVIDIA was founded in 1993." if "founded" in query else "CUDA was created by NVIDIA."
            return bundle(query, action="synthesize", text=text)

        result = await evaluate_evidence_loops([case], gather=gather)

        self.assertEqual(result.decomposition_rate, 1.0)
        self.assertEqual(result.plan_execution_rate, 1.0)
        self.assertEqual(result.ready_rate, 1.0)
        self.assertEqual(result.answer_coverage, 1.0)
        self.assertEqual(result.plan_execution_rate, 1.0)
        self.assertEqual(result.pass_rate, 1.0)

    async def test_insufficient_intermediate_step_fails_the_loop(self):
        case = EvidenceLoopEvalCase(
            "cuda",
            "In what year was the company that created CUDA founded?",
            "1993",
            [],
            ["Who created CUDA?", "When was NVIDIA founded?"],
        )

        async def gather(query, **kwargs):
            action = "refine_query" if query.startswith("Who") else "synthesize"
            return bundle(query, action=action, text="NVIDIA was founded in 1993.")

        result = await evaluate_evidence_loops([case], gather=gather)

        self.assertEqual(result.answer_coverage, 1.0)
        self.assertEqual(result.ready_rate, 0.0)
        self.assertEqual(result.pass_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
