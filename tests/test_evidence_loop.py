"""Bounded evidence-loop tests. No network."""
import unittest
from unittest import mock

from sibyl.evidence import BundleDiagnostics, SourceBundle
from sibyl.evidence_loop import EvidenceLoopManager


def bundle(query: str, action: str = "synthesize") -> SourceBundle:
    status = "ok" if action == "synthesize" else "insufficient_evidence"
    sufficiency = "sufficient" if action == "synthesize" else "insufficient"
    return SourceBundle(
        "1.6",
        f"sb_{query.replace(' ', '_')}",
        query,
        status,
        [],
        BundleDiagnostics(
            2,
            2,
            2,
            2,
            0,
            0,
            0,
            2,
            10,
            10,
            7000,
            7000,
            1,
            evidence_sufficiency=sufficiency,
            recommended_action=action,
            query_complexity="single_step",
        ),
    )


class TestEvidenceLoopManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = EvidenceLoopManager()

    async def test_atomic_question_finishes_in_one_call(self):
        gather = mock.AsyncMock(return_value=bundle("What is Rust?"))

        result = await self.manager.start(
            "What is Rust?",
            max_steps=3,
            max_sources=10,
            chars_per_source=7000,
            ranker="lexical",
            render_thin_pages=False,
            gather=gather,
        )

        self.assertEqual(result.status, "ready")
        self.assertEqual(result.next_action, "synthesize")
        self.assertEqual(result.supporting_step_ids, ["E1"])
        self.assertEqual(result.steps[0].bundle_id, result.current_step.bundle.bundle_id)
        self.assertEqual(result.diagnostics.remaining_steps, 2)
        gather.assert_awaited_once_with(
            "What is Rust?", 10, 7000, "lexical", False
        )

    async def test_complex_question_requires_host_supplied_atomic_steps(self):
        question = "In what year was the company that created CUDA founded?"
        gather = mock.AsyncMock(return_value=bundle("Who created CUDA?"))

        started = await self.manager.start(
            question,
            max_steps=3,
            max_sources=10,
            chars_per_source=7000,
            ranker="lexical",
            render_thin_pages=False,
            gather=gather,
        )
        advanced = await self.manager.advance(
            started.loop_id,
            query="Who created CUDA?",
            finish=False,
            supporting_step_ids=None,
            gather=gather,
        )
        finished = await self.manager.advance(
            started.loop_id,
            query="",
            finish=True,
            supporting_step_ids=["E1"],
            gather=gather,
        )

        self.assertEqual(started.status, "active")
        self.assertEqual(started.next_action, "decompose_query")
        self.assertEqual(started.diagnostics.retrieval_calls, 0)
        self.assertEqual(advanced.next_action, "continue_or_finalize")
        self.assertEqual(advanced.steps[0].step_id, "E1")
        self.assertIsNotNone(advanced.current_step)
        self.assertEqual(finished.status, "ready")
        self.assertIsNone(finished.current_step)
        self.assertEqual(finished.supporting_step_ids, ["E1"])
        gather.assert_awaited_once()

    async def test_compound_and_duplicate_followups_are_rejected(self):
        question = "In what year was the company that created CUDA founded?"
        gather = mock.AsyncMock(return_value=bundle("Who created CUDA?"))
        started = await self.manager.start(
            question,
            max_steps=3,
            max_sources=10,
            chars_per_source=7000,
            ranker="lexical",
            render_thin_pages=False,
            gather=gather,
        )

        compound = await self.manager.advance(
            started.loop_id,
            query=question,
            finish=False,
            supporting_step_ids=None,
            gather=gather,
        )
        first = await self.manager.advance(
            started.loop_id,
            query="Who created CUDA?",
            finish=False,
            supporting_step_ids=None,
            gather=gather,
        )
        duplicate = await self.manager.advance(
            started.loop_id,
            query="who created cuda?",
            finish=False,
            supporting_step_ids=None,
            gather=gather,
        )

        self.assertIn("must be atomic", compound.error)
        self.assertEqual(compound.diagnostics.retrieval_calls, 0)
        self.assertEqual(first.diagnostics.retrieval_calls, 1)
        self.assertIn("must not repeat", duplicate.error)
        self.assertEqual(duplicate.diagnostics.retrieval_calls, 1)
        gather.assert_awaited_once()

    async def test_finish_rejects_steps_that_are_not_synthesis_ready(self):
        gather = mock.AsyncMock(return_value=bundle("Rust evidence", "refine_query"))
        started = await self.manager.start(
            "Rust evidence",
            max_steps=1,
            max_sources=10,
            chars_per_source=7000,
            ranker="lexical",
            render_thin_pages=False,
            gather=gather,
        )
        finished = await self.manager.advance(
            started.loop_id,
            query="",
            finish=True,
            supporting_step_ids=["E1"],
            gather=gather,
        )

        self.assertEqual(started.status, "budget_exhausted")
        self.assertEqual(finished.status, "budget_exhausted")
        self.assertIn("not synthesis-ready", finished.error)

    async def test_unknown_loop_and_invalid_budget_are_explicit(self):
        gather = mock.AsyncMock()
        invalid = await self.manager.start(
            "question",
            max_steps=5,
            max_sources=10,
            chars_per_source=7000,
            ranker="lexical",
            render_thin_pages=False,
            gather=gather,
        )
        missing = await self.manager.advance(
            "el_missing",
            query="atomic query",
            finish=False,
            supporting_step_ids=None,
            gather=gather,
        )

        self.assertEqual(invalid.status, "invalid_request")
        self.assertEqual(invalid.next_action, "revise_request")
        self.assertIn("between 1 and 4", invalid.error)
        self.assertEqual(missing.status, "invalid_request")
        self.assertIn("Unknown or expired", missing.error)
        gather.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
