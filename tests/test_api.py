"""Public Python API tests. No network."""
import unittest
from unittest import mock

from sibyl import gather_bundle, gather_sources
from sibyl.evidence import BundleDiagnostics, SourceBundle


class TestPublicRetrievalApi(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bundle = SourceBundle(
            "1.6",
            "sb_test",
            "question",
            "insufficient_evidence",
            [],
            BundleDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 7000, 7000, 1),
            "No sources found.",
        )

    async def test_gather_bundle_is_a_top_level_api(self):
        gather = mock.AsyncMock(return_value=self.bundle)
        with mock.patch("sibyl.api.gather_source_bundle", new=gather):
            result = await gather_bundle("question", ranker="none")

        self.assertIs(result, self.bundle)
        gather.assert_awaited_once_with(
            "question",
            max_sources=10,
            chars_per_source=7000,
            ranker="none",
            render_thin_pages=False,
            client=None,
        )

    async def test_gather_sources_renders_the_same_bundle(self):
        with mock.patch(
            "sibyl.api.gather_source_bundle",
            new=mock.AsyncMock(return_value=self.bundle),
        ):
            result = await gather_sources("question")

        self.assertEqual(result, "No sources found.")


if __name__ == "__main__":
    unittest.main()
