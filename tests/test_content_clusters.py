"""Deterministic source-content clustering tests."""
import unittest

from sibyl.content_clusters import cluster_content


def _article(prefix: str, middle: str, suffix: str) -> str:
    body = " ".join(
        f"paragraph {index} reports measured value {index * 7} with supporting detail"
        for index in range(30)
    )
    return f"{prefix} {body} {middle} {suffix}"


class TestContentClusters(unittest.TestCase):
    def test_clusters_syndicated_articles_with_different_wrappers(self):
        original = _article(
            "Original publication header",
            "The quoted spokesperson confirmed the result",
            "Original copyright footer",
        )
        syndicated = _article(
            "Partner site navigation and republished header",
            "The quoted spokesperson confirmed the result",
            "Partner newsletter and legal footer",
        )

        result = cluster_content([original, syndicated])

        self.assertEqual(result.cluster_count, 1)
        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(result.cluster_ids[0], result.cluster_ids[1])

    def test_keeps_different_reporting_in_separate_clusters(self):
        first = " ".join(
            f"alpha investigation source interview {index} finding {index * 3}"
            for index in range(35)
        )
        second = " ".join(
            f"beta laboratory experiment sample {index} outcome {index * 11}"
            for index in range(35)
        )

        result = cluster_content([first, second])

        self.assertEqual(result.cluster_count, 2)
        self.assertEqual(result.duplicate_count, 0)
        self.assertNotEqual(result.cluster_ids[0], result.cluster_ids[1])

    def test_short_snippets_are_not_fuzzy_clustered(self):
        first = "shared short search snippet first " * 8
        second = "shared short search snippet second " * 8

        result = cluster_content([first, second])

        self.assertEqual(result.cluster_count, 2)
        self.assertEqual(result.duplicate_count, 0)

    def test_exact_short_snippets_are_duplicates(self):
        snippet = "identical short search result summary"

        result = cluster_content([snippet, snippet])

        self.assertEqual(result.cluster_count, 1)
        self.assertEqual(result.duplicate_count, 1)

    def test_cluster_ids_do_not_depend_on_input_order(self):
        first = _article("First header", "shared article body", "First footer")
        second = _article("Second header", "shared article body", "Second footer")

        forward = cluster_content([first, second]).cluster_ids
        reverse = cluster_content([second, first]).cluster_ids

        self.assertEqual(forward[0], reverse[1])
        self.assertEqual(forward[1], reverse[0])


if __name__ == "__main__":
    unittest.main()
