"""Offline documentation checker contracts."""

import tempfile
import unittest
from pathlib import Path

from scripts.check_docs import anchors, check_document, main


class TestDocumentation(unittest.TestCase):
    def test_repository_documentation_links(self):
        self.assertEqual(main(), 0)

    def test_heading_slugs_duplicates_and_fences(self):
        text = "# A `tool`\n## A tool\n```md\n# Not a heading\n```\n## Usage and limits\n"
        self.assertEqual(anchors(text), {"a-tool", "a-tool-1", "usage-and-limits"})

    def test_missing_target_and_anchor_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "README.md"
            page.write_text("# Title\n[Missing](missing.md)\n[Anchor](#absent)\n[OK](#title)\n")
            errors = check_document(page, root)
            self.assertEqual(len(errors), 2)
            self.assertIn("missing local target", errors[0])
            self.assertIn("missing anchor", errors[1])

    def test_remote_main_links_are_checked_but_history_and_code_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "README.md"
            page.write_text("# Title\n[Current](https://github.com/chriswu727/sibyl/blob/main/README.md#title)\n"
                            "[History](https://github.com/chriswu727/sibyl/blob/old/old.md)\n"
                            "```md\n[Example](missing.md)\n```\n")
            self.assertEqual(check_document(page, root), [])
