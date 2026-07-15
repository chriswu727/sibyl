"""Release guard tests. No network."""
import tempfile
import unittest
from pathlib import Path

from scripts.check_release import validate_release


class TestReleaseValidation(unittest.TestCase):
    def _release_tree(self, version: str = "0.3.0"):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "sibyl").mkdir()
        (root / "sibyl" / "__init__.py").write_text(
            f'__version__ = "{version}"\n', encoding="utf-8"
        )
        (root / "CHANGELOG.md").write_text(
            f"# Changelog\n\n## {version} — 2026-07-14\n", encoding="utf-8"
        )
        return temporary, root

    def test_accepts_matching_tag_version_and_changelog(self):
        temporary, root = self._release_tree()
        with temporary:
            self.assertEqual(validate_release("v0.3.0", root), "0.3.0")

    def test_rejects_tag_that_does_not_match_package(self):
        temporary, root = self._release_tree()
        with temporary, self.assertRaisesRegex(ValueError, "must equal"):
            validate_release("v0.3.1", root)

    def test_rejects_missing_changelog_section(self):
        temporary, root = self._release_tree()
        with temporary:
            (root / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no section"):
                validate_release("v0.3.0", root)


if __name__ == "__main__":
    unittest.main()
