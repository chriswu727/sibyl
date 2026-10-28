"""Release guard tests. No network."""

import re
import tempfile
import tomllib
import unittest
from pathlib import Path

from scripts.check_release import validate_release


class TestRuntimeCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        )["project"]

    def test_minimum_python_excludes_unsupported_runtime(self):
        self.assertEqual(self.project["requires-python"], ">=3.11")

    def test_model_and_finance_dependencies_are_not_declared(self):
        extras = self.project["optional-dependencies"]
        self.assertNotIn("finance", extras)
        requirements = self.project["dependencies"] + [
            requirement for group in extras.values() for requirement in group
        ]
        for name in (
            "mcp",
            "litellm",
            "openai",
            "anthropic",
            "yfinance",
            "pytrends",
            "matplotlib",
            "pandas",
        ):
            self.assertFalse(
                any(item.lower().startswith(name) for item in requirements)
            )


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

    def test_current_repository_release_metadata(self):
        root = Path(__file__).resolve().parents[1]
        version = re.search(
            r'__version__ = "([^"]+)"',
            (root / "sibyl" / "__init__.py").read_text(encoding="utf-8"),
        )[1]
        self.assertEqual(validate_release(f"v{version}", root), version)

    def test_rejects_tag_that_does_not_match_package(self):
        temporary, root = self._release_tree()
        with temporary, self.assertRaisesRegex(ValueError, "must equal"):
            validate_release("v0.3.1", root)

    def test_publication_accepts_matching_pep440_beta(self):
        temporary, root = self._release_tree("0.4.0b1")
        with temporary:
            self.assertEqual(
                validate_release("v0.4.0b1", root, require_release_note=True),
                "0.4.0b1",
            )
            with self.assertRaisesRegex(ValueError, "must equal"):
                validate_release("v0.4.0", root, require_release_note=True)

    def test_publication_rejects_undated_development_changelog(self):
        temporary, root = self._release_tree()
        with temporary:
            (root / "CHANGELOG.md").write_text(
                "# Changelog\n\n## 0.3.0 — Unreleased\n", encoding="utf-8"
            )
            self.assertEqual(validate_release("v0.3.0", root), "0.3.0")
            with self.assertRaisesRegex(ValueError, "dated changelog"):
                validate_release("v0.3.0", root, require_release_note=True)

    def test_publication_accepts_dated_changelog(self):
        temporary, root = self._release_tree()
        with temporary:
            self.assertEqual(
                validate_release("v0.3.0", root, require_release_note=True), "0.3.0"
            )

    def test_publication_rejects_invalid_calendar_date(self):
        temporary, root = self._release_tree()
        with temporary:
            (root / "CHANGELOG.md").write_text(
                "## 0.3.0 — 2026-02-30\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                validate_release("v0.3.0", root, require_release_note=True)

    def test_rejects_missing_changelog_section(self):
        temporary, root = self._release_tree()
        with temporary:
            (root / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no section"):
                validate_release("v0.3.0", root)


if __name__ == "__main__":
    unittest.main()
