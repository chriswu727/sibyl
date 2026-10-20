"""Release guard tests. No network."""
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from scripts.check_release import validate_release


class TestRuntimeCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]

    def test_minimum_python_excludes_unsupported_runtime(self):
        self.assertEqual(self.project["requires-python"], ">=3.11")

    def test_image_using_extras_enforce_pillow_security_floor(self):
        for extra in ("report", "finance", "all"):
            with self.subTest(extra=extra):
                self.assertIn("pillow>=13.0.0", self.project["optional-dependencies"][extra])

    def test_core_does_not_pull_in_pillow(self):
        self.assertFalse(any(item.lower().startswith("pillow") for item in self.project["dependencies"]))


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
        (root / "server.json").write_text(
            json.dumps(
                {
                    "name": "io.github.chriswu727/sibyl",
                    "version": version,
                    "packages": [{"version": version}],
                }
            ),
            encoding="utf-8",
        )
        (root / "README.md").write_text(
            "<!-- mcp-name: io.github.chriswu727/sibyl -->\n",
            encoding="utf-8",
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

    def test_rejects_mismatched_registry_version(self):
        temporary, root = self._release_tree()
        with temporary:
            (root / "server.json").write_text(
                json.dumps(
                    {
                        "name": "io.github.chriswu727/sibyl",
                        "version": "0.2.0",
                        "packages": [{"version": "0.2.0"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "server.json"):
                validate_release("v0.3.0", root)


if __name__ == "__main__":
    unittest.main()
