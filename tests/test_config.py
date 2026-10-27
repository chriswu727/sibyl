"""Retrieval configuration never adopts model credentials."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sibyl.config import Config
from sibyl.mcp_server import _mcp_profile


class TestConfig(unittest.TestCase):
    def load(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sibyl.yaml"
            path.write_text(content, encoding="utf-8")
            return Config.from_yaml(str(path))

    def test_defaults(self):
        self.assertEqual(self.load(""), Config())
        self.assertEqual(Config().extractor, "bs4")

    def test_retrieval_settings(self):
        config = self.load("extractor: trafilatura\njs_render: true\n")
        self.assertEqual(config.extractor, "trafilatura")
        self.assertTrue(config.js_render)

    def test_legacy_model_settings_rejected_without_exposing_secret(self):
        with self.assertRaises(ValueError) as error:
            self.load("providers:\n  - api_key: secret-never-print\n")
        self.assertIn("MCP host", str(error.exception))
        self.assertNotIn("secret-never-print", str(error.exception))

    def test_invalid_settings(self):
        for value in (
            "[]",
            "js_render: yesplease",
            "extractor: unknown",
            "search_engine: 3",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(value)

    def test_model_environment_is_ignored(self):
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "fixture",
                "DEEPSEEK_API_KEY": "fixture",
                "ANTHROPIC_API_KEY": "fixture",
                "SIBYL_MODEL": "fixture",
                "SIBYL_API_BASE": "http://127.0.0.1:1",
            },
            clear=True,
        ):
            self.assertEqual(Config.from_env(), Config())
            self.assertEqual(_mcp_profile(), "agent")

    def test_profile_migration(self):
        for value in ("full", "report", "finance"):
            with mock.patch.dict(os.environ, {"SIBYL_MCP_PROFILE": value}):
                with self.assertRaisesRegex(RuntimeError, "removed"):
                    _mcp_profile()
        with mock.patch.dict(os.environ, {"SIBYL_MCP_PROFILE": "auto"}):
            self.assertEqual(_mcp_profile(), "agent")
