"""Config tests — provider auto-detection and the fast flag.

Run: python -m unittest discover tests    (no external deps)
"""
import os
import unittest
from unittest import mock

from sibyl.config import Config, Provider


class TestFromEnv(unittest.TestCase):
    def test_deepseek_sets_v4_flash_across_roles(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True):
            cfg = Config.from_env()
        models = {p.role: p.model for p in cfg.providers}
        # All roles on v4-flash (the measured-best model for these tasks)
        self.assertEqual(models["general"], "deepseek/deepseek-v4-flash")
        self.assertEqual(models["analysis"], "deepseek/deepseek-v4-flash")
        self.assertEqual(models["fast"], "deepseek/deepseek-v4-flash")
        for p in cfg.providers:
            self.assertEqual(p.api_key, "sk-test")

    def test_openai_fallback_single_provider(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-oai"}, clear=True):
            cfg = Config.from_env()
        self.assertEqual(len(cfg.providers), 1)
        self.assertEqual(cfg.providers[0].model, "gpt-4o-mini")

    def test_explicit_model_beats_env(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True):
            cfg = Config.from_env(model="gemini/gemini-2.5-flash", api_key="k")
        self.assertEqual(cfg.providers[0].model, "gemini/gemini-2.5-flash")


class TestGetProvider(unittest.TestCase):
    def test_role_lookup_and_fallback(self):
        cfg = Config(providers=[
            Provider(model="m-general", role="general"),
            Provider(model="m-analysis", role="analysis"),
        ])
        self.assertEqual(cfg.get_provider("analysis").model, "m-analysis")
        # Unknown role falls back to the first provider
        self.assertEqual(cfg.get_provider("nonexistent").model, "m-general")


class TestFastFlag(unittest.TestCase):
    def test_defaults_off(self):
        self.assertFalse(Config().fast)

    def test_from_yaml_reads_fast(self):
        import tempfile, textwrap
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(textwrap.dedent("""
                providers:
                  - model: deepseek/deepseek-v4-flash
                    api_key: sk-x
                fast: true
            """))
            path = f.name
        cfg = Config.from_yaml(path)
        self.assertTrue(cfg.fast)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
