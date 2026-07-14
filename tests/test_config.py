"""Config tests — provider auto-detection and the fast flag.

Run: python -m unittest discover tests    (no external deps)
"""
import os
import unittest
from unittest import mock

from sibyl.config import Config, Provider, TIERS


class TestTiersAndRoles(unittest.TestCase):
    def test_new_flag_defaults(self):
        c = Config()
        self.assertTrue(c.verify_claims)
        self.assertFalse(c.verify_drop_unsupported)
        self.assertTrue(c.js_render)
        self.assertTrue(c.dedup)
        self.assertEqual(c.reranker, "llm")
        self.assertTrue(c.perspectives)
        self.assertFalse(c.compact_sources)
        self.assertTrue(c.rich_citations)
        self.assertEqual(c.tier, "standard")
        self.assertEqual(c.reflect_rounds, 0)

    def test_standard_tier_is_todays_behavior(self):
        # standard must reproduce the current 1600-token synthesis budget
        self.assertEqual(TIERS["standard"].synthesis_max_tokens, 1600)
        self.assertEqual(TIERS["standard"].depth, 2)

    def test_resolve_tier(self):
        c = Config()
        self.assertEqual(c.resolve_tier(1).name, "quick")
        self.assertEqual(c.resolve_tier(2).name, "standard")
        self.assertEqual(c.resolve_tier(3).name, "deep")
        self.assertEqual(c.resolve_tier(0).name, "standard")  # falls to configured tier

    def test_role_fallback_chain(self):
        # verify/synthesis/compaction all degrade to an existing provider
        c = Config(providers=[Provider(model="m-analysis", role="analysis"),
                              Provider(model="m-general", role="general")])
        self.assertEqual(c.get_provider("verify").model, "m-analysis")     # verify->synthesis->analysis
        self.assertEqual(c.get_provider("synthesis").model, "m-analysis")  # synthesis->analysis
        self.assertEqual(c.get_provider("compaction").model, "m-general")  # compaction->fast->general

    def test_synthesis_uses_strong_provider_when_present(self):
        c = Config(providers=[Provider(model="strong", role="synthesis"),
                              Provider(model="cheap", role="general")])
        self.assertEqual(c.get_provider("synthesis").model, "strong")
        self.assertEqual(c.get_provider("compaction").model, "cheap")


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

    def test_deepseek_model_override(self):
        with mock.patch.dict(os.environ,
                             {"DEEPSEEK_API_KEY": "sk-test", "DEEPSEEK_MODEL": "deepseek-v5-flash"},
                             clear=True):
            cfg = Config.from_env()
        self.assertTrue(all(p.model == "deepseek/deepseek-v5-flash" for p in cfg.providers))

    def test_deepseek_model_override_keeps_explicit_prefix(self):
        with mock.patch.dict(os.environ,
                             {"DEEPSEEK_API_KEY": "sk-test", "DEEPSEEK_MODEL": "deepseek/custom-x"},
                             clear=True):
            cfg = Config.from_env()
        self.assertEqual(cfg.providers[0].model, "deepseek/custom-x")

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


class TestLlmCredentials(unittest.TestCase):
    def test_missing_credentials(self):
        cfg = Config(providers=[Provider(model="deepseek/deepseek-v4-flash")])
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(cfg.has_llm_credentials())

    def test_explicit_key_or_api_base(self):
        self.assertTrue(Config(providers=[Provider(model="x", api_key="k")]).has_llm_credentials())
        self.assertTrue(Config(providers=[Provider(model="x", api_base="http://localhost:8000")]).has_llm_credentials())

    def test_provider_environment_key(self):
        cfg = Config(providers=[Provider(model="gpt-4o-mini")])
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            self.assertTrue(cfg.has_llm_credentials())


class TestExtractor(unittest.TestCase):
    def test_defaults_bs4(self):
        self.assertEqual(Config().extractor, "bs4")

    def test_from_yaml_reads_extractor(self):
        import tempfile, textwrap
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(textwrap.dedent("""
                providers:
                  - model: deepseek/deepseek-v4-flash
                    api_key: sk-x
                extractor: trafilatura
            """))
            path = f.name
        cfg = Config.from_yaml(path)
        self.assertEqual(cfg.extractor, "trafilatura")
        os.unlink(path)


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
