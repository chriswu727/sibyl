"""Configuration for Sibyl — multi-provider LLM setup."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class Provider:
    model: str          # LiteLLM model string, e.g. "deepseek/deepseek-v4-flash"
    api_key: str = ""
    api_base: str = ""
    role: str = "general"   # general, analysis, search, chinese, fast
    weight: float = 1.0     # routing weight


@dataclass
class Config:
    providers: List[Provider] = field(default_factory=list)
    search_engine: str = "all"   # all (ddg+news+reddit+wiki), or duckduckgo only
    max_sources: int = 15
    max_depth: int = 2      # research depth: 1=quick, 2=standard, 3=deep
    language: str = "auto"  # auto, en, zh
    fast: bool = False      # skip the review/refine pass for ~20% faster runs
    extractor: str = "bs4"  # HTML content extractor: "bs4" (default) or "trafilatura"
    jina_fallback: bool = False  # on scrape block/403, retry via r.jina.ai (needs JINA_API_KEY)

    def get_provider(self, role: str = "general") -> Provider:
        """Get the best provider for a given role."""
        for p in self.providers:
            if p.role == role:
                return p
        # Fallback to first provider
        return self.providers[0] if self.providers else Provider(model="deepseek/deepseek-v4-flash")

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path) as f:
            data = yaml.safe_load(f)
        providers = []
        for p in data.get("providers", []):
            providers.append(Provider(**p))
        return cls(
            providers=providers,
            search_engine=data.get("search_engine", "duckduckgo"),
            max_sources=data.get("max_sources", 10),
            max_depth=data.get("max_depth", 2),
            language=data.get("language", "auto"),
            fast=data.get("fast", False),
            extractor=data.get("extractor", "bs4"),
            jina_fallback=data.get("jina_fallback", False),
        )

    @classmethod
    def from_env(cls, model: str = "", api_key: str = "", api_base: str = "") -> Config:
        """Create config from environment/CLI args."""
        import os

        # DeepSeek V4 (latest). Head-to-head on sibyl's long-form structured
        # tasks: v4-flash is faster (~30s vs ~41s), lower-variance, and actually
        # *completes* its output at a 2000-token budget, whereas v4-pro reasons
        # more and truncates mid-section (and occasionally spends the whole
        # budget on reasoning, returning empty). Comparable insight. So flash is
        # used across the board — mechanical steps disable thinking at call time,
        # analytical steps keep it on.
        if not model and os.environ.get("DEEPSEEK_API_KEY"):
            key = os.environ["DEEPSEEK_API_KEY"]
            # Default to the latest V4 tier, but honor a DEEPSEEK_MODEL override so
            # a rename (DeepSeek already went deepseek-chat -> v4-flash) doesn't
            # require a code change — just set DEEPSEEK_MODEL=deepseek-<new>.
            ds_model = os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-v4-flash")
            if "/" not in ds_model:
                ds_model = f"deepseek/{ds_model}"
            return cls(providers=[
                Provider(model=ds_model, api_key=key, role="general"),
                Provider(model=ds_model, api_key=key, role="fast"),
                Provider(model=ds_model, api_key=key, role="analysis"),
            ])

        if not model:
            # Auto-detect from env vars (check all providers)
            env_providers = [
                ("OPENAI_API_KEY", "gpt-4o-mini", ""),
                ("ANTHROPIC_API_KEY", "claude-sonnet-4-20250514", ""),
                ("GEMINI_API_KEY", "gemini/gemini-2.5-flash", ""),
                ("ZHIPUAI_API_KEY", "openai/glm-4-flash", "https://open.bigmodel.cn/api/paas/v4"),
            ]
            for env_key, env_model, env_base in env_providers:
                if os.environ.get(env_key):
                    model = env_model
                    api_key = os.environ[env_key]
                    api_base = env_base or api_base
                    break
            else:
                model = "deepseek/deepseek-v4-pro"

        providers = [Provider(model=model, api_key=api_key, api_base=api_base, role="general")]
        return cls(providers=providers)
