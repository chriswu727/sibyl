"""Configuration for Sibyl — multi-provider LLM setup."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class Provider:
    model: str          # LiteLLM model string, e.g. "deepseek/deepseek-chat"
    api_key: str = ""
    api_base: str = ""
    role: str = "general"   # general, analysis, search, chinese, fast
    weight: float = 1.0     # routing weight


@dataclass
class Config:
    providers: List[Provider] = field(default_factory=list)
    search_engine: str = "duckduckgo"   # duckduckgo (free) or google/brave (needs key)
    max_sources: int = 10
    max_depth: int = 2      # research depth: 1=quick, 2=standard, 3=deep
    language: str = "auto"  # auto, en, zh

    def get_provider(self, role: str = "general") -> Provider:
        """Get the best provider for a given role."""
        for p in self.providers:
            if p.role == role:
                return p
        # Fallback to first provider
        return self.providers[0] if self.providers else Provider(model="deepseek/deepseek-chat")

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
        )

    @classmethod
    def from_env(cls, model: str = "", api_key: str = "", api_base: str = "") -> Config:
        """Create config from environment/CLI args."""
        import os
        if not model:
            # Auto-detect from env vars
            if os.environ.get("DEEPSEEK_API_KEY"):
                model = "deepseek/deepseek-chat"
                api_key = os.environ["DEEPSEEK_API_KEY"]
            elif os.environ.get("OPENAI_API_KEY"):
                model = "gpt-4o-mini"
                api_key = os.environ["OPENAI_API_KEY"]
            elif os.environ.get("ANTHROPIC_API_KEY"):
                model = "claude-sonnet-4-20250514"
                api_key = os.environ["ANTHROPIC_API_KEY"]
            else:
                model = "deepseek/deepseek-chat"

        providers = [Provider(model=model, api_key=api_key, api_base=api_base, role="general")]
        return cls(providers=providers)
