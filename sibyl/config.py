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
    role: str = "general"   # general, analysis, fast, synthesis, verify, compaction
    weight: float = 1.0     # routing weight


@dataclass(frozen=True)
class EffortTier:
    """A named effort level with hard resource caps and a latency target."""
    name: str
    depth: int
    max_queries: int
    max_urls: int
    synthesis_max_tokens: int
    latency_target_s: int


# quick is lean; standard/deep caps are non-restrictive ceilings (>= today's
# ~13 queries / 30 URLs) so they never reduce current behavior — only quick bites.
TIERS = {
    "quick":    EffortTier("quick",    1, 4,  10, 1200, 30),
    "standard": EffortTier("standard", 2, 15, 30, 1600, 90),
    "deep":     EffortTier("deep",     3, 20, 40, 2000, 240),
}
_DEPTH_TO_TIER = {1: "quick", 2: "standard", 3: "deep"}

# Role → ordered fallback chain. A missing role degrades to a keyless-identical
# provider, so per-role model routing is purely additive.
_ROLE_FALLBACKS = {
    "verify":     ["verify", "synthesis", "analysis", "general"],
    "synthesis":  ["synthesis", "analysis", "general"],
    "analysis":   ["analysis", "general"],
    "compaction": ["compaction", "fast", "general"],
    "fast":       ["fast", "general"],
    "general":    ["general"],
}


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
    # ── research-quality capabilities (borrowed from competitors) ──
    verify_claims: bool = True       # re-check each finding against its cited source text
    verify_drop_unsupported: bool = False  # destructive: drop unsupported findings (opt-in)
    js_render: bool = True           # on thin extraction, render via r.jina.ai (keyless)
    js_render_threshold: int = 500   # chars below which a 200 page is treated as JS-shell
    dedup: bool = True               # canonical-URL near-duplicate removal
    reranker: str = "lexical"        # "lexical" | "llm" | "flashrank" | "none"
    rerank_top_n: int = 12           # sources kept after ranked relevance scoring
    perspectives: bool = True        # perspective-guided query generation
    compact_sources: bool = False    # summarize each source before synthesis (weigh more)
    max_synth_sources: int = 12      # sources fed to synthesis
    rich_citations: bool = True      # carry supporting snippet per source
    tier: str = "standard"           # quick | standard | deep (used when depth unset)
    reflect_rounds: int = 0          # extra reflect→search→re-synthesize cycles (opt-in)

    def get_provider(self, role: str = "general") -> Provider:
        """Get the best provider for a role, degrading down its fallback chain."""
        for r in _ROLE_FALLBACKS.get(role, [role]):
            for p in self.providers:
                if p.role == r:
                    return p
        return self.providers[0] if self.providers else Provider(model="deepseek/deepseek-v4-flash")

    def resolve_tier(self, depth: int = 0) -> EffortTier:
        """Resolve the effort tier from an explicit depth, else the configured tier."""
        if depth:
            return TIERS.get(_DEPTH_TO_TIER.get(depth, self.tier), TIERS["standard"])
        return TIERS.get(self.tier, TIERS["standard"])

    def has_llm_credentials(self) -> bool:
        """Whether the configured one-shot LLM backend can be called."""
        import os

        if not self.providers:
            return False
        if any(p.api_key or p.api_base for p in self.providers):
            return True
        if any(p.model.startswith(("ollama/", "lm_studio/", "hosted_vllm/"))
               for p in self.providers):
            return True
        credential_vars = (
            "SIBYL_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "ZHIPUAI_API_KEY",
            "AZURE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
            "MISTRAL_API_KEY", "COHERE_API_KEY",
        )
        return any(os.environ.get(name) for name in credential_vars)

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
            verify_claims=data.get("verify_claims", True),
            verify_drop_unsupported=data.get("verify_drop_unsupported", False),
            js_render=data.get("js_render", True),
            js_render_threshold=data.get("js_render_threshold", 500),
            dedup=data.get("dedup", True),
            reranker=data.get("reranker", "lexical"),
            rerank_top_n=data.get("rerank_top_n", 12),
            perspectives=data.get("perspectives", True),
            compact_sources=data.get("compact_sources", False),
            max_synth_sources=data.get("max_synth_sources", 12),
            rich_citations=data.get("rich_citations", True),
            tier=data.get("tier", "standard"),
            reflect_rounds=data.get("reflect_rounds", 0),
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
