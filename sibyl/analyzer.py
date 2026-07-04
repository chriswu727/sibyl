"""Advanced analysis — sentiment, cross-referencing, source evaluation."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import litellm

from .config import Provider
from .scraper import WebPage

litellm.suppress_debug_info = True


@dataclass
class SourceAnalysis:
    url: str
    title: str
    sentiment: str      # positive, negative, neutral, mixed
    key_claims: List[str]
    credibility: str    # high, medium, low
    bias: str           # none, slight, strong + direction


@dataclass
class CrossAnalysis:
    consensus_points: List[str]     # what sources agree on
    disagreement_points: List[str]  # where sources conflict
    unique_insights: List[str]      # points only one source makes
    overall_sentiment: str          # positive, negative, mixed, neutral
    sentiment_breakdown: Dict[str, int]  # {"positive": 3, "negative": 2, ...}


async def analyze_sources(
    pages: List[WebPage],
    query: str,
    provider: Provider,
) -> CrossAnalysis:
    """Analyze multiple sources for sentiment, consensus, and disagreements."""

    # Build source summaries
    source_summaries = []
    for i, page in enumerate(pages[:8], 1):
        source_summaries.append(f"[Source {i}: {page.title}]\n{page.text[:1500]}")

    context = "\n---\n".join(source_summaries)

    # JSON mode with the sentiment keys pinned to exactly what
    # format_cross_analysis reads (positive/negative/neutral) — a differently
    # keyed breakdown would silently zero the sentiment line.
    prompt = f"""Analyze these {len(source_summaries)} sources about: {query}

{context}

Return json: an object with EXACTLY these keys:
{{
  "overall_sentiment": "positive | negative | mixed | neutral",
  "sentiment_breakdown": {{"positive": <int>, "negative": <int>, "neutral": <int>}},
  "consensus": ["point most sources agree on", "..."],
  "disagreements": ["point where sources conflict — Source X says A, Source Y says B", "..."],
  "unique": ["Source N: a notable point from only one source", "..."]
}}
Cite source numbers. Be specific."""

    kwargs = {
        "model": provider.model,
        "max_tokens": 1800,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if provider.api_key:
        kwargs["api_key"] = provider.api_key
    if provider.api_base:
        kwargs["api_base"] = provider.api_base
    # Structured extraction, not reasoning — disabling DeepSeek V4 thinking keeps
    # the output predictable and stops reasoning from eating the token budget.
    if "deepseek" in provider.model:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    # Defaults preserved from the legacy parser so a failure degrades gracefully
    # rather than crashing (json.loads on empty content would raise).
    overall_sentiment = "mixed"
    sentiment_breakdown = {"positive": 0, "negative": 0, "neutral": 0}
    consensus: List[str] = []
    disagreements: List[str] = []
    unique: List[str] = []

    data = None
    for attempt in range(3):
        try:
            response = await litellm.acompletion(**kwargs)
            text = (response.choices[0].message.content or "").strip()
            if not text:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                break
            data = json.loads(text)
            break
        except json.JSONDecodeError:
            break  # non-JSON: keep defaults
        except Exception:
            if attempt == 2:
                break
            await asyncio.sleep(1 * (attempt + 1))

    if isinstance(data, dict):
        s = str(data.get("overall_sentiment", "")).strip().lower()
        if s in ("positive", "negative", "mixed", "neutral"):
            overall_sentiment = s
        raw = data.get("sentiment_breakdown") or {}
        if isinstance(raw, dict):
            for k in ("positive", "negative", "neutral"):
                try:
                    sentiment_breakdown[k] = int(raw.get(k, 0))
                except (ValueError, TypeError):
                    pass
        consensus = [str(x).strip() for x in (data.get("consensus") or []) if str(x).strip()]
        disagreements = [str(x).strip() for x in (data.get("disagreements") or []) if str(x).strip()]
        unique = [str(x).strip() for x in (data.get("unique") or []) if str(x).strip()]

    return CrossAnalysis(
        consensus_points=consensus,
        disagreement_points=disagreements,
        unique_insights=unique,
        overall_sentiment=overall_sentiment,
        sentiment_breakdown=sentiment_breakdown,
    )


def format_cross_analysis(analysis: CrossAnalysis) -> str:
    """Format cross-analysis for inclusion in reports."""
    lines = [
        "## Source Cross-Analysis",
        "",
        f"**Overall Sentiment:** {analysis.overall_sentiment.upper()}",
        f"(Positive: {analysis.sentiment_breakdown.get('positive', 0)} | "
        f"Negative: {analysis.sentiment_breakdown.get('negative', 0)} | "
        f"Neutral: {analysis.sentiment_breakdown.get('neutral', 0)})",
        "",
    ]

    if analysis.consensus_points:
        lines.append("**What Sources Agree On:**")
        for p in analysis.consensus_points:
            lines.append(f"- {p}")
        lines.append("")

    if analysis.disagreement_points:
        lines.append("**Where Sources Disagree:**")
        for p in analysis.disagreement_points:
            lines.append(f"- {p}")
        lines.append("")

    if analysis.unique_insights:
        lines.append("**Unique Insights:**")
        for p in analysis.unique_insights:
            lines.append(f"- {p}")
        lines.append("")

    return "\n".join(lines)
