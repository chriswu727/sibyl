"""Advanced analysis — sentiment, cross-referencing, source evaluation."""
from __future__ import annotations

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

    prompt = f"""Analyze these {len(source_summaries)} sources about: {query}

{context}

Provide a structured analysis in this EXACT format:

OVERALL_SENTIMENT: [positive/negative/mixed/neutral]

SENTIMENT_COUNTS:
positive: [number]
negative: [number]
neutral: [number]

CONSENSUS (points most sources agree on):
- [point 1]
- [point 2]
- [point 3]

DISAGREEMENTS (where sources conflict):
- [point 1 — Source X says A, Source Y says B]
- [point 2]

UNIQUE_INSIGHTS (notable points from only one source):
- [Source N: insight]
- [Source N: insight]

Be specific and cite source numbers."""

    kwargs = {
        "model": provider.model,
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
    }
    if provider.api_key:
        kwargs["api_key"] = provider.api_key
    if provider.api_base:
        kwargs["api_base"] = provider.api_base

    response = await litellm.acompletion(**kwargs)
    text = response.choices[0].message.content.strip()

    # Parse the response
    overall_sentiment = "mixed"
    sentiment_breakdown = {"positive": 0, "negative": 0, "neutral": 0}
    consensus = []
    disagreements = []
    unique = []

    current_section = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("OVERALL_SENTIMENT:"):
            s = line.split(":", 1)[1].strip().lower()
            if s in ("positive", "negative", "mixed", "neutral"):
                overall_sentiment = s

        elif line.startswith("positive:"):
            try:
                sentiment_breakdown["positive"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("negative:"):
            try:
                sentiment_breakdown["negative"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("neutral:"):
            try:
                sentiment_breakdown["neutral"] = int(line.split(":")[1].strip())
            except ValueError:
                pass

        elif "CONSENSUS" in line.upper():
            current_section = "consensus"
        elif "DISAGREEMENT" in line.upper():
            current_section = "disagreements"
        elif "UNIQUE" in line.upper():
            current_section = "unique"
        elif line.startswith("-"):
            item = line.lstrip("- ").strip()
            if current_section == "consensus":
                consensus.append(item)
            elif current_section == "disagreements":
                disagreements.append(item)
            elif current_section == "unique":
                unique.append(item)

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
