"""Advanced analysis tools — comparison tables, SWOT, trends, timeline."""
from __future__ import annotations

from typing import Dict, List, Optional

import litellm

from .config import Provider

litellm.suppress_debug_info = True


async def _llm(provider: Provider, prompt: str, max_tokens: int = 2000) -> str:
    kwargs = {
        "model": provider.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if provider.api_key:
        kwargs["api_key"] = provider.api_key
    if provider.api_base:
        kwargs["api_base"] = provider.api_base
    resp = await litellm.acompletion(**kwargs)
    return resp.choices[0].message.content.strip()


# ── 1. Structured Comparison ─────────────────────────────────────

async def generate_comparison(
    items: List[str],
    context: str,
    provider: Provider,
) -> str:
    """Generate a structured comparison table from research data."""
    prompt = f"""Create a detailed structured comparison of: {', '.join(items)}

Use the following research data as context:
{context[:4000]}

Output a MARKDOWN TABLE with these columns:
| Aspect | {' | '.join(items)} |

Include rows for:
- Current status/price/position
- Key strengths
- Key weaknesses
- Recent trend (up/down/flat)
- Outlook/prediction
- Risk level
- Key metric 1 (relevant to the topic)
- Key metric 2

After the table, add a brief "Bottom Line" paragraph recommending which is strongest and why.

Use specific data from the sources where available. Mark estimates with ~."""

    return await _llm(provider, prompt, max_tokens=2000)


# ── 2. Google Trends ─────────────────────────────────────────────

async def fetch_google_trends(
    keywords: List[str],
    timeframe: str = "today 12-m",
) -> Dict:
    """Fetch Google Trends data for keywords."""
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload(keywords[:5], timeframe=timeframe)

        # Interest over time
        interest = pytrends.interest_over_time()
        if interest.empty:
            return {"error": "No trend data available"}

        # Get latest values and trends
        result = {"keywords": {}, "timeframe": timeframe}
        for kw in keywords[:5]:
            if kw in interest.columns:
                values = interest[kw].tolist()
                current = values[-1] if values else 0
                avg = sum(values) / len(values) if values else 0
                peak = max(values) if values else 0
                # Trend direction
                recent = values[-4:] if len(values) >= 4 else values
                earlier = values[-8:-4] if len(values) >= 8 else values[:4]
                recent_avg = sum(recent) / len(recent) if recent else 0
                earlier_avg = sum(earlier) / len(earlier) if earlier else 0
                if recent_avg > earlier_avg * 1.1:
                    trend = "rising"
                elif recent_avg < earlier_avg * 0.9:
                    trend = "declining"
                else:
                    trend = "stable"

                result["keywords"][kw] = {
                    "current": current,
                    "average": round(avg, 1),
                    "peak": peak,
                    "trend": trend,
                }

        # Related queries
        try:
            related = pytrends.related_queries()
            for kw in keywords[:5]:
                if kw in related and related[kw].get("rising") is not None:
                    rising = related[kw]["rising"]
                    if rising is not None and not rising.empty:
                        result["keywords"][kw]["rising_queries"] = rising["query"].tolist()[:5]
        except Exception:
            pass

        return result
    except Exception as e:
        return {"error": str(e)}


def format_trends(data: Dict) -> str:
    """Format Google Trends data as readable text."""
    if "error" in data:
        return f"Google Trends error: {data['error']}"

    lines = ["## Google Trends Data", f"*Timeframe: {data.get('timeframe', 'unknown')}*", ""]

    for kw, info in data.get("keywords", {}).items():
        trend_emoji = {"rising": "^", "declining": "v", "stable": "~"}.get(info["trend"], "?")
        lines.append(f"**{kw}** [{trend_emoji} {info['trend']}]")
        lines.append(f"- Current interest: {info['current']}/100 (avg: {info['average']}, peak: {info['peak']})")
        if "rising_queries" in info:
            lines.append(f"- Rising related searches: {', '.join(info['rising_queries'][:3])}")
        lines.append("")

    return "\n".join(lines)


# ── 3. SWOT Analysis ─────────────────────────────────────────────

async def generate_swot(
    subject: str,
    context: str,
    provider: Provider,
) -> str:
    """Generate a SWOT analysis from research data."""
    prompt = f"""Create a thorough SWOT analysis for: {subject}

Use the following research data as context:
{context[:4000]}

Output in this EXACT format:

## SWOT Analysis: {subject}

### Strengths
- [strength 1 — with specific data/evidence]
- [strength 2]
- [strength 3]

### Weaknesses
- [weakness 1 — with specific data/evidence]
- [weakness 2]
- [weakness 3]

### Opportunities
- [opportunity 1 — with specific data/evidence]
- [opportunity 2]
- [opportunity 3]

### Threats
- [threat 1 — with specific data/evidence]
- [threat 2]
- [threat 3]

### Strategic Implications
One paragraph synthesizing the SWOT into actionable advice.

Be specific. Use real data from the sources. Avoid generic statements."""

    return await _llm(provider, prompt, max_tokens=2000)


# ── 4. Timeline Analysis ─────────────────────────────────────────

async def generate_timeline(
    topic: str,
    context: str,
    provider: Provider,
) -> str:
    """Extract and organize key events into a chronological timeline."""
    prompt = f"""Create a detailed chronological timeline for: {topic}

Use the following research data to extract specific events, dates, and milestones:
{context[:4000]}

Output in this format:

## Timeline: {topic}

| Date | Event | Impact |
|------|-------|--------|
| YYYY-MM or YYYY-MM-DD | Specific event description | Brief impact assessment |

Include:
- Past events that led to the current situation
- Recent developments (last 6-12 months)
- Upcoming scheduled events or expected milestones
- Key turning points

Order chronologically. Be specific with dates. Include 10-20 events.
After the table, add a "Trajectory" paragraph summarizing the direction of change."""

    return await _llm(provider, prompt, max_tokens=2500)
