"""Sibyl MCP Server — deep research tools for Claude Code and other MCP clients."""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .config import Config, Provider
from .researcher import Researcher, ResearchReport

mcp = FastMCP(
    "sibyl",
    instructions="""Sibyl gives you keyless web research. Two modes — pick based on who should do the reasoning:

RECOMMENDED — you (the host model) are the researcher. Sibyl retrieves; YOU reason:
  • gather_sources(query) — keyless search + scrape + dedup; returns the top FULL-TEXT
    sources for a query WITHOUT writing an answer. Call it several times with different
    sub-queries, read the returned sources, cross-reference them, and synthesize the
    answer YOURSELF — citing sources and saying "not found" rather than guessing when
    the sources don't contain it. This uses no API key and gives the best quality,
    because YOUR reasoning is applied to real retrieved evidence.
  • quick_search(query) — raw search hits (title/url/snippet), no scraping.
  • read_url(url) — clean full text of one page.

ONE-SHOT — sibyl does the whole thing with its own model (needs sibyl's provider key):
  • research(query, depth) — full pipeline (search→scrape→rank→synthesize→verify→report)
    run by sibyl's configured LLM (e.g. DeepSeek). Findings are verified against sources.
    Use when you want a finished report in one call and accept sibyl's model/key does it.

Also: fetch_market_data(symbols), chart(symbols), compare/swot/timeline/trends, save_report().
For factual questions, prefer gather_sources + your own synthesis — it beats the one-shot
pipeline on hard questions and never fabricates. depth=1/2/3 = quick/standard/deep.
""",
)

_config: Optional[Config] = None
_last_report: Optional[ResearchReport] = None


def _get_config() -> Config:
    global _config
    if _config is None:
        config_path = os.environ.get("SIBYL_CONFIG")
        if config_path and os.path.exists(config_path):
            _config = Config.from_yaml(config_path)
        else:
            _config = Config.from_env(
                model=os.environ.get("SIBYL_MODEL", ""),
                api_key=os.environ.get("SIBYL_API_KEY", ""),
                api_base=os.environ.get("SIBYL_API_BASE", ""),
            )
    return _config


def _format_report(report: ResearchReport) -> str:
    """Format a research report as readable text."""
    lines = [
        f"# Research Report: {report.query}",
        f"*Generated at {report.timestamp.strftime('%Y-%m-%d %H:%M')} using {report.model_used}*",
        "",
        "## Summary",
        report.summary,
        "",
        "## Key Findings",
    ]
    for i, finding in enumerate(report.key_findings, 1):
        lines.append(f"{i}. {finding}")

    if report.analysis:
        lines.append("")
        lines.append("## Analysis")
        lines.append(report.analysis)

    if report.predictions:
        lines.append("")
        lines.append("## Predictions")
        lines.append(report.predictions)

    if report.confidence:
        lines.append("")
        lines.append(f"**Confidence:** {report.confidence}")

    if report.cross_analysis:
        lines.append("")
        lines.append(report.cross_analysis)

    lines.append("")
    lines.append(f"## Sources ({len(report.sources)})")
    for i, src in enumerate(report.sources, 1):
        lines.append(f"{i}. [{src.title}]({src.url})")
        if src.snippet:
            lines.append(f"   {src.snippet[:100]}")

    if report.market_data_summary:
        lines.append("")
        lines.append(report.market_data_summary)

    lines.append("")
    lines.append(f"*Search queries used: {', '.join(report.search_queries)}*")
    return "\n".join(lines)


@mcp.tool()
async def gather_sources(query: str, max_sources: int = 10, chars_per_source: int = 7000) -> str:
    """Keyless web retrieval: search + scrape + dedup, returning the top FULL-TEXT
    sources for a query WITHOUT writing an answer — so YOU (the calling model)
    read the evidence and reason over it yourself.

    Use this to research a question: call it several times with different focused
    sub-queries, read the numbered [Source N] blocks it returns, cross-reference
    them, then write the answer yourself with citations. If the sources don't
    contain the answer, gather more or say you don't know — do not guess. No API
    key required.

    Args:
        query: One focused search query (issue several calls for a multi-part question)
        max_sources: How many sources to return (default 8)
        chars_per_source: Max characters of text per source (default 3000)
    """
    import httpx
    from .search import search_web, fetch_wikipedia_extract, wikipedia_lookup
    from .scraper import scrape_urls, WebPage
    from .dedup import dedup_pages
    from .context import relevant_window

    async with httpx.AsyncClient(
        follow_redirects=True, timeout=12.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    ) as client:
        results = await search_web(query, "all", max_results=6, client=client, include_academic=True)
        seen, urls = set(), []
        for r in results:
            if r.url.startswith("http") and r.url not in seen:
                seen.add(r.url)
                urls.append(r.url)
        # Scrape DEEP (full page), then window each source to its answer region —
        # many facts live in a tail History table / infobox / deep paragraph that a
        # top-of-page slice never reaches.
        pages = await scrape_urls(urls[:max(max_sources * 2, 12)], max_chars=30000,
                                  client=client, js_render=True)
        good = [p for p in pages if p.text and len(p.text) > 150 and not p.error]
        scraped = {p.url for p in good}
        for r in results:  # supplement failed scrapes with their snippet — but only
            # substantive ones (title-only Google-News RSS stubs are noise, not evidence)
            if r.url not in scraped and r.snippet and len(r.snippet) > 120:
                good.append(WebPage(url=r.url, title=r.title, text=r.snippet))
        good = dedup_pages(good)
        substantive = [p for p in good if len(p.text) > 200]
        # Encyclopedic fallback: when general web search came up thin (obscure entity,
        # or engines rate-limited), pull the matching Wikipedia article(s) directly.
        if len(substantive) < 3:
            wiki_pages = await wikipedia_lookup(query, client=client, max_pages=2)
            if wiki_pages:
                good = dedup_pages(good + wiki_pages)
                substantive = [p for p in good if len(p.text) > 200]
        chosen = (substantive if len(substantive) >= 3 else good)[:max_sources]

        # Upgrade Wikipedia sources to clean full-text via the API — HTML scraping
        # truncates long articles before the infobox / tail sections that hold the fact.
        wiki_idx = [i for i, p in enumerate(chosen) if "wikipedia.org/wiki/" in p.url]
        if wiki_idx:
            extracts = await asyncio.gather(
                *[fetch_wikipedia_extract(chosen[i].url, client) for i in wiki_idx],
                return_exceptions=True,
            )
            for i, ex in zip(wiki_idx, extracts):
                if isinstance(ex, str) and len(ex) > len(chosen[i].text):
                    chosen[i] = WebPage(url=chosen[i].url, title=chosen[i].title, text=ex)

    if not chosen:
        return f"No sources found for query: {query!r}. Try a different phrasing."
    parts = []
    for i, p in enumerate(chosen, 1):
        window = relevant_window(query, p.text, width=chars_per_source)
        parts.append(f"[Source {i}: {p.title}]\nURL: {p.url}\n{window}\n")
    return (f"Retrieved {len(chosen)} sources for query {query!r}. Reason over these and "
            f"cite [Source N]; if the answer isn't here, gather more or say you don't know.\n\n"
            + "\n---\n".join(parts))


@mcp.tool()
async def research(query: str, depth: int = 2, language: str = "auto", fast: bool = False,
                   verify: bool = True) -> str:
    """Run a deep research cycle on any topic.

    Searches the web, reads multiple sources, and synthesizes a comprehensive
    report with findings, analysis, and optionally predictions.

    Args:
        query: The research question (e.g. "What's the outlook for Canadian housing market in 2026?")
        depth: Research depth. 1=quick (2 queries), 2=standard (4 queries), 3=deep with predictions (6 queries)
        language: Output language. "auto" (match query language), "en", "zh" (Chinese), or any language name
        fast: Skip the review/refine pass for ~20% faster results with slightly less polish
        verify: Verify each finding against its cited source and flag unsupported claims
    """
    config = _get_config()
    config.fast = fast
    config.verify_claims = verify
    researcher = Researcher(config)

    progress_lines = []
    def on_progress(msg: str):
        progress_lines.append(msg)

    global _last_report
    try:
        report = await researcher.research(query, depth=depth, language=language, on_progress=on_progress)
        _last_report = report
        return _format_report(report)
    except Exception as e:
        return f"Research failed: {str(e)[:200]}\n\nProgress so far:\n" + "\n".join(progress_lines)


@mcp.tool()
async def quick_search(query: str, max_results: int = 5) -> str:
    """Quick web search without deep analysis. Returns raw search results.

    Args:
        query: What to search for
        max_results: Maximum number of results (default 5)
    """
    from .search import search_web
    config = _get_config()
    results = await search_web(query, config.search_engine, max_results)

    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r.title}**")
        lines.append(f"   {r.url}")
        if r.snippet:
            lines.append(f"   {r.snippet[:150]}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def read_url(url: str) -> str:
    """Read and extract clean text content from a URL.

    Fetches the page, strips navigation/scripts/ads, and returns the main
    article or body text. Useful for reading a specific source in detail
    before or after running research().

    Returns the page title, URL, and up to 8000 characters of clean text.
    Handles retries, anti-bot protection, and Google Cache fallback.

    Args:
        url: The full URL to read (e.g. "https://www.reuters.com/article/...")
    """
    from .scraper import scrape_url
    page = await scrape_url(url, max_chars=8000)

    if page.error:
        return f"Error reading {url}: {page.error}"

    lines = [
        f"# {page.title}",
        f"URL: {page.url}",
        "",
        page.text,
    ]
    return "\n".join(lines)


@mcp.tool()
async def analyze(text: str, question: str) -> str:
    """Analyze provided text with a specific question using LLM.

    Takes any text (article, report, data, transcript) and answers a specific
    question about it using the configured LLM provider. Useful for follow-up
    analysis on content from read_url() or research().

    Returns a structured analysis with clear reasoning and evidence from the text.

    Args:
        text: The text content to analyze (article, report, data — up to 5000 chars)
        question: The specific question to answer about the text (e.g. "What are the main risks mentioned?" or "Summarize the key arguments for and against")
    """
    config = _get_config()
    provider = config.get_provider("analysis")

    import litellm
    litellm.suppress_debug_info = True

    kwargs = {
        "model": provider.model,
        "max_tokens": 1500,
        "messages": [
            {"role": "system", "content": "You are a research analyst. Provide clear, structured analysis."},
            {"role": "user", "content": f"Analyze the following text and answer this question: {question}\n\nText:\n{text[:5000]}"},
        ],
    }
    if provider.api_key:
        kwargs["api_key"] = provider.api_key
    if provider.api_base:
        kwargs["api_base"] = provider.api_base

    response = await litellm.acompletion(**kwargs)
    return response.choices[0].message.content.strip()


@mcp.tool()
async def compare(items: str, query: str = "") -> str:
    """Generate a structured comparison table for 2-5 items.

    Researches each item and produces a side-by-side markdown table with
    key metrics, strengths, weaknesses, and a bottom-line recommendation.

    Args:
        items: Comma-separated items to compare (e.g. "NVDA,AMD,INTC" or "React,Vue,Angular")
        query: Context for the comparison (e.g. "for AI/ML workloads" or "for a startup in 2026")
    """
    config = _get_config()
    provider = config.get_provider("analysis")

    item_list = [i.strip() for i in items.split(",") if i.strip()]
    full_query = f"Compare {', '.join(item_list)}"
    if query:
        full_query += f" — {query}"

    # Quick research for context
    researcher = Researcher(config)
    report = await researcher.research(full_query, depth=1)

    context = report.summary + "\n" + "\n".join(report.key_findings)
    from .tools import generate_comparison
    return await generate_comparison(item_list, context, provider)


@mcp.tool()
async def swot(subject: str) -> str:
    """Generate a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats).

    Researches the subject and produces a structured SWOT with specific data points.

    Args:
        subject: What to analyze (e.g. "Tesla", "Canadian housing market", "remote work trend")
    """
    config = _get_config()
    provider = config.get_provider("analysis")

    researcher = Researcher(config)
    report = await researcher.research(f"SWOT analysis {subject}", depth=1)

    context = report.summary + "\n" + "\n".join(report.key_findings)
    from .tools import generate_swot
    return await generate_swot(subject, context, provider)


@mcp.tool()
async def trends(keywords: str, timeframe: str = "today 12-m") -> str:
    """Get Google Trends data for keywords — real search interest over time.

    Shows current interest level, trend direction, peak, and rising related searches.

    Args:
        keywords: Comma-separated keywords (max 5, e.g. "ChatGPT,Claude,Gemini")
        timeframe: "today 1-m", "today 3-m", "today 12-m", "today 5-y" (default: 12 months)
    """
    from .tools import fetch_google_trends, format_trends
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()][:5]
    data = await fetch_google_trends(kw_list, timeframe)
    return format_trends(data)


@mcp.tool()
async def timeline(topic: str) -> str:
    """Generate a chronological timeline of key events for a topic.

    Researches the topic and extracts specific dates, events, and milestones
    into a structured timeline table.

    Args:
        topic: The topic to build a timeline for (e.g. "OpenAI history", "Canada immigration policy changes 2024-2026")
    """
    config = _get_config()
    provider = config.get_provider("analysis")

    researcher = Researcher(config)
    report = await researcher.research(f"timeline of key events {topic}", depth=1)

    context = report.summary + "\n" + "\n".join(report.key_findings)
    for src in report.sources:
        context += f"\n{src.snippet}"

    from .tools import generate_timeline
    return await generate_timeline(topic, context, provider)


@mcp.tool()
async def fetch_market_data(symbols: str, period: str = "1y") -> str:
    """Fetch real financial/stock/ETF data from Yahoo Finance.

    Returns current price, trend, moving averages, 52-week range, and % change.

    Args:
        symbols: Comma-separated ticker symbols (e.g. "AAPL,MSFT,SPY" or "XIU.TO,XRE.TO" for Canadian ETFs)
        period: Time period — "1mo", "3mo", "6mo", "1y", "2y", "5y" (default: 1y)

    Common symbols:
        US: SPY (S&P 500), QQQ (Nasdaq), DIA (Dow), VTI (total market)
        Canada: XIU.TO (TSX 60), XRE.TO (REIT), XIC.TO (composite)
        Crypto: BTC-USD, ETH-USD
        Commodities: GC=F (gold), CL=F (oil)
    """
    from .data import fetch_multiple, format_data_summary

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return "Error: provide at least one symbol"

    series = await fetch_multiple(symbol_list, period)
    if not series:
        return f"Could not fetch data for: {symbols}"

    summary = format_data_summary(series)

    # Attach to last report if available
    if _last_report is not None:
        _last_report.market_data_summary = summary

    return summary


@mcp.tool()
async def chart(symbols: str, period: str = "1y", title: str = "") -> str:
    """Generate a price history line chart for one or more financial symbols.

    Fetches historical price data from Yahoo Finance and creates a professional
    multi-line chart saved as PNG. The chart is automatically attached to the
    current research report and embedded in PDF output when save_report() is called.

    Supports stocks, ETFs, indices, crypto, and commodities.

    Args:
        symbols: Comma-separated ticker symbols (e.g. "NVDA,AMD,INTC" for stocks, "BTC-USD,ETH-USD" for crypto, "GC=F" for gold)
        period: Historical time period — "1mo", "3mo", "6mo", "1y", "2y", "5y" (default: "1y")
        title: Custom chart title. If empty, auto-generated from symbol names and period.
    """
    from .data import fetch_multiple, generate_chart

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    series = await fetch_multiple(symbol_list, period)
    if not series:
        return f"Could not fetch data for: {symbols}"

    chart_title = title or f"{', '.join(s.name for s in series)} — {period}"
    path = generate_chart(series, chart_title)

    # Attach to last report if available
    if _last_report is not None:
        _last_report.charts.append(path)

    return f"Chart saved: {path}"


@mcp.tool()
async def save_report(format: str = "both", output_dir: str = ".") -> str:
    """Save the last research report as PDF and/or Markdown file.

    Call this after research() to save the report.

    Args:
        format: "pdf", "md", or "both" (default: both)
        output_dir: Directory to save files (default: current directory)
    """
    if _last_report is None:
        return "No research report to save. Run research() first."

    from .reporter import generate_pdf, _report_to_markdown
    from pathlib import Path
    from datetime import datetime

    results = []

    if format in ("pdf", "both"):
        try:
            path = generate_pdf(_last_report, output_dir)
            results.append(f"PDF saved: {path}")
        except Exception as e:
            results.append(f"PDF failed: {e}")

    if format in ("md", "both"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        safe_q = "".join(c if c.isalnum() or c in " -_" else "" for c in _last_report.query)[:50].strip()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = out / f"sibyl_{safe_q}_{ts}.md"
        md_path.write_text(_report_to_markdown(_last_report), encoding="utf-8")
        results.append(f"Markdown saved: {md_path}")

    return "\n".join(results)


def main():
    """Entry point for sibyl-mcp command."""
    mcp.run()


if __name__ == "__main__":
    main()
