"""Sibyl MCP Server — deep research tools for Claude Code and other MCP clients."""
from __future__ import annotations

import os
from dataclasses import replace
from importlib.util import find_spec
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .async_cache import AsyncSingleFlightTTL
from .config import Config
from .evidence import SourceBundle
from .ranking import RankingBackend
from .retrieval import gather_source_bundle, render_source_bundle

mcp = FastMCP(
    "sibyl",
    log_level="WARNING",
    instructions="""Sibyl gives you keyless web research. Two modes — pick based on who should do the reasoning:

RECOMMENDED — you (the host model) are the researcher. Sibyl retrieves; YOU reason:
  • gather_bundle(query) — structured keyless retrieval for agents and pipelines. Returns
    versioned sources/passages with stable IDs, hashes, timestamps, and diagnostics.
  • gather_sources(query) — the same retrieval rendered as readable [Source N] blocks
    for conversational use. Call either tool several times with focused sub-queries,
    cross-reference the evidence, and synthesize the answer YOURSELF — citing sources
    and saying "not found" rather than guessing when the sources don't contain it.
    Both default to local lexical ranking; optional ranker="flashrank" falls back
    explicitly in diagnostics, while ranker="none" preserves retrieval order.
  • quick_search(query) — raw search hits (title/url/snippet), no scraping.
  • read_url(url) — clean full text of one page.

If optional report tools are enabled, research(query, depth) runs the full
search→scrape→rank→synthesize→verify→report pipeline with the configured model.
For factual questions, prefer gather_bundle/gather_sources + your own synthesis.
""",
)

_config: Optional[Config] = None
_last_report: Optional[Any] = None
_bundle_cache = AsyncSingleFlightTTL[tuple, SourceBundle](
    ttl_seconds=30.0,
    max_entries=64,
)

_KEYLESS_TOOLS = {
    "gather_bundle",
    "gather_sources",
    "quick_search",
    "read_url",
}
_REPORT_TOOLS = {
    "analyze",
    "compare",
    "research",
    "save_report",
    "swot",
    "timeline",
}
_FINANCE_TOOLS = {"chart", "fetch_market_data", "trends"}


async def _cached_source_bundle(
    query: str,
    max_sources: int,
    chars_per_source: int,
    ranker: RankingBackend,
    render_thin_pages: bool,
) -> SourceBundle:
    key = (
        str(query or "").strip(),
        str(max_sources),
        str(chars_per_source),
        str(ranker or "").strip().lower(),
        bool(render_thin_pages),
    )

    async def retrieve() -> SourceBundle:
        return await gather_source_bundle(
            query,
            max_sources,
            chars_per_source,
            ranker=ranker,
            render_thin_pages=render_thin_pages,
        )

    return await _bundle_cache.get_or_create(
        key,
        retrieve,
        should_cache=lambda bundle: bundle.status in {"ok", "insufficient_evidence"},
    )


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


def _mcp_profile(config: Config) -> str:
    requested = os.environ.get("SIBYL_MCP_PROFILE", "auto").strip().lower()
    profiles = {"auto", "keyless", "report", "finance", "full"}
    if requested not in profiles:
        raise RuntimeError(
            "SIBYL_MCP_PROFILE must be one of: auto, keyless, report, finance, full"
        )
    if requested == "auto":
        report_installed = all(find_spec(name) for name in ("litellm", "fpdf"))
        return "report" if config.has_llm_credentials() and report_installed else "keyless"
    return requested


async def _configure_tool_profile() -> str:
    profile = _mcp_profile(_get_config())
    enabled = set(_KEYLESS_TOOLS)
    if profile in {"report", "full"}:
        missing = [name for name in ("litellm", "fpdf") if find_spec(name) is None]
        if missing:
            raise RuntimeError(
                "Report tools require 'pip install sibyl-research[report]' "
                f"(missing: {', '.join(missing)})."
            )
        _require_llm_config("Report tools")
        enabled.update(_REPORT_TOOLS)
    if profile in {"finance", "full"}:
        missing = [
            name
            for name in ("matplotlib", "pandas", "pytrends", "yfinance")
            if find_spec(name) is None
        ]
        if missing:
            raise RuntimeError(
                "Finance tools require 'pip install sibyl-research[finance]' "
                f"(missing: {', '.join(missing)})."
            )
        enabled.update(_FINANCE_TOOLS)
    for tool in await mcp.list_tools():
        if tool.name not in enabled:
            mcp.remove_tool(tool.name)
    return profile


def _require_llm_config(capability: str) -> Config:
    config = _get_config()
    if not config.has_llm_credentials():
        raise ToolError(
            f"{capability} requires an LLM provider key or a configured "
            "local/API-base backend. Use gather_bundle or gather_sources for "
            "keyless research."
        )
    return config


def _format_report(report: Any) -> str:
    """Format a research report as readable text."""
    if report.status != "ok":
        heading = "Insufficient evidence" if report.status == "insufficient_evidence" else "Research failed"
        lines = [
            f"# {heading}: {report.query}",
            "",
            report.error or "No report was produced.",
        ]
        if report.search_queries:
            lines.extend(["", f"*Search queries attempted: {', '.join(report.search_queries)}*"])
        return "\n".join(lines)

    lines = [
        f"# Research Report: {report.query}",
        f"*Generated at {report.timestamp.strftime('%Y-%m-%d %H:%M')} using {report.model_used}*",
        "",
        "## Summary",
        report.summary,
        "",
        "## Key Findings",
    ]
    fv = report.finding_verifications or []
    if report.key_findings and not fv:
        lines.append("> Claim verification was not performed for this report.")
    for i, finding in enumerate(report.key_findings, 1):
        verdict = fv[i - 1] if i - 1 < len(fv) else None
        tag = ""
        if verdict is not None:
            tag = f"  [verified: {verdict.confidence}]" if verdict.supported else "  (unverified)"
        lines.append(f"{i}. {finding}{tag}")

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
        evidence = src.supporting_snippet or src.snippet
        if evidence:
            lines.append(f"   {evidence}")

    if report.market_data_summary:
        lines.append("")
        lines.append(report.market_data_summary)

    lines.append("")
    lines.append(f"*Search queries used: {', '.join(report.search_queries)}*")
    return "\n".join(lines)


@mcp.tool()
async def gather_sources(
    query: str,
    max_sources: int = 10,
    chars_per_source: int = 7000,
    ranker: RankingBackend = "lexical",
    render_thin_pages: bool = False,
) -> str:
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
        max_sources: How many sources to return (default 10; bounded to 1-20)
        chars_per_source: Max characters of text per source (default 7000; bounded to 500-10000)
        ranker: lexical (default), flashrank (optional extra), or none (retrieval order)
        render_thin_pages: Send thin-page URLs to Jina Reader (default false)
    """
    bundle = await _cached_source_bundle(
        query, max_sources, chars_per_source, ranker, render_thin_pages
    )
    return render_source_bundle(bundle)


@mcp.tool(structured_output=True)
async def gather_bundle(
    query: str,
    max_sources: int = 10,
    chars_per_source: int = 7000,
    ranker: RankingBackend = "lexical",
    render_thin_pages: bool = False,
) -> SourceBundle:
    """Return a structured, keyless SourceBundle without synthesizing an answer.

    This is the programmatic form of gather_sources, intended for agents and
    pipelines that need stable evidence identifiers and retrieval provenance.
    Passage/source relevance defaults to the dependency-free lexical_v1 ranker.
    FlashRank is optional and falls back to lexical_v1 with an explicit diagnostic.
    Source quality remains null until a separate quality evaluator computes it.

    Args:
        query: One focused search query
        max_sources: How many sources to return (default 10; bounded to 1-20)
        chars_per_source: Max characters per evidence passage (default 7000; bounded to 500-10000)
        ranker: lexical (default), flashrank (optional extra), or none (retrieval order)
        render_thin_pages: Send thin-page URLs to Jina Reader (default false)
    """
    return await _cached_source_bundle(
        query, max_sources, chars_per_source, ranker, render_thin_pages
    )


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
        verify: At depth 2+, verify each finding and flag unsupported claims (skipped in fast mode)
    """
    global _last_report
    _last_report = None
    base_config = _require_llm_config("One-shot research")
    config = replace(base_config, providers=list(base_config.providers),
                     fast=fast, verify_claims=verify)
    from .researcher import Researcher

    researcher = Researcher(config)

    progress_lines = []
    def on_progress(msg: str):
        progress_lines.append(msg)

    try:
        report = await researcher.research(query, depth=depth, language=language, on_progress=on_progress)
        if report.status == "ok":
            _last_report = report
        if report.status == "failed":
            raise ToolError(report.error or "One-shot research failed.")
        return _format_report(report)
    except ToolError:
        raise
    except Exception as e:
        detail = str(e).strip().replace("\n", " ")[:200]
        raise ToolError(f"One-shot research failed: {detail}") from e


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
    config = _require_llm_config("Analysis")
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
    config = _require_llm_config("Comparison")
    provider = config.get_provider("analysis")

    item_list = [i.strip() for i in items.split(",") if i.strip()]
    full_query = f"Compare {', '.join(item_list)}"
    if query:
        full_query += f" — {query}"

    # Quick research for context
    from .researcher import Researcher

    researcher = Researcher(config)
    report = await researcher.research(full_query, depth=1)
    if report.status != "ok":
        return _format_report(report)

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
    config = _require_llm_config("SWOT analysis")
    provider = config.get_provider("analysis")

    from .researcher import Researcher

    researcher = Researcher(config)
    report = await researcher.research(f"SWOT analysis {subject}", depth=1)
    if report.status != "ok":
        return _format_report(report)

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
    config = _require_llm_config("Timeline generation")
    provider = config.get_provider("analysis")

    from .researcher import Researcher

    researcher = Researcher(config)
    report = await researcher.research(f"timeline of key events {topic}", depth=1)
    if report.status != "ok":
        return _format_report(report)

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
        raise ToolError("No research report to save. Run research() first.")

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
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Sibyl MCP server")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument(
        "--profile",
        choices=["auto", "keyless", "report", "finance", "full"],
    )
    args = parser.parse_args()
    if args.version:
        from . import __version__

        print(__version__)
        return
    if args.profile:
        os.environ["SIBYL_MCP_PROFILE"] = args.profile
    profile = asyncio.run(_configure_tool_profile())
    if args.list_tools:
        tools = asyncio.run(mcp.list_tools())
        print(f"profile: {profile}")
        for tool in tools:
            print(tool.name)
        return
    mcp.run()


if __name__ == "__main__":
    main()
