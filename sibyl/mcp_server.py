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
    instructions="""Sibyl is an AI-powered deep research agent. Use these tools to research topics,
analyze markets, predict trends, and generate comprehensive reports.

Workflow:
1. research(query) — Full research: search web → scrape → analyze → report
2. fetch_market_data(symbols) — Pull real financial data (prices, trends, moving averages)
3. chart(symbols) — Generate price trend charts as PNG
4. analyze(text, question) — Analyze text with LLM
5. save_report(format) — Save last report as PDF/Markdown

For financial/market research, combine research() + fetch_market_data() for
data-backed analysis. Use chart() to visualize trends.

Tips:
- depth=1 for quick answers, depth=2 for standard, depth=3 for deep with predictions
- Common symbols: SPY, QQQ (US), XIU.TO, XRE.TO (Canada), BTC-USD (crypto), GC=F (gold)
- Always save reports with save_report() when done
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

    lines.append("")
    lines.append(f"## Sources ({len(report.sources)})")
    for i, src in enumerate(report.sources, 1):
        lines.append(f"{i}. [{src.title}]({src.url})")
        if src.snippet:
            lines.append(f"   {src.snippet[:100]}")

    lines.append("")
    lines.append(f"*Search queries used: {', '.join(report.search_queries)}*")
    return "\n".join(lines)


@mcp.tool()
async def research(query: str, depth: int = 2) -> str:
    """Run a deep research cycle on any topic.

    Searches the web, reads multiple sources, and synthesizes a comprehensive
    report with findings, analysis, and optionally predictions.

    Args:
        query: The research question (e.g. "What's the outlook for Canadian housing market in 2026?")
        depth: Research depth. 1=quick (2 queries), 2=standard (4 queries), 3=deep with predictions (6 queries)
    """
    config = _get_config()
    researcher = Researcher(config)

    progress_lines = []
    def on_progress(msg: str):
        progress_lines.append(msg)

    global _last_report
    report = await researcher.research(query, depth=depth, on_progress=on_progress)
    _last_report = report
    return _format_report(report)


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

    Args:
        url: The URL to read
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

    Args:
        text: The text content to analyze
        question: What to analyze about the text
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

    return format_data_summary(series)


@mcp.tool()
async def chart(symbols: str, period: str = "1y", title: str = "") -> str:
    """Generate a price chart for one or more symbols and save as PNG.

    Args:
        symbols: Comma-separated ticker symbols (e.g. "AAPL,MSFT")
        period: Time period (default: 1y)
        title: Chart title (auto-generated if empty)
    """
    from .data import fetch_multiple, generate_chart

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    series = await fetch_multiple(symbol_list, period)
    if not series:
        return f"Could not fetch data for: {symbols}"

    chart_title = title or f"{', '.join(s.name for s in series)} — {period}"
    path = generate_chart(series, chart_title)
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
