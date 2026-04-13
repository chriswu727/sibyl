"""CLI interface for Sibyl research agent."""
import asyncio

import click
from rich.console import Console
from rich.markdown import Markdown

from .config import Config
from .researcher import Researcher

console = Console()


@click.command()
@click.argument("query")
@click.option("--depth", "-d", default=2, help="Research depth: 1=quick, 2=standard, 3=deep with predictions")
@click.option("--model", default="", help="LLM model (e.g. deepseek/deepseek-chat, gpt-4o)")
@click.option("--api-key", default="", help="API key (overrides env var)")
@click.option("--api-base", default="", help="Custom API base URL")
@click.option("--config", "-c", "config_file", default="", help="YAML config file")
@click.option("--max-sources", "-n", default=10, help="Max sources to read")
@click.option("--output", "-o", default=".", help="Output directory for reports")
@click.option("--pdf", is_flag=True, help="Generate PDF report")
@click.option("--md", is_flag=True, help="Generate Markdown report")
@click.option("--symbols", "-s", default="", help="Fetch market data for these symbols (e.g. NVDA,GOOGL,SPY)")
@click.option("--language", "-l", default="auto", help="Output language: auto, en, zh (Chinese), etc.")
def main(query, depth, model, api_key, api_base, config_file, max_sources, output, pdf, md, symbols, language):
    """Sibyl -- AI-powered deep research agent.

    Research any topic with web search + LLM analysis + market data.

    \b
    Examples:
        sibyl "Canadian housing market outlook 2026"
        sibyl "AI industry analysis" -d 3 --pdf --symbols NVDA,GOOGL,META
        sibyl "Bitcoin price prediction" -s BTC-USD,ETH-USD --pdf -o reports/
    """
    if config_file:
        cfg = Config.from_yaml(config_file)
    else:
        cfg = Config.from_env(model=model, api_key=api_key, api_base=api_base)

    cfg.max_depth = depth
    cfg.max_sources = max_sources

    console.print(f"\n[bold blue]Sibyl[/] researching: [cyan]{query}[/]")
    console.print(f"[dim]Depth: {depth} | Model: {cfg.providers[0].model if cfg.providers else 'auto'}[/]")
    if symbols:
        console.print(f"[dim]Market data: {symbols}[/]")
    console.print()

    result = asyncio.run(_run(cfg, query, depth, symbols, language))

    # Terminal output
    from .mcp_server import _format_report
    report_text = _format_report(result)
    console.print(Markdown(report_text))

    # PDF output
    if pdf:
        from .reporter import generate_pdf
        path = generate_pdf(result, output)
        console.print(f"\n[bold green]PDF saved:[/] {path}")

    # Markdown output
    if md:
        from .reporter import _report_to_markdown
        from pathlib import Path
        from datetime import datetime
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        safe_q = "".join(c if c.isalnum() or c in " -_" else "" for c in query)[:50].strip()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = out / f"sibyl_{safe_q}_{ts}.md"
        md_path.write_text(_report_to_markdown(result), encoding="utf-8")
        console.print(f"[bold green]Markdown saved:[/] {md_path}")


async def _run(cfg, query, depth, symbols="", language="auto"):
    researcher = Researcher(cfg)
    result = await researcher.research(
        query,
        depth=depth,
        language=language,
        on_progress=lambda msg: console.print(f"  [dim]{msg}[/]"),
    )

    # Fetch market data if symbols provided
    if symbols:
        from .data import fetch_multiple, format_data_summary, generate_chart
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        console.print(f"  [dim]Fetching market data for {', '.join(symbol_list)}...[/]")
        series = await fetch_multiple(symbol_list, "1y")
        if series:
            result.market_data_summary = format_data_summary(series)
            chart_path = generate_chart(series, f"{', '.join(s.name for s in series)} — 1 Year")
            result.charts.append(chart_path)
            console.print(f"  [dim]Chart generated: {chart_path}[/]")

    return result
