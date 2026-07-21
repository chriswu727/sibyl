"""CLI interface for Sibyl retrieval and optional report generation."""
import asyncio
import json
import sys
from importlib.util import find_spec

import click

from .api import gather_bundle
from .config import Config


@click.command(name="research")
@click.argument("query")
@click.option("--depth", "-d", default=2, help="Research depth: 1=quick, 2=standard, 3=deep with predictions")
@click.option("--model", default="", help="LLM model (e.g. deepseek/deepseek-v4-flash, gpt-4o)")
@click.option("--api-key", default="", help="API key (overrides env var)")
@click.option("--api-base", default="", help="Custom API base URL")
@click.option("--config", "-c", "config_file", default="", help="YAML config file")
@click.option("--max-sources", "-n", default=10, help="Max sources to read")
@click.option("--output", "-o", default=".", help="Output directory for reports")
@click.option("--pdf", is_flag=True, help="Generate PDF report")
@click.option("--md", is_flag=True, help="Generate Markdown report")
@click.option("--symbols", "-s", default="", help="Fetch market data for these symbols (e.g. NVDA,GOOGL,SPY)")
@click.option("--language", "-l", default="auto", help="Output language: auto, en, zh (Chinese), etc.")
@click.option("--fast", is_flag=True, help="Skip the review/refine pass (~20% faster, slightly less polished)")
@click.option("--extractor", type=click.Choice(["bs4", "trafilatura"]), default="bs4", help="HTML content extractor")
@click.option("--jina-fallback", is_flag=True, help="On a blocked scrape, retry via r.jina.ai (set JINA_API_KEY)")
@click.option("--js-render/--no-js-render", default=True, help="Render thin/JS pages via r.jina.ai (keyless)")
@click.option("--effort", "-e", type=click.Choice(["quick", "standard", "deep"]), default=None, help="Effort tier (overrides --depth)")
@click.option("--compact", is_flag=True, help="Compact each source before synthesis to weigh more sources")
@click.option("--reflect-rounds", default=0, help="Extra reflect->search->re-synthesize cycles (0=off)")
@click.option("--perspectives/--no-perspectives", default=True, help="Perspective-guided query generation")
@click.option("--verify/--no-verify", default=True, help="Verify each finding against its cited source")
def research_cli(query, depth, model, api_key, api_base, config_file, max_sources, output, pdf, md, symbols, language, fast, extractor, jina_fallback, js_render, effort, compact, reflect_rounds, perspectives, verify):
    """Generate a cited research report with a configured LLM.

    Research any topic with web search + LLM analysis + market data.

    \b
    Examples:
        sibyl "Canadian housing market outlook 2026"
        sibyl "AI industry analysis" -d 3 --pdf --symbols NVDA,GOOGL,META
        sibyl "Bitcoin price prediction" -s BTC-USD,ETH-USD --pdf -o reports/
    """
    if find_spec("litellm") is None or find_spec("fpdf") is None:
        raise click.ClickException(
            "Report generation requires: pip install 'sibyl-research[report]'."
        )
    if symbols and any(
        find_spec(name) is None
        for name in ("matplotlib", "pandas", "pytrends", "yfinance")
    ):
        raise click.ClickException(
            "Market data requires: pip install 'sibyl-research[finance]'."
        )
    if extractor == "trafilatura" and find_spec("trafilatura") is None:
        raise click.ClickException(
            "The trafilatura extractor requires: pip install 'sibyl-research[extract]'."
        )

    if config_file:
        cfg = Config.from_yaml(config_file)
    else:
        cfg = Config.from_env(model=model, api_key=api_key, api_base=api_base)

    if effort:
        from .config import TIERS
        depth = TIERS[effort].depth  # effort tier wins over --depth
    cfg.max_depth = depth
    cfg.max_sources = max_sources
    cfg.fast = fast
    cfg.extractor = extractor
    cfg.jina_fallback = jina_fallback
    cfg.js_render = js_render
    cfg.compact_sources = compact
    if compact and cfg.max_synth_sources == 12:
        cfg.max_synth_sources = 40  # compaction makes more sources affordable
    cfg.reflect_rounds = reflect_rounds
    cfg.perspectives = perspectives
    cfg.verify_claims = verify

    if not cfg.has_llm_credentials():
        raise click.ClickException(
            "One-shot research requires an LLM provider key or a configured "
            "local/API-base backend. Use `sibyl gather` for keyless research."
        )

    click.echo(f"Sibyl researching: {query}")
    click.echo(f"Depth: {depth} | Model: {cfg.providers[0].model if cfg.providers else 'auto'}")
    if symbols:
        click.echo(f"Market data: {symbols}")
    click.echo()

    result = asyncio.run(_run(cfg, query, depth, symbols, language))

    # Terminal output
    from .mcp_server import _format_report
    report_text = _format_report(result)
    click.echo(report_text)
    if result.status != "ok":
        raise click.exceptions.Exit(1)

    # PDF output
    if pdf:
        from .reporter import generate_pdf
        path = generate_pdf(result, output)
        click.echo(f"\nPDF saved: {path}")

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
        click.echo(f"Markdown saved: {md_path}")


async def _run(cfg, query, depth, symbols="", language="auto"):
    from .researcher import Researcher

    researcher = Researcher(cfg)
    result = await researcher.research(
        query,
        depth=depth,
        language=language,
        on_progress=lambda msg: click.echo(f"  {msg}"),
    )

    # Fetch market data if symbols provided
    if symbols and result.status == "ok":
        from .data import fetch_multiple, format_data_summary, generate_chart, generate_comparison_chart
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        click.echo(f"  Fetching market data for {', '.join(symbol_list)}...")
        series = await fetch_multiple(symbol_list, "1y")
        if series:
            result.market_data_summary = format_data_summary(series)
            chart_path = generate_chart(series, f"{', '.join(s.name for s in series)} — 1 Year")
            result.charts.append(chart_path)
            # Also generate comparison bar chart if multiple symbols
            if len(series) > 1:
                comp_path = generate_comparison_chart(series, f"Performance Comparison — 1 Year")
                result.charts.append(comp_path)
            click.echo(f"  Chart generated: {chart_path}")

    return result


@click.command(name="gather")
@click.argument("query")
@click.option("--max-sources", "-n", default=10, show_default=True, type=int)
@click.option("--chars-per-source", default=7000, show_default=True, type=int)
@click.option(
    "--ranker",
    type=click.Choice(["lexical", "flashrank", "none"]),
    default="lexical",
    show_default=True,
)
@click.option(
    "--render-thin-pages",
    is_flag=True,
    help="Send thin-page URLs to Jina Reader for another extraction attempt.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
def gather_cli(query, max_sources, chars_per_source, ranker, render_thin_pages, output_format):
    """Gather keyless web evidence without generating an answer."""
    bundle = asyncio.run(
        gather_bundle(
            query,
            max_sources=max_sources,
            chars_per_source=chars_per_source,
            ranker=ranker,
            render_thin_pages=render_thin_pages,
        )
    )
    if output_format == "json":
        click.echo(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2))
    else:
        from .retrieval import render_source_bundle

        click.echo(render_source_bundle(bundle))
    if bundle.status in {"invalid_request", "failed"}:
        raise click.exceptions.Exit(1)


@click.group()
@click.version_option(package_name="sibyl-research")
def cli():
    """Research the web as evidence or generate an optional cited report.

    Existing `sibyl "query"` report commands remain supported.
    """


cli.add_command(gather_cli)
cli.add_command(research_cli)


def main():
    args = sys.argv[1:]
    group_commands = {"gather", "research", "--help", "-h", "--version"}
    if args and args[0] not in group_commands:
        research_cli.main(args=args, prog_name="sibyl")
        return
    cli.main(args=args or ["--help"], prog_name="sibyl")


if __name__ == "__main__":
    main()
