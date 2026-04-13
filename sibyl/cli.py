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
def main(query, depth, model, api_key, api_base, config_file, max_sources, output, pdf, md):
    """Sibyl -- AI-powered deep research agent.

    Research any topic with web search + LLM analysis.

    \b
    Examples:
        sibyl "Canadian housing market outlook 2026"
        sibyl "Impact of AI on software engineering jobs" -d 3 --pdf
        sibyl "Bitcoin price prediction" --md --pdf -o reports/
    """
    if config_file:
        cfg = Config.from_yaml(config_file)
    else:
        cfg = Config.from_env(model=model, api_key=api_key, api_base=api_base)

    cfg.max_depth = depth
    cfg.max_sources = max_sources

    console.print(f"\n[bold blue]Sibyl[/] researching: [cyan]{query}[/]")
    console.print(f"[dim]Depth: {depth} | Model: {cfg.providers[0].model if cfg.providers else 'auto'}[/]\n")

    result = asyncio.run(_run(cfg, query, depth))

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


async def _run(cfg: Config, query: str, depth: int):
    researcher = Researcher(cfg)
    return await researcher.research(
        query,
        depth=depth,
        on_progress=lambda msg: console.print(f"  [dim]{msg}[/]"),
    )
