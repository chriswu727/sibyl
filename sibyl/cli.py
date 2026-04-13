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
def main(query, depth, model, api_key, api_base, config_file, max_sources):
    """Sibyl — AI-powered deep research agent.

    Research any topic with web search + LLM analysis.

    \b
    Examples:
        sibyl "Canadian housing market outlook 2026"
        sibyl "Impact of AI on software engineering jobs" -d 3
        sibyl "Bitcoin price prediction" --model deepseek/deepseek-chat
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

    # Format report
    from .mcp_server import _format_report
    report_text = _format_report(result)
    console.print(Markdown(report_text))


async def _run(cfg: Config, query: str, depth: int):
    researcher = Researcher(cfg)
    return await researcher.research(
        query,
        depth=depth,
        on_progress=lambda msg: console.print(f"  [dim]{msg}[/]"),
    )
