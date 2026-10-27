"""Command-line evidence inspection for the Sibyl research MCP."""

import asyncio
import json
import sys

import click

from .api import gather_bundle


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
def gather_cli(
    query, max_sources, chars_per_source, ranker, render_thin_pages, output_format
):
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
    """Inspect web evidence. Use sibyl-mcp to connect your AI agent."""


cli.add_command(gather_cli)


def main():
    args = sys.argv[1:]
    cli.main(args=args or ["--help"], prog_name="sibyl")


if __name__ == "__main__":
    main()
