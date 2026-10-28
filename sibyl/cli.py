"""Command line: check a ledger, render its report, install the skill."""

from __future__ import annotations

import json
import sys
from importlib import resources
from pathlib import Path

import click

from . import __version__
from .checks import run_check, summary
from .ledger import TEMPLATE, LedgerError, check_path, read_ledger
from .report import render

DEFAULT_SKILL_DIR = Path.home() / ".claude" / "skills" / "sibyl"


def _skill_text() -> str:
    return resources.files("sibyl").joinpath("skill", "SKILL.md").read_text("utf-8")


def _load(path: Path):
    try:
        return read_ledger(path)
    except LedgerError as exc:
        click.echo(f"Invalid ledger {path}:\n{exc}", err=True)
        sys.exit(2)


@click.group()
@click.version_option(__version__, prog_name="sibyl")
def cli():
    """Check research claims against the pages they quote."""


@cli.command("template")
def template_cmd():
    """Print an example ledger to start from."""
    click.echo(json.dumps(TEMPLATE, indent=2, ensure_ascii=False))


@cli.command("check")
@click.argument("ledger", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Print the check result as JSON.")
@click.option(
    "--save-pages",
    type=click.Path(path_type=Path, file_okay=False),
    help="Keep the text read from each page in this directory, named by its SHA-256.",
)
def check_cmd(ledger: Path, as_json: bool, save_pages: Path | None):
    """Read every cited page and check each quote, number and name.

    Writes LEDGER's check result beside it (NAME.check.json). Exits 0 when no
    blocking issue remains, 1 when one does, 2 when the ledger is invalid.
    """
    parsed, digest = _load(ledger)
    result = run_check(parsed, digest, pages_dir=save_pages)
    check_path(ledger).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    click.echo(
        json.dumps(result, indent=2, ensure_ascii=False) if as_json else summary(result)
    )
    sys.exit(0 if result["passed"] else 1)


@cli.command("report")
@click.argument("ledger", type=click.Path(path_type=Path, dir_okay=False))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Markdown file to write (default: NAME.md beside the ledger).",
)
@click.option(
    "--allow-issues",
    is_flag=True,
    help="Render even when the check found blocking issues; they are marked.",
)
def report_cmd(ledger: Path, output: Path | None, allow_issues: bool):
    """Render the Markdown report from a ledger checked with `sibyl check`."""
    parsed, digest = _load(ledger)
    try:
        result = json.loads(check_path(ledger).read_text("utf-8"))
    except (OSError, ValueError):
        click.echo(
            f"No check result for {ledger}; run `sibyl check {ledger}` first.", err=True
        )
        sys.exit(2)
    if result.get("ledger_sha256") != digest:
        click.echo(
            f"{ledger} changed after it was checked; run `sibyl check {ledger}` again.",
            err=True,
        )
        sys.exit(2)
    if not result["passed"] and not allow_issues:
        click.echo(
            "The last check found blocking issues. Fix them and check again, or "
            "pass --allow-issues to render a report that marks them.",
            err=True,
        )
        sys.exit(1)
    target = output or ledger.with_suffix(".md")
    target.write_text(render(parsed, result), "utf-8")
    click.echo(str(target))


@cli.group("skill")
def skill_group():
    """Show or install the agent skill that teaches the workflow."""


@skill_group.command("show")
def skill_show():
    """Print SKILL.md."""
    click.echo(_skill_text(), nl=False)


@skill_group.command("install")
@click.option(
    "--dir",
    "directory",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_SKILL_DIR,
    show_default=True,
    help="Skill directory to write SKILL.md into.",
)
def skill_install(directory: Path):
    """Write SKILL.md into a skills directory."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "SKILL.md"
    target.write_text(_skill_text(), "utf-8")
    click.echo(str(target))


def main():
    cli()


if __name__ == "__main__":
    main()
