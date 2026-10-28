"""Validate a wheel in clean, temporary environments outside the source checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import venv
from pathlib import Path

PROFILES = {"core": "", "pdf": "pdf"}

# Checks a ledger against an offline page, renders it, and confirms the
# installed distribution, skill file and optional PDF parser are present.
SMOKE = """
import asyncio, importlib.util, json, sys
from pathlib import Path
import sibyl
from sibyl.checks import run_check
from sibyl.cli import _skill_text
from sibyl.ledger import TEMPLATE, Ledger
from sibyl.report import render
from sibyl.scraper import WebPage

version, profile = sys.argv[1:3]
assert sibyl.__version__ == version, sibyl.__version__
assert _skill_text().startswith("---\\nname: sibyl\\n")
assert bool(importlib.util.find_spec("pypdf")) == (profile == "pdf"), profile
ledger = Ledger.model_validate(TEMPLATE)
quotes = [e.quote for c in ledger.claims for e in c.evidence]

async def reader(urls):
    return [WebPage(url, "Page", " ".join(quotes)) for url in urls]

result = run_check(ledger, "0" * 64, reader=reader)
assert all(e["found"] for c in result["claims"] for e in c["evidence"]), result
assert render(ledger, result).startswith("# What does the research question ask?")
print(json.dumps({"version": version, "profile": profile, "passed": result["passed"]}))
"""


def check(wheel: Path, version: str, profile: str) -> None:
    with tempfile.TemporaryDirectory(prefix="sibyl-wheel-") as directory:
        root = Path(directory)
        environment = root / "environment"
        venv.EnvBuilder(with_pip=True).create(environment)
        bin_path = environment / ("Scripts" if os.name == "nt" else "bin")
        python = bin_path / ("python.exe" if os.name == "nt" else "python")
        executable = bin_path / ("sibyl.exe" if os.name == "nt" else "sibyl")
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SIBYL_") and key not in {"PYTHONPATH", "PYTHONHOME"}
        }

        def run(*arguments: str | Path) -> str:
            return subprocess.run(
                [str(item) for item in arguments],
                cwd=root,
                env=env,
                check=True,
                timeout=600,
                capture_output=True,
                text=True,
            ).stdout

        run(
            python,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--upgrade",
            "pip>=26.2.1",
            "setuptools>=83.0.0",
        )
        extras = PROFILES[profile]
        requirement = f"{wheel}[{extras}]" if extras else str(wheel)
        run(python, "-m", "pip", "install", "--no-cache-dir", requirement)
        run(python, "-m", "pip", "check")
        assert version in run(executable, "--version")
        json.loads(run(executable, "template"))
        print(run(python, "-I", "-c", SMOKE, version, profile).strip(), flush=True)
        print(
            f"Clean-wheel {profile} passed; temporary environment removed on exit",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("version")
    parser.add_argument("--profile", choices=PROFILES, action="append")
    args = parser.parse_args()
    for profile in args.profile or PROFILES:
        check(args.wheel.resolve(), args.version, profile)


if __name__ == "__main__":
    main()
