"""Validate a wheel in clean, temporary environments outside the source checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path

PROFILES = {
    "core": "",
    "pdf-monitor": "pdf,monitor",
    "report-extract": "report,extract",
}
SCRIPTS = Path(__file__).resolve().parent


def check(wheel: Path, version: str, profile: str) -> None:
    with tempfile.TemporaryDirectory(prefix="sibyl-wheel-") as directory:
        root = Path(directory)
        environment = root / "environment"
        venv.EnvBuilder(with_pip=True).create(environment)
        bin_path = environment / ("Scripts" if os.name == "nt" else "bin")
        python = bin_path / ("python.exe" if os.name == "nt" else "python")
        executable = bin_path / ("sibyl-mcp.exe" if os.name == "nt" else "sibyl-mcp")
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SIBYL_") and key not in {"PYTHONPATH", "PYTHONHOME"}
        }

        def run(*arguments: str | Path) -> None:
            subprocess.run(
                [str(item) for item in arguments],
                cwd=root,
                env=env,
                check=True,
                timeout=600,
            )

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
        run(python, SCRIPTS / "check_installed_distribution.py", version)
        run(python, SCRIPTS / "check_optional_dependencies.py", profile)
        run(
            python,
            SCRIPTS / "check_mcp_stdio.py",
            executable,
            *(["--monitor"] if profile == "pdf-monitor" else []),
        )
        run(
            python,
            SCRIPTS / "check_research_workflow.py",
            *(["--pdf"] if profile == "report-extract" else []),
        )
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
    wheel = args.wheel.resolve(strict=True)
    if wheel.suffix != ".whl":
        parser.error("Expected one built wheel")
    for profile in args.profile or PROFILES:
        check(wheel, args.version, profile)


if __name__ == "__main__":
    main()
