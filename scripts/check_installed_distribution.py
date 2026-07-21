"""Smoke-check an installed Sibyl wheel without importing optional tool modules."""
from __future__ import annotations

import sys
from importlib.metadata import distribution

import sibyl


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_installed_distribution.py <expected-version>", file=sys.stderr)
        return 2

    expected_version = sys.argv[1]
    installed = distribution("sibyl-research")
    if sibyl.__version__ != expected_version or installed.version != expected_version:
        print(
            "installed version mismatch: "
            f"module={sibyl.__version__!r}, metadata={installed.version!r}, "
            f"expected={expected_version!r}",
            file=sys.stderr,
        )
        return 1

    console_scripts = {
        entry.name: entry.value
        for entry in installed.entry_points
        if entry.group == "console_scripts"
    }
    expected_scripts = {
        "sibyl": "sibyl.cli:main",
        "sibyl-mcp": "sibyl.mcp_server:main",
    }
    if not expected_scripts.items() <= console_scripts.items():
        print(f"missing console scripts: {expected_scripts!r} not in {console_scripts!r}", file=sys.stderr)
        return 1
    if not callable(sibyl.gather_bundle) or not callable(sibyl.gather_sources):
        print("missing top-level keyless retrieval API", file=sys.stderr)
        return 1

    print(
        f"installed sibyl-research {installed.version} with console scripts "
        "and the keyless Python API"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
