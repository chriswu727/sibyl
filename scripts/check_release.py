"""Validate that a release tag matches the package and changelog versions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path


def validate_release(
    tag: str, root: Path, *, require_release_note: bool = False
) -> str:
    init_text = (root / "sibyl" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$',
        init_text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError("sibyl/__init__.py does not define __version__")

    version = match.group(1)
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ValueError(f"release tag {tag!r} must equal {expected_tag!r}")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if re.search(rf"^## {re.escape(version)}(?:\s|$)", changelog, re.MULTILINE) is None:
        raise ValueError(f"CHANGELOG.md has no section for {version}")
    if require_release_note:
        heading = re.search(
            rf"^## {re.escape(version)} — (\d{{4}}-\d{{2}}-\d{{2}})\s*$",
            changelog,
            re.MULTILINE,
        )
        if heading is None:
            raise ValueError(
                "Publication requires a dated changelog section, not Unreleased"
            )
        date.fromisoformat(heading[1])

    server = json.loads((root / "server.json").read_text(encoding="utf-8"))
    if server.get("version") != version:
        raise ValueError("server.json version must match the package version")
    packages = server.get("packages") or []
    if len(packages) != 1 or packages[0].get("version") != version:
        raise ValueError("server.json package version must match the package version")
    server_name = server.get("name")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if not server_name or f"mcp-name: {server_name}" not in readme:
        raise ValueError("README.md must contain the matching MCP Registry name")
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--require-release-note", action="store_true")
    args = parser.parse_args()
    try:
        version = validate_release(
            args.tag,
            Path(__file__).resolve().parents[1],
            require_release_note=args.require_release_note,
        )
    except (OSError, ValueError) as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"release tag and package metadata agree on {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
