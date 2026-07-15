"""Validate that a release tag matches the package and changelog versions."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def validate_release(tag: str, root: Path) -> str:
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
    return version


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_release.py v<package-version>", file=sys.stderr)
        return 2
    try:
        version = validate_release(sys.argv[1], Path(__file__).resolve().parents[1])
    except (OSError, ValueError) as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"release tag and package metadata agree on {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
