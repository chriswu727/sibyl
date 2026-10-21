"""Check repository documentation links and heading anchors without network calls."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def prose(text: str) -> str:
    lines, fence = [], ""
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if not fence:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = ""
            continue
        if not fence:
            lines.append(line)
    return "\n".join(lines)


def anchors(text: str) -> set[str]:
    result, counts = set(), {}
    for title in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", prose(text), re.MULTILINE):
        title = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", title)
        title = re.sub(r"<[^>]+>", "", title).replace("`", "")
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(f"{slug}-{count}" if count else slug)
    result.update(re.findall(r'\bid=["\']([^"\']+)["\']', text))
    return result


def check_document(path: Path, root: Path) -> list[str]:
    text = prose(path.read_text(encoding="utf-8"))
    errors = []
    for target in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", text):
        target = target.strip().split(' "', 1)[0].strip("<>")
        url = urlsplit(target)
        if url.scheme or url.netloc:
            prefix = "/chriswu727/sibyl/blob/main/"
            if url.netloc != "github.com" or not url.path.startswith(prefix):
                continue
            destination = root / unquote(url.path[len(prefix):])
        else:
            destination = path.parent / unquote(url.path) if url.path else path
        destination = destination.resolve()
        if not destination.is_relative_to(root.resolve()) or not destination.exists():
            errors.append(f"{path.relative_to(root)}: missing local target {target}")
        elif url.fragment and destination.suffix == ".md":
            if unquote(url.fragment) not in anchors(destination.read_text(encoding="utf-8")):
                errors.append(f"{path.relative_to(root)}: missing anchor {target}")
    return errors


def main() -> int:
    paths = sorted({*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md")})
    errors = [error for path in paths for error in check_document(path, ROOT)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Documentation links and anchors passed for {len(paths)} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
