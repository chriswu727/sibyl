"""Render a checked ledger as a Markdown report."""

from __future__ import annotations

from .checks import WARNINGS
from .ledger import Claim, Ledger

SCOPE = (
    "Sibyl looked for each quote above in the text it read from the cited page; "
    "unless marked, every quote was found, and numbers and names in each finding "
    "were checked against its quotes. These are string checks: they do not "
    "establish that a quote means what the claim says, and the conclusions are "
    "the author's."
)


def _escape(text: str) -> str:
    return " ".join(text.split())


def _quote_lines(claim: Claim, numbers: dict[str, int]) -> list[str]:
    lines = []
    for item in claim.evidence:
        label = "Against" if item.role == "challenges" else "Source"
        lines.append(f"   > {_escape(item.quote)}")
        lines.append(f"   > — {label} [{numbers[item.url]}]")
        lines.append("")
    return lines


def render(ledger: Ledger, result: dict) -> str:
    """The report for a ledger and the check result made from it."""
    numbers: dict[str, int] = {}
    for claim in ledger.claims:
        for item in claim.evidence:
            numbers.setdefault(item.url, len(numbers) + 1)
    rows = {row["id"]: row for row in result["claims"]}
    facts = [c for c in ledger.claims if c.kind == "fact" and c.status != "unresolved"]
    inferences = [
        c for c in ledger.claims if c.kind == "inference" and c.status != "unresolved"
    ]
    unresolved = [c for c in ledger.claims if c.status == "unresolved"]
    position = {claim.id: f"finding {i}" for i, claim in enumerate(facts, 1)}
    position |= {claim.id: f"reasoning {i}" for i, claim in enumerate(inferences, 1)}

    lines = [f"# {_escape(ledger.question)}", ""]
    if not result["passed"]:
        lines += [
            "**This report contains claims that failed Sibyl's checks; they are "
            "marked below.**",
            "",
        ]
    if facts:
        lines += ["## Findings", ""]
        for index, claim in enumerate(facts, 1):
            row = rows[claim.id]
            marks = []
            if claim.status == "disputed":
                marks.append("disputed")
            if row["blocking"]:
                marks.append("failed checks: " + ", ".join(row["blocking"]))
            suffix = f" *({'; '.join(marks)})*" if marks else ""
            lines += [f"{index}. {_escape(claim.text)}{suffix}", ""]
            lines += _quote_lines(claim, numbers)
    if inferences:
        lines += ["## Reasoning", ""]
        for index, claim in enumerate(inferences, 1):
            basis = ", ".join(
                position.get(item, "an unresolved claim") for item in claim.based_on
            )
            flag = (
                " *(rests on a claim that failed checks)*"
                if rows[claim.id]["blocking"]
                else ""
            )
            lines += [
                f"{index}. {_escape(claim.text)}{flag}",
                f"   - Based on {basis}: {_escape(claim.rationale)}",
                "",
            ]
    if unresolved:
        lines += ["## Unresolved", ""]
        for claim in unresolved:
            failed = rows[claim.id]["blocking"]
            flag = f" *(failed checks: {', '.join(failed)})*" if failed else ""
            lines += [f"- {_escape(claim.text)}: {_escape(claim.note)}{flag}"]
        lines.append("")
    lines += ["## Counterevidence", ""]
    if ledger.counterevidence:
        lines.append(f"- Searched for: {_escape(ledger.counterevidence.searched_for)}")
        lines.append(f"- Found: {_escape(ledger.counterevidence.found or 'nothing')}")
    else:
        lines.append("- No search for counterevidence is recorded.")
    lines.append("")
    if ledger.gaps:
        lines += ["## Gaps", ""]
        lines += [f"- {_escape(gap)}" for gap in ledger.gaps]
        lines.append("")
    reads = {source["url"]: source for source in result["sources"]}
    lines += ["## Sources", ""]
    for url, number in numbers.items():
        source = reads.get(url, {})
        title = _escape(source.get("title") or "") or url
        detail = (
            f"could not be read: {source['error']}"
            if source.get("error")
            else f"text SHA-256 {source.get('text_sha256', '')[:12]}"
        )
        lines.append(f"{number}. {title} — <{url}> ({detail})")
    lines.append("")
    warnings = [
        f"- {position[row['id']].capitalize()}: "
        + "; ".join(WARNINGS[w] for w in row["warnings"])
        for row in result["claims"]
        if row["warnings"] and row["id"] in position
    ]
    lines += ["## Check notes", "", SCOPE, ""]
    lines.append(
        f"Checked {result['checked_at']} with Sibyl {result['sibyl_version']}; "
        f"ledger SHA-256 {result['ledger_sha256'][:12]}."
    )
    if warnings or result["notes"]:
        lines.append("")
        lines += warnings
        lines += [f"- {note[0].upper()}{note[1:]}." for note in result["notes"]]
    return "\n".join(lines).rstrip() + "\n"
