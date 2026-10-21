"""Report-wide verification without treating lexical similarity as entailment."""

from __future__ import annotations

import re

from .report_models import ResearchReport


def citation_errors(report: ResearchReport) -> list[str]:
    errors = []
    for text in [
        report.summary,
        *report.key_findings,
        report.analysis,
        report.predictions,
    ]:
        for number in re.findall(r"\[Source\s+(\d+)\]", text, re.IGNORECASE):
            if not 1 <= int(number) <= len(report.sources):
                errors.append(f"Unknown source citation: {number}")
    return list(dict.fromkeys(errors))


async def audit_report(report, pages, provider, completion=None):
    from .verifier import verify_findings

    blocks = []
    sections = {}
    for name in ("summary", "analysis", "predictions"):
        parts = [
            part for part in re.split(r"\n\s*\n", getattr(report, name)) if part.strip()
        ]
        sections[name] = parts
        blocks.extend((name, index, part) for index, part in enumerate(parts))
    blocks.extend(
        ("key_findings", index, text) for index, text in enumerate(report.key_findings)
    )
    findings = list(report.key_findings)
    report.finding_verifications = []
    checked = []
    for offset in range(0, min(len(blocks), 64), 8):
        batch = blocks[offset : offset + 8]
        verdicts = await verify_findings(
            [text for _, _, text in batch], pages, provider, completion=completion
        )
        checked.extend(verdicts)
    for position, (section, index, text) in enumerate(blocks):
        verdict = checked[position] if position < len(checked) else None
        if verdict is None or not verdict.supported:
            flagged = f"**Unverified:** {text}"
            if section == "key_findings":
                findings[index] = flagged
            else:
                sections[section][index] = flagged
        if section == "key_findings" and verdict is not None:
            verdict.index = index + 1
            report.finding_verifications.append(verdict)
    for name, parts in sections.items():
        setattr(report, name, "\n\n".join(parts))
    report.key_findings = findings
    report.verification_scope = "report_blocks_model_checked"
    report.limitations.append(
        "Automated source-support checks are not independent fact verification. "
        "Unverified blocks are marked inline."
    )
    if len(blocks) > 64:
        report.limitations.append(
            "Verification block budget reached; remaining blocks are unverified."
        )
