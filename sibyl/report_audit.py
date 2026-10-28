"""Deterministic report citation validation, not semantic verification."""

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
