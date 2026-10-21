"""Citation-based source grouping without changing persisted citation numbers."""

from __future__ import annotations

import re

from .report_models import ResearchReport, Source

REFERENCE_NOTE = "Citation use does not establish that a source supports a claim."
BACKGROUND_NOTE = "Retained research material not cited in the report text."


def reference_groups(
    report: ResearchReport,
) -> list[tuple[str, str, list[tuple[int, Source]]]]:
    sections = [
        report.summary,
        *report.key_findings,
        report.analysis,
        report.predictions,
        report.cross_analysis,
        report.market_data_summary,
        *report.limitations,
    ]
    for question in report.research_plan.get("questions", []):
        review = question.get("review") or {}
        sections.extend(
            [review.get("conclusion", ""), review.get("counterevidence_note", "")]
        )
    cited = {
        int(number)
        for section in sections
        for number in re.findall(r"\[Source\s+(\d+)\]", section, re.IGNORECASE)
    }
    sources = list(enumerate(report.sources, 1))
    return [
        (title, note, entries)
        for title, note, entries in [
            (
                "Cited Sources",
                REFERENCE_NOTE,
                [(i, src) for i, src in sources if i in cited],
            ),
            (
                "Uncited Background",
                BACKGROUND_NOTE,
                [(i, src) for i, src in sources if i not in cited],
            ),
        ]
        if entries
    ]
