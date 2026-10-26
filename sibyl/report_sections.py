"""Shared presentation of evidence-linked report conclusions."""

from .report_models import ResearchReport

CLAIM_SECTION_TITLE = "Evidence-linked conclusions"
CLAIM_SECTION_NOTICE = (
    "Host-declared facts, inferences, recommendations and unknowns. "
    "Citation checks do not independently verify semantic support."
)


def claim_report_body(report: ResearchReport) -> str:
    return report.analysis or "\n\n".join(
        dict.fromkeys([report.summary, *report.key_findings])
    )
