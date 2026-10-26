"""Shared presentation of evidence-linked report conclusions."""

from .report_models import FindingVerification, ResearchReport

CLAIM_SECTION_TITLE = "Evidence-linked conclusions"
CLAIM_SECTION_NOTICE = (
    "Host-declared facts, inferences, recommendations and unknowns. "
    "Citation checks do not independently verify semantic support."
)
LEGACY_VERDICT_NOTICE = (
    "Historical recorded verdicts; not revalidated on read or export. "
    "Recorded support and confidence are not independent verification."
)


def legacy_verdict_tag(verdict: FindingVerification) -> str:
    support = "supported" if verdict.supported else "unverified"
    return f"[historical: {support}; recorded confidence: {verdict.confidence}]"


def claim_report_body(report: ResearchReport) -> str:
    return report.analysis or "\n\n".join(
        dict.fromkeys([report.summary, *report.key_findings])
    )
