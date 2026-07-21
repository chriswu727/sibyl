"""Evidence retrieval for AI agents with auditable source provenance."""

from .api import gather_bundle, gather_sources

from .evidence import (
    BundleDiagnostics,
    BundleStatus,
    EvidenceLoop,
    EvidenceLoopAction,
    EvidenceLoopDiagnostics,
    EvidenceLoopStatus,
    EvidenceLoopStep,
    EvidenceLoopStepSummary,
    EvidencePassage,
    EvidenceSufficiency,
    EvidenceSource,
    SourceBundle,
)

__all__ = [
    "BundleDiagnostics",
    "BundleStatus",
    "EvidenceLoop",
    "EvidenceLoopAction",
    "EvidenceLoopDiagnostics",
    "EvidenceLoopStatus",
    "EvidenceLoopStep",
    "EvidenceLoopStepSummary",
    "EvidencePassage",
    "EvidenceSufficiency",
    "EvidenceSource",
    "SourceBundle",
    "gather_bundle",
    "gather_sources",
]

__version__ = "0.4.0"
