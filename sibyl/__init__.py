"""Web research for agents: structured evidence and optional cited reports."""

from .api import gather_bundle, gather_sources

from .evidence import (
    BundleDiagnostics,
    BundleStatus,
    EvidencePassage,
    EvidenceSufficiency,
    EvidenceSource,
    SourceBundle,
)

__all__ = [
    "BundleDiagnostics",
    "BundleStatus",
    "EvidencePassage",
    "EvidenceSufficiency",
    "EvidenceSource",
    "SourceBundle",
    "gather_bundle",
    "gather_sources",
]

__version__ = "0.3.0"
