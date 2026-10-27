"""Evidence retrieval for AI agents with auditable source provenance."""

from .api import gather_bundle, gather_sources
from .evidence import (
    BundleDiagnostics,
    BundleStatus,
    EvidencePassage,
    EvidenceSource,
    EvidenceSufficiency,
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

__version__ = "0.4.0b1"
