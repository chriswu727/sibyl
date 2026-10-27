"""Model-independent recorded evidence for offline research tests."""

from sibyl.evidence import (
    BundleDiagnostics,
    EvidencePassage,
    EvidenceSource,
    SourceBundle,
)


def evidence_bundle(query="question", count=6, origin="direct_fetch"):
    sources = []
    for index in range(count):
        text = f"Evidence {query} {index}. " * 40
        sources.append(
            EvidenceSource(
                f"{query}-{index}",
                f"https://example.com/{query}/{index}",
                f"Title {index}",
                "2026-10-20T00:00:00Z",
                f"hash-{query}-{index}",
                "web",
                len(text),
                [
                    EvidencePassage(
                        "P1", f"{query}/{index}/P1", text, f"passage-{query}-{index}"
                    )
                ],
                content_origin=origin,
            )
        )
    return SourceBundle(
        "1.6",
        f"bundle-{query}",
        query,
        "ok" if count else "insufficient_evidence",
        sources,
        BundleDiagnostics(
            count,
            count,
            count,
            count,
            0,
            0,
            0,
            count,
            8,
            8,
            4000,
            4000,
            1,
            evidence_sufficiency="sufficient" if count else "insufficient",
            recommended_action="synthesize" if count else "refine_query",
        ),
    )
