# SourceBundle 1.6 consumer contract

Use [`source_bundle_1_6.example.json`](source_bundle_1_6.example.json) as a contract fixture for consuming agents and applications. Sibyl's tests validate it against the typed MCP model and check serialization. This describes unreleased `main`; see the [documentation index](README.md) for installation and release status.

## Compatibility rules

- Require schema major version `1`; accept additive fields within the same major version instead of comparing the entire version string or exact key set.
- Check both `status` and `diagnostics.recommended_action` before synthesis. An `ok` status alone is not sufficient; `insufficient_evidence` may contain leads but must retain its warning.
- In the search-bundle pipeline, `limited` evidence maps to `insufficient_evidence`. A direct source read inside a research job can have `status="ok"` for usable extraction while remaining `limited` with `refine_query`; it is not a research-completeness verdict.
- Read `diagnostics.recommended_action` before continuing: `synthesize` permits evidence-based synthesis, `refine_query` asks for a narrower retrieval, `decompose_query` requires separately verifiable subquestions, `retry` means the retrieval path failed, and `revise_request` means the request itself was invalid.
- `diagnostics.query_complexity` is `multi_step` only for high-confidence dependent fact chains. Sibyl fails those broad calls closed with `multi_step_query`; gather each atomic fact separately instead of assuming one result set proves the chain.
- For an atomic query with only one relevant domain, Sibyl may issue one bounded exclusion search for another domain. `refinement_searches` and `refinement_failures` expose that attempt; failure of this supplemental search retains the initial evidence instead of failing the whole bundle.
- Inspect `diagnostics.search_queries` to see the original and any deterministic focused query used during retrieval.
- Multi-domain `site:` queries may perform up to two missing-domain follow-ups;
  these also increment `refinement_searches` and retain failures in `refinement_failures`.
- Inspect `diagnostics.search_providers` to see which search and metadata providers contributed retained candidates. A configured provider can fail over, so do not infer the actual path from environment configuration alone.
- Compare aggregate `query_term_coverage` with `max_source_query_term_coverage`. A large gap can mean different sources each mention disconnected pieces of the question rather than one source supporting the requested fact.
- Treat `citation_id` as bundle-scoped. Persist `bundle_id`, `source_id`, and `passage_id` together when storing evidence.
- Verify `content_hash` before reusing cached source or passage text.
- Treat `relevance_score` and passage `score` as ranking signals, not truth probabilities.
- Treat `quality_score: null` as unassessed, never as zero.

## Content origin

`content_origin` is an enum with five values:

- `direct_fetch`: bounded text extracted from the destination response.
- `jina_reader`: bounded text returned by Jina Reader after a blocked or thin direct response.
- `wikipedia_api`: article text returned by the Wikipedia API.
- `search_snippet`: partial search-result text used only when full text was unavailable.
- `crossref_api`: structured bibliographic metadata returned directly by Crossref.

Consumers should use `search_snippet` as a lead for additional retrieval and avoid treating it as equivalent to full-page evidence.

`crossref_api` is stronger than an ordinary search snippet for fields explicitly identified in the returned text, such as DOI or `published-online`. It does not authorize interpreting Crossref record creation or update timestamps as publication dates.

Search snippets still participate in relevance ranking when full pages are available. Their origin label and the substantive-source diagnostics prevent a precise metadata result from being discarded while preserving the distinction from retrieved full text.

`substantive_sources` counts retrieved full-text sources that also contain every detected key query anchor. A long page about the general topic does not count as corroboration for a quoted paper, named entity, or other anchored target when that target is absent.

Historical role questions with an explicit year require one local statement that connects the requested role to that year or to a tenure range covering it. Mentions of the role and year in unrelated sections do not satisfy this check, and modified roles such as `honorary`, `acting`, or `vice` do not substitute for the requested role.

## PDF evidence and offsets

Optional PDF input uses `source_type="pdf"` and `content_origin="direct_fetch"`.
These labels describe format and retrieval route, not peer review or credibility.
Extracted text includes `[PDF page N]` markers where page text is emitted. Passage
hashes and character offsets refer to that extracted text, not raw PDF byte offsets
or page coordinates; a passage can span page boundaries.

Preserve `diagnostics.sufficiency_reasons`, including character/page-limit and
unreadable-page warnings. Research-job limitations retain PDF warnings through
report export. A text PDF may be partially read; image-only input is not OCR'd.
The contract does not claim layout/table/formula fidelity or complete-paper coverage.

## Publication time

`published_at` is a normalized ISO 8601 date or timestamp when Sibyl finds explicit publication metadata; it is `null` when no supported value is present. `published_at_method` identifies the exact metadata path, such as `meta_article_published_time`, `json_ld_date_published`, or `jina_published_time`. It never claims that the publisher-supplied value is true, and consumers must not substitute `retrieved_at` for a missing publication time.

Explicit article-date text can supply `published_at_method="article_date_text"`
when metadata is absent and the supported publication markers are unambiguous.
Unix-epoch timestamp placeholders are rejected. Date parsing and period matching
are hints, not a guarantee that a page is current or its claims are correct.

## Independent content

`content_cluster_id` groups exact or high-containment near-duplicate text, including syndicated copies on different domains. Sibyl prefers one source from each cluster before filling remaining source slots with duplicates. Diagnostics expose candidate and selected duplicate counts plus `independent_content_clusters`; this is stronger than domain count because two domains can carry the same underlying report. Cluster IDs are content-derived comparison labels, not source identities or credibility scores.

## Forward-compatible parsing

Consumers should ignore unknown object fields, preserve unknown diagnostic reason strings, and fail closed only when the schema major version, `status`, or required identity fields are unsupported. New minor versions may add optional source metadata without changing existing meanings.
