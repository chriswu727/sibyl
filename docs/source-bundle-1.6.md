# SourceBundle 1.6 consumer contract

Use [`source_bundle_1_6.example.json`](source_bundle_1_6.example.json) as a contract fixture in Loop Agent, Argus, and other consumers. Sibyl's tests validate the fixture against the same typed model used by the MCP response and verify that MCP serialization produces the fixture unchanged.

## Compatibility rules

- Require schema major version `1`; accept additive fields within the same major version instead of comparing the entire version string or exact key set.
- Check `status` before reading evidence. Only `ok` is synthesis-ready; `insufficient_evidence` may contain leads but must retain its warning.
- `limited` evidence also maps to `insufficient_evidence`; consumers should refine the query instead of treating a thin or single-cluster result as complete.
- Read `diagnostics.recommended_action` before continuing: `synthesize` permits evidence-based synthesis, `refine_query` asks for a narrower retrieval, `decompose_query` requires separately verifiable subquestions, `retry` means the retrieval path failed, and `revise_request` means the request itself was invalid.
- `diagnostics.query_complexity` is `multi_step` only for high-confidence dependent fact chains. Sibyl fails those broad calls closed with `multi_step_query`; gather each atomic fact separately instead of assuming one result set proves the chain.
- Inspect `diagnostics.search_queries` to see the original and any deterministic focused query used during retrieval.
- Inspect `diagnostics.search_providers` to see which search and metadata providers contributed retained candidates. A configured provider can fail over, so do not infer the actual path from environment configuration alone.
- Compare aggregate `query_term_coverage` with `max_source_query_term_coverage`. A large gap can mean different sources each mention disconnected pieces of the question rather than one source supporting the requested fact.
- Treat `citation_id` as bundle-scoped. Persist `bundle_id`, `source_id`, and `passage_id` together when storing evidence.
- Verify `content_hash` before reusing cached source or passage text.
- Treat `relevance_score` and passage `score` as ranking signals, not truth probabilities.
- Treat `quality_score: null` as unassessed, never as zero.

## Content origin

`content_origin` is an enum with five values:

- `direct_fetch`: full text extracted from the destination response.
- `jina_reader`: full text returned by Jina Reader after a blocked or thin direct response.
- `wikipedia_api`: article text returned by the Wikipedia API.
- `search_snippet`: partial search-result text used only when full text was unavailable.
- `crossref_api`: structured bibliographic metadata returned directly by Crossref.

Consumers should use `search_snippet` as a lead for additional retrieval and avoid treating it as equivalent to full-page evidence.

`crossref_api` is stronger than an ordinary search snippet for fields explicitly identified in the returned text, such as DOI or `published-online`. It does not authorize interpreting Crossref record creation or update timestamps as publication dates.

Search snippets still participate in relevance ranking when full pages are available. Their origin label and the substantive-source diagnostics prevent a precise metadata result from being discarded while preserving the distinction from retrieved full text.

`substantive_sources` counts retrieved full-text sources that also contain every detected key query anchor. A long page about the general topic does not count as corroboration for a quoted paper, named entity, or other anchored target when that target is absent.

Historical role questions with an explicit year require one local statement that connects the requested role to that year or to a tenure range covering it. Mentions of the role and year in unrelated sections do not satisfy this check, and modified roles such as `honorary`, `acting`, or `vice` do not substitute for the requested role.

## Publication time

`published_at` is a normalized ISO 8601 date or timestamp when Sibyl finds explicit publication metadata; it is `null` when no supported value is present. `published_at_method` identifies the exact metadata path, such as `meta_article_published_time`, `json_ld_date_published`, or `jina_published_time`. It never claims that the publisher-supplied value is true, and consumers must not substitute `retrieved_at` for a missing publication time.

## Independent content

`content_cluster_id` groups exact or high-containment near-duplicate text, including syndicated copies on different domains. Sibyl prefers one source from each cluster before filling remaining source slots with duplicates. Diagnostics expose candidate and selected duplicate counts plus `independent_content_clusters`; this is stronger than domain count because two domains can carry the same underlying report. Cluster IDs are content-derived comparison labels, not source identities or credibility scores.

## Forward-compatible parsing

Consumers should ignore unknown object fields, preserve unknown diagnostic reason strings, and fail closed only when the schema major version, `status`, or required identity fields are unsupported. New minor versions may add optional source metadata without changing existing meanings.
