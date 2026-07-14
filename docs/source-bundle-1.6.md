# SourceBundle 1.6 consumer contract

Use [`source_bundle_1_6.example.json`](source_bundle_1_6.example.json) as a contract fixture in Loop Agent, Argus, and other consumers. Sibyl's tests validate the fixture against the same typed model used by the MCP response and verify that MCP serialization produces the fixture unchanged.

## Compatibility rules

- Require schema major version `1`; accept additive fields within the same major version instead of comparing the entire version string or exact key set.
- Check `status` before reading evidence. Only `ok` is synthesis-ready; `insufficient_evidence` may contain leads but must retain its warning.
- Treat `citation_id` as bundle-scoped. Persist `bundle_id`, `source_id`, and `passage_id` together when storing evidence.
- Verify `content_hash` before reusing cached source or passage text.
- Treat `relevance_score` and passage `score` as ranking signals, not truth probabilities.
- Treat `quality_score: null` as unassessed, never as zero.

## Content origin

`content_origin` is an enum with four values:

- `direct_fetch`: full text extracted from the destination response.
- `jina_reader`: full text returned by Jina Reader after a blocked or thin direct response.
- `wikipedia_api`: article text returned by the Wikipedia API.
- `search_snippet`: partial search-result text used only when full text was unavailable.

Consumers should use `search_snippet` as a lead for additional retrieval and avoid treating it as equivalent to full-page evidence.

## Publication time

`published_at` is a normalized ISO 8601 date or timestamp when Sibyl finds explicit publication metadata; it is `null` when no supported value is present. `published_at_method` identifies the exact metadata path, such as `meta_article_published_time`, `json_ld_date_published`, or `jina_published_time`. It never claims that the publisher-supplied value is true, and consumers must not substitute `retrieved_at` for a missing publication time.

## Forward-compatible parsing

Consumers should ignore unknown object fields, preserve unknown diagnostic reason strings, and fail closed only when the schema major version, `status`, or required identity fields are unsupported. New minor versions may add optional source metadata without changing existing meanings.
