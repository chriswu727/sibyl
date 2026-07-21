# Changelog

## 0.4.0 — Unreleased

### Added

- A public keyless Python API through `from sibyl import gather_bundle, gather_sources`.
- The `sibyl gather` CLI command with text and JSON output.
- Capability-aware MCP profiles plus `--version`, `--list-tools`, and `--profile` verification commands.
- Deterministic focused search variants for natural-language questions, recorded in SourceBundle diagnostics.
- A 66-question, model-free live retrieval launch gate covering answer evidence, adversarial safety, availability, stability, and latency.
- An explicitly configured Tavily general-web provider with transparent keyless fallback.
- Crossref metadata retrieval for academic titles, DOIs, and publication dates.
- Provider-path and per-source query-coverage diagnostics.
- Machine-readable query-complexity and recommended-action diagnostics for agent retry and decomposition loops.
- One bounded exclusion search for atomic queries whose relevant full-text evidence comes from only one domain, with explicit attempt and failure diagnostics.
- A distinct `crossref_api` provenance label and metadata-fallback diagnostic, avoiding confusion with ordinary search snippets.
- Live retrieval metric version 2 separates answerable readiness from trap safety, measures ready-state precision, and records configured and actual search providers.
- Official MCP Registry metadata, machine-readable install guidance, privacy and security policies, contribution guidance, and issue templates.

### Changed

- The default installation contains the keyless evidence product; report, finance, extraction, and reranking dependencies are optional extras.
- Limited evidence now returns `insufficient_evidence` instead of a completed-looking `ok` state.
- The no-key MCP surface exposes only tools that can run with the installed profile.
- Thin-page Jina rendering is opt-in in every workflow, improving default latency and preventing implicit URL disclosure.
- PyPI metadata and CLI language now lead with evidence-first web research rather than prediction use cases.
- Release validation installs the real default dependency set and checks the public API.
- Keyless search uses paced provider requests, an independent DuckDuckGo/Mojeek/Yahoo fallback chain, bounded search batches, and direct Wikipedia API enrichment.
- Search snippets remain eligible for relevance ranking when full pages exist, while retaining their weaker provenance and sufficiency treatment.
- Ranked source selection prefers a new content cluster on a new domain before filling slots with additional same-domain pages.
- The post-Crossref three-repeat keyless launch gate improved answer coverage from 75.9% to 85.2%, stability from 33.3% to 62.1%, and ready bundles from 56.1% to 64.6%; the gate remains failed on stability and readiness.
- A 105-request Tavily pilot reached 88.9% answer coverage and improved answerable readiness from 66.0% to 70.4%, while ready-state precision remained below its 95% gate.

### Fixed

- Default stdio operation no longer emits per-request MCP and HTTP INFO logs.
- Model-backed MCP failures now use protocol-level tool errors instead of successful text responses containing failure messages.
- Natural-language questions retain the original query while adding a deterministic focused search variant.
- Questions with leading context or prepositions, such as `In what year...` and `On which...`, now receive the same focused search variant as questions that begin with a question word.
- Quoted titles use an exact focused query, and named entities receive a bounded Wikipedia lookup.
- Missing query entities and explicitly requested future outcomes no longer produce synthesis-ready evidence states.
- Quoted-target questions no longer combine disconnected query fragments across unrelated sources into a synthesis-ready state.
- Unrelated long pages no longer count as substantive corroboration when they omit the query's key entities or quoted target.
- High-confidence dependent fact chains fail closed with `decompose_query` instead of appearing synthesis-ready after one broad retrieval.
- Historical role queries no longer combine a role mention and an unrelated year from different statements into apparent tenure evidence.

## 0.3.0 — 2026-07-14

### Added

- SourceBundle 1.6 with stable source and passage IDs, content hashes, source offsets, retrieval timestamps, explicit failure states, and per-source content origin.
- Dependency-free lexical source and passage ranking, with optional local FlashRank reranking and explicit fallback diagnostics.
- Query coverage, domain diversity, evidence volume, and deterministic evidence-sufficiency diagnostics.
- Fixed offline ranker and end-to-end retrieval pipeline regression suites.
- Offline contextual source-quality labels and an abstention-aware evaluator.
- A machine-validated SourceBundle 1.6 consumer contract fixture and integration guide.
- Explicit publication-time extraction with normalized values and per-source extraction methods.
- Content-derived near-duplicate clusters, diversity-first source selection, and provenance-aware evidence sufficiency.
- MCP single-flight retrieval reuse with a 30-second TTL for matching successful requests.

### Changed

- The one-shot research path uses local retrieval ranking instead of an LLM reranking call by default.
- Missing or weak evidence now produces truthful `insufficient_evidence` or `failed` results instead of a completed-looking answer.
- Jina rendering shares concurrency and keyless start-rate limits across retrieval batches.
- GitHub Actions use their Node 24 runtime generations.

### Security

- URL fetching rejects non-public destinations, unsafe ports, credential-bearing URLs, and unsafe redirects; validated DNS results are pinned so the TCP connection cannot re-resolve to a different address.
- Direct and Jina response bodies are streamed with a 2 MiB decompressed-size limit.

### Fixed

- PDF validation is portable across systems without the optional macOS font.
- Deprecated PDF font arguments no longer emit runtime warnings.
