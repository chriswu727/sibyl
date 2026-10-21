# Changelog

## 0.4.0 — Unreleased

### Documentation and discoverability

- Reconcile the GitHub landing page, installation/configuration examples, evidence
  contracts, operations, architecture, release readiness, and active roadmap.
- Separate historical experiments from current engineering evidence; remove
  unsupported adjusted-accuracy and model-performance claims from current guidance.
- Add a documentation index and offline checks for repository links and anchors.
- Keep PyPI 0.3.0 distinct from the unreleased 0.4.0 source workflow; no release
  gates are waived by the documentation refresh.

### Operational hardening

- Added optional resource-bounded text-PDF ingestion with page markers, explicit
  extraction limitations, killable workers, and no inherited environment secrets.
- Added opt-in authenticated loopback health/readiness/status/Prometheus endpoints,
  bounded content-free metrics, structured stderr events, and shutdown cleanup.
- Recover oversized evidence checkpoints without losing prior evidence or refunding
  consumed budgets; classify interrupted attempts explicitly on restart.
- Bound export file size and per-directory count/bytes without deleting user files.
- Extend CI and dependency audits to PDF input and monitoring, with real stdio/HTTP
  lifecycle tests and an operations runbook.

### Retrieval reliability

- Preserve main/article content inside navigation and page-wide form wrappers;
  select substantive content instead of empty widgets while retaining table structure.
- Parse parenthesized and quoted `site:` filters without dropping the final domain.
- Bound missing-domain follow-up searches and prioritize domain representation
  before the fetch limit so one publisher cannot consume all initial slots.
- Prioritize requested reporting periods in ranked retrieval and the unpinned
  evidence window; preserve historical queries, fiscal-year context, and pinned sources.
- Reject Unix-epoch timestamp placeholders and recover explicitly labeled issue dates.
- Retain direct-read HTTP/content-type failure reasons without inventing evidence or
  enabling third-party fallback by default.
- Export incomplete host reports with `status="partial"` and visible Markdown/PDF
  warnings. The job lifecycle still becomes `completed` when the report is finalized.

### Runtime compatibility

- Raised the minimum Python version to 3.11 and test/audit coverage to 3.11–3.13.
- Required Pillow 13.0.0 or newer in report, finance, and all extras to exclude
  known-vulnerable older releases; the core installation remains Pillow-free.
- Security matrix jobs now finish independently so one failure does not cancel
  the remaining audits. Python 3.10 users must recreate their environment with
  a supported interpreter before upgrading.

### Evidence-led host research

- Made multi-round retrieval explicit: reviews can propose focused follow-up queries,
  attempts retain prior gaps/revisions and host rationale, and completed searches
  are omitted from follow-up suggestions without resetting the retrieval budget.
- Added per-question support/challenge reviews, counterevidence investigation gates,
  stale-review detection, and explicit incomplete-report delivery.
- Added budgeted source reads and host-declared citation lineage inside persistent jobs.
- Prioritized primary/scholarly source candidates over background material; pinned
  reviewed evidence and exposed excluded passages for inspection.
- Added academic discovery routing and operator-selected primary domains without
  treating source types as proof of authority, truth, or peer review.
- Preserved short numeric cells and table boundaries during HTML extraction; thin
  direct reads no longer look like successful evidence extraction.
- JSON exports now retain selected passage text/hashes/offsets and the review ledger;
  Markdown references are compact. Keyless/full catalogs contain 12/20 tools.
- Existing checkpoints remain readable. New host jobs require review unless
  `allow_incomplete=true` is explicitly supplied. Controlled model-quality A/B testing
  and OCR/layout-aware full-paper ingestion remain outstanding.

### Persistent research workflows

- Added host-assisted and explicitly configured-model research jobs sharing the
  SourceBundle evidence pipeline. Existing retrieval and one-shot tools remain.
- Added bounded SQLite checkpoints, progress events, cancellation, explicit resume,
  stale-revision protection, immutable completed reports, and checkpoint deletion.
- Interleave evidence across queries so gap-filling results can enter a full source window.
- Removed mandatory finding counts, numeric claims, fabricated forecast probabilities,
  and minimum-length regeneration from report synthesis.
- Verify report blocks as well as findings, visibly flag unsupported content, and
  never treat lexical overlap or string-valued booleans as semantic verification.
- Added provider-free report serialization and JSON export; host-authored reports
  disclose citation-target-only validation.
- Fixed PDF cursor positioning that could silently omit consecutive list items.
- Parallel model failures now cancel and await sibling calls before returning.
- The current keyless MCP profile exposes twelve tools; the full catalog contains twenty.
  Model operations have a 600-second default deadline. Research jobs additionally
  bound retrieval attempts, model-call count, reserved output tokens, and storage.
- Repositioned the product around deep research and traceable reports, without a
  new live-quality claim. Incremental refresh remains future work.

### Runtime and security hardening

- Split MCP composition, tool registration, application operations, and per-server
  runtime state. Added deterministic shutdown and fail-fast admission limits.
- Added total MCP operation deadlines, bounded single-flight retrieval, and
  cancellation when the final consumer leaves. Active initial evidence loops
  can no longer be pruned while retrieval is running.
- Serialized stateful report operations and separated report contracts from the
  optional LLM SDK.
- Bounded gzip/zlib-deflate expansion before allocating decoded page bodies.
- Confined MCP artifacts to `SIBYL_OUTPUT_ROOT`, with private atomic writes,
  collision-resistant names, and relative-path/symlink checks.
- Hid provider exception details and credential-bearing displayed URL metadata;
  preserved structured evidence text, hashes, and offsets.
- Raised MCP and LiteLLM security floors, added isolated core/optional dependency
  audits and scheduled checks, and added Dependabot.
- Restricted Docker build inputs and switched its runtime to a non-root user.
- Redesigned the repository landing page and documented module responsibilities,
  extension points, operational limits, and remaining trust boundaries.

Migration: `save_report(output_dir=...)` now accepts only relative subdirectories
of the operator-controlled output root (default `reports/`). Cancelling the final
retrieval consumer no longer leaves work running in the background. Public
SourceBundle 1.6 and EvidenceLoop 1.0 schemas and tool names are unchanged.

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
- A bounded `gather_evidence` MCP workflow that lets the host plan atomic follow-ups, preserves a lightweight retrieval trace, and validates synthesis-ready supporting steps without MCP sampling.
- A fixed-plan, model-free evidence-loop evaluation; the initial four dependent fact chains passed decomposition, readiness, and answer-evidence checks on the keyless path.
- One bounded exclusion search for atomic queries whose relevant full-text evidence comes from only one domain, with explicit attempt and failure diagnostics.
- A distinct `crossref_api` provenance label and metadata-fallback diagnostic, avoiding confusion with ordinary search snippets.
- Live retrieval metric version 2 separates answerable readiness from trap safety, measures ready-state precision, and records configured and actual search providers.
- Official MCP Registry metadata, machine-readable install guidance, privacy and security policies, contribution guidance, and issue templates.

### Changed

- Product descriptions now consistently position Sibyl as an evidence-retrieval and delivery layer for AI agents; optional one-shot reports are explicitly secondary and experimental.
- Package maturity metadata now matches Sibyl's documented public-beta status.
- The default installation contains the keyless evidence product; report, finance, extraction, and reranking dependencies are optional extras.
- Limited evidence now returns `insufficient_evidence` instead of a completed-looking `ok` state.
- The no-key MCP surface exposes only tools that can run with the installed profile.
- The MCP SDK dependency stays on the compatible stable v1 line; a v2 migration requires separate compatibility validation.
- Thin-page Jina rendering is opt-in in every workflow, improving default latency and preventing implicit URL disclosure.
- PyPI metadata and CLI language now lead with evidence-first web research rather than prediction use cases.
- Release validation installs the real default dependency set and checks the public API.
- CI and release validation open a real stdio client session, verify the five-tool keyless profile, and exercise the no-network decomposition start path.
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
