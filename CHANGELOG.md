# Changelog

## 0.4.0 — 2026-07-20

### Added

- A public keyless Python API through `from sibyl import gather_bundle, gather_sources`.
- The `sibyl gather` CLI command with text and JSON output.
- Capability-aware MCP profiles plus `--version`, `--list-tools`, and `--profile` verification commands.
- Deterministic focused search variants for natural-language questions, recorded in SourceBundle diagnostics.
- A 66-question, model-free live retrieval launch gate covering answer evidence, adversarial safety, availability, stability, and latency.
- Official MCP Registry metadata, machine-readable install guidance, privacy and security policies, contribution guidance, and issue templates.

### Changed

- The default installation contains the keyless evidence product; report, finance, extraction, and reranking dependencies are optional extras.
- Limited evidence now returns `insufficient_evidence` instead of a completed-looking `ok` state.
- The no-key MCP surface exposes only tools that can run with the installed profile.
- Thin-page Jina rendering is opt-in in every workflow, improving default latency and preventing implicit URL disclosure.
- PyPI metadata and CLI language now lead with evidence-first web research rather than prediction use cases.
- Release validation installs the real default dependency set and checks the public API.

### Fixed

- Model-backed MCP failures now use protocol-level tool errors instead of successful text responses containing failure messages.
- Natural-language questions retain the original query while adding a deterministic focused search variant.

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
