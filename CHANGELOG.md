# Changelog

## 0.3.0 — 2026-07-14

### Added

- SourceBundle 1.5 with stable source and passage IDs, content hashes, source offsets, retrieval timestamps, and explicit failure states.
- Dependency-free lexical source and passage ranking, with optional local FlashRank reranking and explicit fallback diagnostics.
- Query coverage, domain diversity, evidence volume, and deterministic evidence-sufficiency diagnostics.
- Fixed offline ranker and end-to-end retrieval pipeline regression suites.
- MCP single-flight retrieval reuse with a 30-second TTL for matching successful requests.

### Changed

- The one-shot research path uses local retrieval ranking instead of an LLM reranking call by default.
- Missing or weak evidence now produces truthful `insufficient_evidence` or `failed` results instead of a completed-looking answer.
- Jina rendering shares concurrency and keyless start-rate limits across retrieval batches.

### Security

- URL fetching rejects non-public destinations, unsafe ports, credential-bearing URLs, and unsafe redirects; validated DNS results are pinned so the TCP connection cannot re-resolve to a different address.
- Direct and Jina response bodies are streamed with a 2 MiB decompressed-size limit.

### Fixed

- PDF validation is portable across systems without the optional macOS font.
- Deprecated PDF font arguments no longer emit runtime warnings.
