# Sibyl launch readiness

Last reviewed: 2026-07-20

## Decision

Sibyl is ready to be shown as a public beta. It is not ready for a broad production launch or a v0.4.0 release.

The package, interfaces, evidence contract, safety controls, and offline regressions are working. The blocking evidence is the dated three-repeat live retrieval run: safety and latency passed, but answer coverage, synthesis-ready bundle rate, and repeat stability missed their thresholds. The optional one-shot report path also lacks a current, bounded model-backed evaluation.

## What is genuinely differentiated

Sibyl's differentiation is the combination of these properties, not web search or MCP alone:

- The default product is a local, keyless evidence handoff rather than a hosted answer API.
- `SourceBundle` preserves bundle, source, and passage identities; hashes; character offsets; timestamps; content origin; and diagnostics in a typed, versioned contract.
- Content-derived clusters expose syndicated or substantially duplicated text instead of treating different domains as independent support.
- Limited, missing, and failed retrievals remain explicit rather than being rendered as completed research.
- The host model can remain the reasoning layer, while an optional configured model can produce a cited report when a one-shot workflow is preferred.

Individual capabilities are not unique. Exa, Tavily, Firecrawl, and Perplexity all offer search or research through MCP. Exa offers AI-oriented search and extracted contents; Tavily offers hosted search and extraction; Firecrawl adds crawling and broad extraction; Perplexity provides hosted search, reasoning, and deep-research answers. Their hosted paths are stronger today in index quality, operational reliability, and turnkey synthesis, and they require API credentials or account authorization for normal production use.

Primary references:

- [Exa Search API](https://exa.ai/docs/reference/search)
- [Tavily MCP server](https://docs.tavily.com/documentation/mcp)
- [Firecrawl MCP server](https://docs.firecrawl.dev/mcp)
- [Perplexity MCP server](https://docs.perplexity.ai/docs/getting-started/integrations/mcp-server)

## Verified product evidence

| Gate | Result | Status |
|---|---:|---|
| Unit suite | 244 tests on the launch-candidate branch | Pass |
| Python support in CI | 3.10, 3.11, 3.12 | Pass |
| Fixed lexical ranking | 8/8 at rank 1 | Pass |
| Fixed end-to-end retrieval | 8/8 usable and structurally valid | Pass |
| Source-quality baseline | 83.3% selective accuracy at 75% coverage | Pass |
| Default wheel installation | Clean install, dependency check, four keyless tools | Pass |
| Report-extra installation | Clean install, dependency check, ten report-profile tools | Pass |
| Real stdio MCP call | Structured evidence returned from the built product path | Pass |
| Adversarial live safety | 100% across 12 traps and three repeats | Pass |
| Live p95 latency | 10.5 seconds; threshold 30 seconds | Pass |
| Live answer coverage | 75.9%; threshold 80% | **Fail** |
| Live synthesis-ready bundles | 56.1%; threshold 75% | **Fail** |
| Live repeat stability | 33.3%; threshold 90% | **Fail** |
| One-shot report quality | No current authorized model-backed launch run | **Not tested** |

The complete live artifact is [`evals/results/live-retrieval-2026-07-21.json`](../evals/results/live-retrieval-2026-07-21.json). It records `passed: false` and must not be described as launch evidence.

## Why the live gate failed

The live run exposed two different limits:

1. Anonymous public search endpoints vary and throttle under sustained use. The same question often alternated between multiple substantive domains and a Wikipedia-only result. This changed the bundle from `ok` to `insufficient_evidence` even when the answer text remained present.
2. Some tasks need specialized retrieval or multiple queries. DOI lookup, exact publication dates, obscure historical facts, and multi-hop questions were the persistent misses. One broad keyless query is not yet a reliable replacement for an academic index or an agent-directed evidence loop.

Lowering the thresholds or treating a single domain as independent corroboration would hide these limits rather than fix them.

## Launch gates

Do not publish v0.4.0, update the official MCP Registry, or submit broad directory promotions until all of the following are true:

1. A new dated 66-case, three-repeat live run passes every committed threshold without changing the thresholds after seeing the result.
2. The optional report path completes a bounded evaluation with a declared model, configuration, cost, rubric, and saved results. Running this gate requires explicit approval because it consumes model quota or API spend.
3. The release wheel passes the clean default and report-extra installation smoke tests.
4. GitHub CI passes on Python 3.10, 3.11, and 3.12.
5. README, PyPI metadata, Registry metadata, and the product page describe the same capabilities, maturity, install command, and limitations.

## Work required before launch

Priority order:

1. Add an optional production search provider backed by a supported API, while retaining the keyless path as a zero-setup beta fallback.
2. Add specialized academic metadata retrieval for quoted paper titles and DOI/publication-date questions.
3. Add an agent-facing retry/refinement signal for limited bundles so the host can issue a focused follow-up instead of treating the first broad query as final.
4. Rerun the live gate and compare it with the retained failed baseline.
5. With explicit quota approval, evaluate the one-shot report path and either improve it or keep it clearly secondary.
6. Only after every gate is green, date the changelog, merge the launch candidate, tag v0.4.0, and let the release workflow publish GitHub assets, PyPI, and Registry metadata.

## Claims allowed today

Allowed:

- Public-beta web research for AI agents.
- Keyless evidence retrieval through MCP, CLI, and Python.
- Typed passage provenance, duplicate-content signals, and explicit evidence gaps.
- Optional model-backed cited reports whose quality depends on the configured model.

Not supported yet:

- Production-grade or highly available keyless search.
- A guarantee that citations are correct, independent, or true.
- A claim that one call reliably completes deep or multi-hop research.
- A claim that the one-shot report path matches leading hosted research products.
