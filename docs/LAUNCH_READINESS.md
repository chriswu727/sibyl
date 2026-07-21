# Sibyl launch readiness

Last reviewed: 2026-07-20

## Decision

Sibyl is ready to be shown as a public beta. It is not ready for a broad production launch or a v0.4.0 release.

The package, interfaces, evidence contract, safety controls, and offline regressions are working. The blocking evidence is the latest dated three-repeat keyless retrieval run: safety, latency, and answer coverage passed, but synthesis-ready bundle rate and repeat stability missed their thresholds. The optional one-shot report path also lacks a current, bounded model-backed evaluation.

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
| Unit suite | 258 tests on the launch-candidate branch | Pass |
| Python support in CI | 3.10, 3.11, 3.12 | Pass |
| Fixed lexical ranking | 8/8 at rank 1 | Pass |
| Fixed end-to-end retrieval | 8/8 usable and structurally valid | Pass |
| Source-quality baseline | 83.3% selective accuracy at 75% coverage | Pass |
| Default wheel installation | Clean install, dependency check, four keyless tools | Pass |
| Report-extra installation | Clean install, dependency check, ten report-profile tools | Pass |
| Real stdio MCP call | Structured evidence returned from the built product path | Pass |
| Adversarial live safety | 100% across 12 traps and three repeats | Pass |
| Live p95 latency | 11.0 seconds; threshold 30 seconds | Pass |
| Live answer coverage | 85.2%; threshold 80% | Pass |
| Live answerable ready rate | 66.0%; threshold 75% | **Fail** |
| Live ready-state precision | 93.9%; threshold 95% | **Fail** |
| Live repeat stability | 62.1%; threshold 90% | **Fail** |
| One-shot report quality | No current authorized model-backed launch run | **Not tested** |

The latest complete three-repeat artifact is [`evals/results/live-retrieval-keyless-post-crossref-2026-07-21.json`](../evals/results/live-retrieval-keyless-post-crossref-2026-07-21.json). It records `passed: false` and must not be described as launch evidence. The [original keyless baseline](../evals/results/live-retrieval-2026-07-21.json) remains available for comparison: answer coverage rose from 75.9% to 85.2% and stability from 33.3% to 62.1% without weakening those thresholds.

Those artifacts used metric version 1, whose `ready_bundle_rate` counted every `status == ok`, including adversarial cases, and did not verify that an `ok` answerable bundle contained the expected answer. Metric version 2 replaces that launch gate with two product-facing measures: `answerable_ready_rate` requires both `ok` and answer evidence on answerable cases, while `ready_answer_precision` measures how often an `ok` state actually contains the answer. Historical v2 values can be derived from the retained run records: the latest keyless run scores 66.0% and 93.9%, below the new 75% and 95% gates.

## Tavily production-search pilot

The authorized single-repeat pilot consumed 105 Tavily basic searches and is retained in [`evals/results/live-retrieval-tavily-pilot-2026-07-21.json`](../evals/results/live-retrieval-tavily-pilot-2026-07-21.json).

| Pilot metric | Tavily | Keyless three-run aggregate | Gate |
|---|---:|---:|---:|
| Answer coverage | 88.9% | 85.2% | 80% |
| Trap safety | 100% | 100% | 80% |
| Answerable ready rate | 70.4% | 66.0% | 75% |
| Ready-state precision | 92.7% | 93.9% | 95% |
| p95 latency | 10.7 s | 11.0 s | 30 s |

Tavily improved answerable readiness by 4.4 percentage points, but the single run did not pass readiness or precision and cannot measure repeat stability. A full 315-request run is not justified until the false-ready cases and metric instrumentation are addressed.

## Why the live gate failed

The two live runs exposed two different limits:

1. Anonymous public search endpoints vary and throttle under sustained use. The same question often alternated between multiple substantive domains and a Wikipedia-only result. This changed the bundle from `ok` to `insufficient_evidence` even when the answer text remained present. The new pacing, candidate handling, and academic metadata reduced but did not remove this instability.
2. Some tasks need specialized retrieval or multiple queries. DOI lookup, exact publication dates, obscure historical facts, and multi-hop questions were the persistent misses. One broad keyless query is not yet a reliable replacement for an academic index or an agent-directed evidence loop.

The launch candidate now includes an explicit Tavily path and Crossref academic metadata, but neither changes this decision yet. The Tavily path has offline contract coverage and still requires a complete budgeted live run. Crossref now supplies the retained DOI answer in all three keyless repeats while honestly leaving an ambiguous online-publication date case insufficient; record creation timestamps are not treated as publication dates.

Lowering the thresholds or treating a single domain as independent corroboration would hide these limits rather than fix them.

## Launch gates

Do not publish v0.4.0, update the official MCP Registry, or submit broad directory promotions until all of the following are true:

1. A new dated 66-case, three-repeat live run passes every committed threshold without changing the thresholds after seeing the result.
2. The optional report path completes a bounded evaluation with a declared model, configuration, cost, rubric, and saved results. Running this gate requires explicit approval because it consumes model quota or API spend.
3. The release wheel passes the clean default and report-extra installation smoke tests.
4. GitHub CI passes on Python 3.10, 3.11, and 3.12.
5. README, PyPI metadata, Registry metadata, and the product page describe the same capabilities, maturity, install command, and limitations.
6. At least three external beta users install Sibyl without maintainer intervention, complete a representative research task, and report whether the evidence state and citations were understandable.

## Work required before launch

Priority order:

1. Use metric version 2 provider-path diagnostics to address false-ready and conservative-abstention cases from the Tavily pilot, then decide whether a complete 315-request run is justified; the keyless path remains the zero-setup beta fallback.
2. Extend the new Crossref metadata path where publisher-specific facts are absent or ambiguous; never substitute Crossref record timestamps for publication dates.
3. Add an agent-facing retry/refinement signal for limited bundles so the host can issue a focused follow-up instead of treating the first broad query as final.
4. Rerun the live gate and compare it with the retained failed baseline.
5. With explicit quota approval, evaluate the one-shot report path and either improve it or keep it clearly secondary.
6. Run a small external beta and convert installation, tool-selection, and evidence-interpretation failures into regression cases or documentation fixes.
7. Only after every gate is green, date the changelog, merge the launch candidate, tag v0.4.0, and let the release workflow publish GitHub assets, PyPI, and Registry metadata.

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
