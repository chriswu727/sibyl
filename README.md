<!-- mcp-name: io.github.chriswu727/sibyl -->

<div align="center">

# Sibyl

**Web research for AI agents — structured evidence or a finished cited report.**

Sibyl is an MCP server and CLI that searches the web, extracts and ranks source material, detects syndicated copies, and preserves citation provenance. Use its keyless retrieval tools to give your own agent a typed, versioned `SourceBundle`, or run the optional LLM-backed research pipeline to synthesize, verify, and format a finished report.

[![Tests](https://github.com/chriswu727/sibyl/actions/workflows/tests.yml/badge.svg)](https://github.com/chriswu727/sibyl/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sibyl-research?color=blue)](https://pypi.org/project/sibyl-research/)
[![Python](https://img.shields.io/pypi/pyversions/sibyl-research)](https://pypi.org/project/sibyl-research/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chriswu727/sibyl/blob/main/LICENSE)

[Install](#quick-start) · [SourceBundle contract](https://github.com/chriswu727/sibyl/blob/main/docs/source-bundle-1.6.md) · [Privacy](https://github.com/chriswu727/sibyl/blob/main/PRIVACY.md) · [Security](https://github.com/chriswu727/sibyl/blob/main/SECURITY.md) · [Changelog](https://github.com/chriswu727/sibyl/blob/main/CHANGELOG.md)

</div>

## What Sibyl is

Sibyl supports two research workflows built around a simple boundary:

> Sibyl retrieves evidence. The calling model decides what the evidence supports.

The recommended path is `gather_bundle()`: a keyless retrieval tool that returns a typed, versioned `SourceBundle`. It does not generate an answer. The bundle carries source and passage identities, hashes, offsets, retrieval and publication metadata, content-origin labels, near-duplicate clusters, relevance signals, and explicit failure states.

Sibyl also provides an optional one-shot `research()` pipeline. That mode uses your configured LLM to search, synthesize, verify, and format a report. It is convenient, but its answer quality depends on the configured model; for agent systems, keeping retrieval and reasoning separate is usually the stronger design.

| Mode | Tool | API key | Who reasons? | Best for |
|---|---|---:|---|---|
| Structured retrieval | `gather_bundle()` | No | Your agent | Pipelines, contracts, machine-readable citations |
| Readable retrieval | `gather_sources()` | No | Your agent | Conversational hosts and manual inspection |
| One-shot research | `research()` | Yes | Sibyl's configured LLM | A finished report from one call |

## Quick start

### 1. Install the current release

```bash
python -m pip install sibyl-research
```

Sibyl requires Python 3.10 or newer. The default installation is the lightweight keyless retrieval product. Optional features are installed explicitly:

```bash
python -m pip install 'sibyl-research[report]'   # LLM report + PDF
python -m pip install 'sibyl-research[finance]'  # market data, trends, charts
python -m pip install 'sibyl-research[rerank]'   # local cross-encoder ranking
python -m pip install 'sibyl-research[all]'      # every optional capability
```

### 2. Add the keyless MCP server

For Claude Code:

```bash
claude mcp add sibyl -- uvx --from sibyl-research sibyl-mcp
```

For clients that accept an MCP server configuration:

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "uvx",
      "args": ["--from", "sibyl-research", "sibyl-mcp"]
    }
  }
}
```

No search or model key is required for `gather_bundle`, `gather_sources`, `quick_search`, or `read_url`.

For repeatable production retrieval, opt into Tavily explicitly:

```bash
export SIBYL_SEARCH_PROVIDER=tavily
export TAVILY_API_KEY=tvly-...
```

Sibyl sends general-web queries to Tavily's basic Search API and falls back to the keyless DuckDuckGo/Mojeek/Yahoo chain if a Tavily request fails or returns no results. This setting is never enabled merely because a key exists. Tavily bills each request according to its own plan, and one `gather_bundle()` call can issue more than one focused query; review the [Tavily Search API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/search) before enabling it.

Academic and DOI-oriented questions also query the public Crossref REST API. No key is required. Set `CROSSREF_MAILTO=you@example.com` to identify your client to Crossref's polite pool; see [Crossref's API guidance](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/).

Verify the isolated installation:

```bash
uvx --from sibyl-research sibyl-mcp --version
uvx --from sibyl-research sibyl-mcp --list-tools
```

### 3. Use the keyless CLI or Python API

```bash
sibyl gather "Who was the Serbian quarterfinalist in the 2018 Madrid Open?"
sibyl gather "Python 3.14 release date" --format json
```

```python
import asyncio
from sibyl import gather_bundle

bundle = asyncio.run(gather_bundle("Python 3.14 release date"))
if bundle.status == "ok":
    for source in bundle.sources:
        print(source.title, source.url)
```

### 4. Give the host a retrieval policy

```text
Use Sibyl's gather_bundle tool for factual research.
Check bundle status before answering. Treat search_snippet content as a lead,
not full evidence. Do not count sources with the same content_cluster_id as
independent corroboration. Follow diagnostics.recommended_action: synthesize,
refine_query, decompose_query, retry, or revise_request. Cite passage citation_id
values. If the bundle does not contain the answer, retrieve again or say it was not found.
```

That policy matters more than a long system prompt: it tells the agent when evidence is usable, what counts as independent support, and when to abstain.

## MCP profiles and tools

Without an LLM credential, the default `auto` profile exposes only the four keyless retrieval tools. A configured `[report]` installation exposes report tools automatically. Finance tools remain explicit so an agent does not pay the context cost for unrelated capabilities.

Use `sibyl-mcp --profile keyless|report|finance|full` to select a surface directly. Missing extras or credentials fail at startup with an actionable installation message.

| Group | Tool | Result | LLM key |
|---|---|---|---:|
| Retrieval | `gather_bundle(query)` | Structured SourceBundle 1.6 | No |
| Retrieval | `gather_sources(query)` | Readable `[Source N]` evidence blocks | No |
| Retrieval | `quick_search(query)` | Titles, URLs, and search snippets | No |
| Retrieval | `read_url(url)` | Clean text from one public URL | No |
| Research | `research(query, depth)` | Synthesized and cited report | Yes |
| Research | `analyze(text, question)` | Analysis of supplied text | Yes |
| Analysis | `compare(items, query)` | Researched comparison | Yes |
| Analysis | `swot(subject)` | Researched SWOT | Yes |
| Analysis | `timeline(topic)` | Researched event timeline | Yes |
| Data | `trends(keywords)` | Google Trends series and related queries | No |
| Data | `fetch_market_data(symbols)` | Yahoo Finance market summary | No |
| Data | `chart(symbols)` | Local PNG price chart | No |
| Output | `save_report(format)` | PDF and/or Markdown from the last report | After `research()` |

## Recommended agent workflow

For a non-trivial question, one broad retrieval call is rarely enough. Use a small evidence loop:

1. Break the question into focused, searchable claims.
2. Call `gather_bundle()` for the first claim.
3. Check `status`, `diagnostics.evidence_sufficiency`, and `sufficiency_reasons`.
4. Inspect `content_origin`; use `search_snippet` only to plan another query.
5. Compare `content_cluster_id` before treating domains as independent corroboration.
6. Retrieve again for gaps, conflicts, or missing primary evidence.
7. Synthesize only from returned passages and retain their `citation_id` values.

This makes missing evidence observable. It also prevents several different websites carrying the same syndicated article from looking like independent confirmation.

## SourceBundle 1.6

`gather_bundle()` returns a typed MCP structured result. This abridged example shows the fields a consumer normally uses:

```json
{
  "schema_version": "1.6",
  "bundle_id": "sb_<bundle-hash>",
  "query": "example query",
  "status": "ok",
  "sources": [
    {
      "source_id": "S1",
      "url": "https://example.com/article",
      "title": "Example article",
      "retrieved_at": "2026-07-14T00:00:00+00:00",
      "published_at": "2026-07-13",
      "published_at_method": "json_ld_date_published",
      "content_origin": "direct_fetch",
      "content_cluster_id": "cc_<content-hash>",
      "relevance_score": 0.91,
      "quality_score": null,
      "evidence": [
        {
          "passage_id": "P1",
          "citation_id": "sb_<bundle-hash>/S1/P1",
          "text": "The selected evidence passage...",
          "content_hash": "<sha256>",
          "start_char": 120,
          "end_char": 480,
          "score": 0.93
        }
      ]
    }
  ],
  "diagnostics": {
    "ranking_method": "lexical_v1",
    "query_term_coverage": 0.75,
    "max_source_query_term_coverage": 0.67,
    "substantive_sources": 3,
    "independent_content_clusters": 2,
    "evidence_sufficiency": "sufficient",
    "sufficiency_reasons": [],
    "search_queries": ["Python 3.14 release date"],
    "search_providers": ["tavily", "wikipedia"],
    "metadata_fallbacks": 0,
    "query_complexity": "single_step",
    "recommended_action": "synthesize"
  },
  "error": ""
}
```

Consumer rules:

- Require schema major version `1`; allow additive fields in later minor versions.
- Check `status` before reading evidence. Only `ok` is synthesis-ready.
- Require `diagnostics.recommended_action == "synthesize"` before synthesis. Decompose dependent fact chains into atomic `gather_bundle()` calls.
- Treat `citation_id` as bundle-scoped. Persist it with `bundle_id`, `source_id`, and `passage_id`.
- Treat source and passage scores as relevance signals, not truth probabilities.
- Treat `published_at` as publisher-supplied metadata, not an independently verified date.
- Treat identical `content_cluster_id` values as the same underlying content even across domains.
- Treat `quality_score: null` as unassessed, never as zero.
- Ignore unknown additive fields and preserve unknown diagnostic reason strings.

The complete rules and machine-validated fixture are in the [consumer contract](https://github.com/chriswu727/sibyl/blob/main/docs/source-bundle-1.6.md) and [`source_bundle_1_6.example.json`](https://github.com/chriswu727/sibyl/blob/main/docs/source_bundle_1_6.example.json).

## Retrieval behavior

For each focused query, Sibyl:

1. Searches the configured general-web provider plus complementary public sources with provider pacing and bounded waits. The default path uses independent keyless failover; Tavily is an explicit opt-in.
2. Fetches public pages and extracts readable content.
3. Uses Wikipedia to expand thin result coverage; Jina Reader rendering is available only when `render_thin_pages=true` is explicitly requested.
4. Canonicalizes URLs, removes duplicates, and clusters syndicated text.
5. Ranks sources and passages for query relevance.
6. Prefers independent content clusters before filling remaining source slots.
7. Returns selected passages, provenance, and retrieval diagnostics.

Ranking is local by default:

| Ranker | Setup | Behavior |
|---|---|---|
| `lexical` | Built in | Deterministic, dependency-free relevance ranking |
| `flashrank` | `pip install 'sibyl-research[rerank]'` | Optional local cross-encoder; falls back explicitly to lexical |
| `none` | Built in | Preserves retrieval order and returns `null` scores |

Matching `gather_bundle()` and `gather_sources()` calls share in-flight work and reuse successful retrievals for 30 seconds within one MCP process. Failed retrievals are not cached.

## Truthful failure states

SourceBundle never turns a retrieval failure into a completed-looking answer.

| Status | Meaning | Consumer action |
|---|---|---|
| `ok` | Evidence passed the deterministic sufficiency checks | Inspect diagnostics and synthesize carefully |
| `insufficient_evidence` | Evidence is limited or insufficient; returned sources are leads | Refine the query or abstain |
| `invalid_request` | Query or parameters are invalid | Fix the request |
| `failed` | Search or retrieval failed | Retry later or use another source path |

`evidence_sufficiency` is a deterministic retrieval signal based on evidence volume, lexical coverage, domain diversity, and independent content clusters. It is not a correctness or credibility score.

The checks also require named query anchors to appear in the selected evidence. A question that asks for a specific outcome in a future year is not marked synthesis-ready merely because forecasts or similarly named events were retrieved. Historical role questions require a local statement connecting the role to the requested year or a covering tenure range; unrelated mentions elsewhere on the same page do not count.

## Network safety

Sibyl retrieves untrusted URLs, so the fetch path is deliberately constrained:

- only public HTTP(S) destinations are allowed;
- URL credentials and non-web ports are rejected;
- DNS results are validated and pinned before connecting;
- every redirect destination is validated again;
- local, private, loopback, link-local, and otherwise non-global addresses are blocked;
- decompressed response bodies are capped at 2 MiB;
- Jina rendering has bounded concurrency and request start-rate limits.

When Jina Reader is used, the target URL is sent to that external service. It is disabled by default in every workflow. Set `render_thin_pages=true` for retrieval or `js_render: true`/`--js-render` for one-shot research only when this disclosure is acceptable.

## Optional one-shot research

The `research()` tool and `sibyl` CLI run a full pipeline:

```text
decompose → search → scrape → deduplicate → rank → synthesize → verify → report
```

They require an LLM provider key or a configured local/API-compatible backend. Sibyl auto-detects common provider environment variables:

Install the report capability first:

```bash
python -m pip install 'sibyl-research[report]'
```

| Provider | Environment variable |
|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| ZhipuAI / GLM | `ZHIPUAI_API_KEY` |

Run the MCP server with one-shot tools enabled:

```bash
claude mcp add sibyl -e DEEPSEEK_API_KEY=sk-... -- \
  uvx --from 'sibyl-research[report]' sibyl-mcp --profile report
```

Or use the CLI:

```bash
export DEEPSEEK_API_KEY=sk-...

sibyl research "Canadian housing market outlook" --depth 2
sibyl research "加拿大移民政策变化" --language zh --md --output reports/
```

Market symbols and charts also require `sibyl-research[finance]`. The historical `sibyl "query"` report form remains supported, but the explicit `gather` and `research` subcommands make the model and network boundary clearer.

Depth controls the amount of decomposition and review work:

| Depth | Intended use |
|---:|---|
| `1` | Quick research with tight query and source limits |
| `2` | Standard research with review and claim verification |
| `3` | Deeper gap-filling and prediction scenarios |

For role-specific model routing, point `SIBYL_CONFIG` at a YAML file:

```yaml
providers:
  - model: deepseek/deepseek-v4-flash
    api_key: sk-...
    role: general
  - model: anthropic/claude-sonnet-4-20250514
    api_key: sk-ant-...
    role: synthesis
  - model: anthropic/claude-sonnet-4-20250514
    api_key: sk-ant-...
    role: verify

search_engine: all
max_sources: 15
max_depth: 2
reranker: lexical
```

Do not commit real API keys. The keyless retrieval tools remain available when no LLM provider is configured.

## Evaluation and reproducibility

The default CI suite is network-free and runs on Python 3.10, 3.11, and 3.12. It includes unit tests, fixed retrieval-ranking cases, full retrieval-pipeline fixtures, SourceBundle contract checks, and a source-quality control baseline.

```bash
python -m unittest discover tests -v
python scripts/eval_retrieval.py --ranker lexical
python scripts/eval_retrieval_pipeline.py --ranker lexical
python scripts/eval_source_quality.py
python scripts/eval_live_retrieval.py --repeats 3 \
  --output evals/results/live-retrieval-YYYY-MM-DD.json
```

The first three checks are deterministic and network-free. `eval_live_retrieval.py` runs 66 natural-language and adversarial questions against the public web without an LLM. Metric version 2 measures answer coverage, safe trap handling, run-to-run stability, the fraction of answerable runs that are both `ok` and contain the expected answer, the precision of `ok` states on answerable questions, and p50/p95 latency. It also records the configured and actual search-provider paths. Live results vary with public search and publisher availability and must be saved with their date when used as launch evidence.

The latest formal keyless three-repeat run is retained in [`evals/results/live-retrieval-keyless-post-crossref-2026-07-21.json`](evals/results/live-retrieval-keyless-post-crossref-2026-07-21.json). Answer coverage improved from 75.9% in the [original baseline](evals/results/live-retrieval-2026-07-21.json) to 85.2%, trap safety remained 100%, and p95 latency was 11.0 seconds. A subsequent [105-request Tavily pilot](evals/results/live-retrieval-tavily-pilot-2026-07-21.json) reached 88.9% answer coverage and 10.7-second p95 latency. Recomputed under metric version 2, Tavily produced synthesis-ready answer evidence on 70.4% of answerable cases with 92.7% ready-state precision; the gates are 75% and 95%. Sibyl therefore remains a public beta until a dated three-repeat run clears every threshold; offline checks or a single-repeat pilot are not release evidence.

The repository also contains an exploratory 30-question SimpleQA comparison. In the measured concurrent run, a host model reasoning over `gather_sources()` answered 26/30 correctly, while the configured one-shot model answered 5/30. The experiment is small, agent-graded, and sensitive to keyless search throttling; read the [full method and caveats](https://github.com/chriswu727/sibyl/blob/main/docs/EVAL_HOST_CLAUDE.md) rather than treating it as a broad benchmark.

## Known limitations

- Keyless search engines and public websites can throttle, block, or change behavior.
- Tavily improves the operational search path but is optional, credentialed, and usage-billed by Tavily; it does not make publisher pages or the wider web deterministic.
- Retrieval relevance does not establish factual truth or source credibility.
- `quality_score` is intentionally unpopulated while the production credibility model is still under evaluation.
- Publication dates are extracted from explicit page metadata and may be wrong at the publisher.
- Near-duplicate clustering is content-based and can miss heavily rewritten syndication.
- The one-shot pipeline is only as capable and reliable as its configured LLM.
- Live web results are not deterministic; use the offline fixtures for regression testing.

## Development

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[all]'
python -m unittest discover tests -v
```

Useful project documents:

- [SourceBundle 1.6 consumer contract](https://github.com/chriswu727/sibyl/blob/main/docs/source-bundle-1.6.md)
- [Performance notes](https://github.com/chriswu727/sibyl/blob/main/docs/PERFORMANCE.md)
- [MCP architecture notes](https://github.com/chriswu727/sibyl/blob/main/docs/MCP_ARCHITECTURE.md)
- [Launch readiness and competitive position](https://github.com/chriswu727/sibyl/blob/main/docs/LAUNCH_READINESS.md)
- [MCP installation reference](https://github.com/chriswu727/sibyl/blob/main/llms-install.md)
- [Privacy](https://github.com/chriswu727/sibyl/blob/main/PRIVACY.md)
- [Security](https://github.com/chriswu727/sibyl/blob/main/SECURITY.md)
- [Contributing](https://github.com/chriswu727/sibyl/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/chriswu727/sibyl/blob/main/CHANGELOG.md)
- [Release runbook](https://github.com/chriswu727/sibyl/blob/main/docs/RELEASING.md)

## License

[MIT](https://github.com/chriswu727/sibyl/blob/main/LICENSE)
