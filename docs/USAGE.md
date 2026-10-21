<!-- mcp-name: io.github.chriswu727/sibyl -->

<div align="center">

# Sibyl usage reference

This reference follows the `main` branch. See the [landing page](../README.md)
for release status and the [architecture guide](MCP_ARCHITECTURE.md) for current
execution limits.

**Deep research through MCP — persistent investigations and traceable reports.**

Sibyl is a local MCP server for public-web research and cited reports. Its CLI and Python API also expose evidence retrieval and model-backed research. A shared `SourceBundle` pipeline searches, extracts, ranks, and preserves provenance. Persistent jobs support host-assisted writing or an explicitly configured autonomous model; Sibyl is not a hosted service.

[![Tests](https://github.com/chriswu727/sibyl/actions/workflows/tests.yml/badge.svg)](https://github.com/chriswu727/sibyl/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sibyl-research?color=blue)](https://pypi.org/project/sibyl-research/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-0f766e)](../pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chriswu727/sibyl/blob/main/LICENSE)

[Install](#quick-start) · [SourceBundle contract](https://github.com/chriswu727/sibyl/blob/main/docs/source-bundle-1.6.md) · [Evidence-loop contract](https://github.com/chriswu727/sibyl/blob/main/docs/evidence-loop-1.0.md) · [Privacy](https://github.com/chriswu727/sibyl/blob/main/PRIVACY.md) · [Security](https://github.com/chriswu727/sibyl/blob/main/SECURITY.md) · [Changelog](https://github.com/chriswu727/sibyl/blob/main/CHANGELOG.md)

</div>

## What Sibyl is

Sibyl combines a research workflow with an inspectable evidence boundary:

> Research produces a report. Evidence makes that report inspectable—not automatically true.

The core path is `gather_bundle()`: a keyless retrieval tool that returns a typed, versioned `SourceBundle`. It does not generate an answer. The bundle carries source and passage identities, hashes, offsets, retrieval and publication metadata, content-origin labels, near-duplicate clusters, relevance signals, and explicit failure states. For dependent questions, `gather_evidence()` adds a bounded, auditable sequence of atomic retrieval steps while leaving planning and synthesis to the host agent.

Persistent jobs are the report-oriented workflow. The existing `research()` tool remains a blocking configured-model interface. Both use the same retrieval pipeline. Report quality depends on evidence and model behavior; no current-model comparative quality claim has been established.

| Mode | Tool | API key | Who reasons? | Best for |
|---|---|---:|---|---|
| Structured retrieval | `gather_bundle()` | No | Your agent | Pipelines, contracts, machine-readable citations |
| Readable retrieval | `gather_sources()` | No | Your agent | Conversational hosts and manual inspection |
| Bounded evidence loop | `gather_evidence()` | No | Your agent | Multi-step questions and auditable follow-ups |
| Host-assisted report | `start_research(mode="host")` | No | Your agent | Persistent research and Markdown/JSON delivery |
| Autonomous report | `start_research(mode="autonomous")` | Configured backend | Sibyl's model | Background research, progress, explicit resume |
| Blocking report | `research()` | Configured backend | Sibyl's model | Existing one-call integrations |

## Quick start

### 1. Install the current source checkout

This guide describes unreleased `main`. The published 0.3.0 package does not
provide the current CLI subcommands, capability profiles, or evidence-loop API.

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Sibyl requires Python 3.11 or newer. The default installation is the lightweight keyless retrieval product. Optional features are installed explicitly:

```bash
python -m pip install -e '.[report]'   # experimental LLM report + PDF
python -m pip install -e '.[finance]'  # market data, trends, charts
python -m pip install -e '.[rerank]'   # local cross-encoder ranking
python -m pip install -e '.[pdf,monitor]'  # text-PDF reads and local operational endpoints
python -m pip install -e '.[all]'      # every optional capability
```

### 2. Add the keyless MCP server

For Claude Code:

```bash
claude mcp add sibyl -- /absolute/path/to/sibyl/.venv/bin/sibyl-mcp --profile keyless
```

For clients that accept an MCP server configuration, replace the absolute path:

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "/absolute/path/to/sibyl/.venv/bin/sibyl-mcp",
      "args": ["--profile", "keyless"]
    }
  }
}
```

No search or model key is required for the twelve keyless tools, including host
research jobs, evidence primitives, and Markdown/JSON export.

For an optional configured search-provider path, opt into Tavily explicitly:

```bash
export SIBYL_SEARCH_PROVIDER=tavily
export TAVILY_API_KEY=tvly-...
```

Sibyl sends general-web queries to Tavily's basic Search API and falls back to the keyless DuckDuckGo/Mojeek/Yahoo chain if a Tavily request fails or returns no results. This setting is never enabled merely because a key exists. Tavily bills each request according to its own plan, and one `gather_bundle()` call can issue more than one focused query; review the [Tavily Search API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/search) before enabling it.

Academic and DOI-oriented questions also query the public Crossref REST API. No key is required. Set `CROSSREF_MAILTO=you@example.com` to identify your client to Crossref's polite pool; see [Crossref's API guidance](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/).

Verify the isolated installation:

```bash
sibyl-mcp --version
sibyl-mcp --profile keyless --list-tools
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

MCP report/chart exports use `SIBYL_OUTPUT_ROOT` (default `reports/`).
`save_report(output_dir=...)` selects a relative subdirectory of that root, not
an absolute path. Stateful report operations reject overlapping calls.
The CLI's explicit `--output` option remains an operator-controlled path.

Without an LLM credential, the default `auto` profile exposes twelve tools: five evidence primitives, six research-job operations, and report export. Host jobs and Markdown/JSON export need no model key. Autonomous mode requires `[report]` and a configured model; PDF requires the rendering extra. A configured `[report]` installation also exposes the legacy model tools. Finance tools remain explicit.

Use `sibyl-mcp --profile keyless|report|finance|full` to select a surface directly. Missing extras or credentials fail at startup with an actionable installation message.

| Group | Tool | Result | LLM key |
|---|---|---|---:|
| Jobs | `start_research(question)` | Persistent host or autonomous job | Autonomous only |
| Jobs | `get_research(job_id)` | Progress, evidence, revision, report; omit ID to list | No |
| Jobs | `continue_research(job_id, query)` | Focused host search or explicit autonomous resume | Autonomous only |
| Jobs | `complete_research(job_id, revision, ...)` | Immutable host-authored report | No |
| Jobs | `assess_research(job_id, revision, question_id, ...)` | Host assessment with stable support/challenge links | No |
| Jobs | `cancel_research(job_id, delete=false)` | Cancel or explicitly remove checkpoint | No |
| Retrieval | `gather_evidence(question)` | Bounded multi-step evidence loop | No |
| Retrieval | `gather_bundle(query)` | Structured SourceBundle 1.6 | No |
| Retrieval | `gather_sources(query)` | Readable `[Source N]` evidence blocks | No |
| Retrieval | `quick_search(query)` | Titles, URLs, and search snippets | No |
| Retrieval | `read_url(url)` | Public HTML or optional text PDF, with extraction warnings | No |
| Research | `research(query, depth)` | Synthesized and cited report | Yes |
| Research | `analyze(text, question)` | Analysis of supplied text | Yes |
| Analysis | `compare(items, query)` | Researched comparison | Yes |
| Analysis | `swot(subject)` | Researched SWOT | Yes |
| Analysis | `timeline(topic)` | Researched event timeline | Yes |
| Data | `trends(keywords)` | Google Trends series and related queries | No |
| Data | `fetch_market_data(symbols)` | Yahoo Finance market summary | No |
| Data | `chart(symbols)` | Local PNG price chart | No |
| Output | `save_report(format, job_id)` | Markdown, JSON, or optional PDF; omit ID for last one-shot report | No |

## Persistent research jobs

`start_research` defaults to `mode="host"`. Supply up to five sub-questions,
then call `continue_research` with focused queries. Responses include selected
passages, retrieval timestamps, citation IDs, attempts, progress events, and
limitations. Primary and scholarly candidates are selected before encyclopedia,
community, and metadata-only sources, with coverage across retrieval rounds.
Reviewed evidence is pinned; duplicate URLs and content clusters do not establish
independent corroboration. Type labels are heuristics and do not certify authority.
Search snippets are leads and are excluded from report source selection.

### Evidence-led investigation

Choose `source_policy="academic"` to enable scholarly discovery through the existing
search/Crossref path, or `primary` (default) for general research. Supply up to
sixteen `preferred_domains` such as a relevant official documentation host. Match
the source type to the claim: papers for methods and experiments, current official
docs for product capabilities and pricing. Preprints and publisher-platform entries
are not automatically peer reviewed. Crossref metadata does not count as full text.

`research_plan.questions` provides stable IDs (`Q1` through `Q5`, or `Q0` when no
sub-questions were supplied), explicit gaps, and suggested follow-up searches.
Link each `continue_research` to a `question_id`; a single question is inferred.
Use `purpose="counterevidence"` to seek replication, limitations, or conflicting
results, not merely more confirming material. An empty or failed retrieval does not
establish the absence of contrary evidence and does not satisfy that gate.

To follow a paper citation or source link, call:

```text
continue_research(job_id="...", question_id="Q1", query="Inspect the original experiment",
                  url="https://public.example/paper", purpose="follow_citation",
                  parent_passage_id="the previously inspected passage ID")
```

URL reads use the same safe-network boundary, timeout, worker, and retrieval budgets
as search. Extracted text is bounded to 16,000 characters and split into hashed
passages with offsets. Thin/unsupported pages add no evidence. Text PDFs require
the `[pdf]` extra and a POSIX host; page markers and truncation warnings are retained.
Automatic reference-link crawling is not implemented. Parent links
are host-declared; the server does not prove that a hyperlink existed in the parent.
The source catalog retains passage IDs even when a source falls out of the selected
report window. Inspect one with `get_research(job_id="...", passage_id="...")`.

After reading the evidence, call `assess_research` for each question:

```text
assess_research(job_id="...", revision=the_current_revision, question_id="Q1",
                status="disputed", supporting_passage_ids=["bundle/S1/P1"],
                challenging_passage_ids=["other/S1/P2"],
                conclusion="The results differ under these experimental conditions...",
                counterevidence_note="Compared the original study and the replication...")
```

Use `supported`, `disputed`, or `unresolved`. A supported review needs supporting
passages; disputed needs both sides; unresolved needs an explanation, not invented
evidence. Only cite passages actually inspected. The server checks link integrity,
revision freshness, and structural completeness, **not semantic correctness**.
New evidence for a reviewed question makes that review stale. Linked sources stay
in the report window; more than twelve distinct reviewed sources or contradictory
duplicate snapshots require narrowing the review rather than silently losing links.

After investigating and reviewing all questions, get the latest job and submit:

```text
complete_research(
  job_id="the returned ID",
  revision=the_current_revision,
  summary="An evidence-backed answer [Source 1].",
  key_findings=["A supported finding [Source 2]."],
  analysis="Interpretation, explicitly distinguished from sourced facts.",
  limitations=["A specific unresolved question."]
)
save_report(job_id="the returned ID", format="json")
```

Summary and each finding must cite a selected `[Source N]`. Numbering may change
after retrieval; the revision check prevents stale submissions. Citation target
validation is not entailment checking. Host reports state
`verification_scope="citation_targets_only"`; the host owns semantic judgment.
Completed reports are immutable. Start another job for a new edition.
New host jobs reject completion when required reviews or counterevidence checks are
missing. `allow_incomplete=true` explicitly permits delivery with every remaining
gap recorded in report limitations and `report.status="partial"`; it is not a quality
pass. The job remains `status="completed"` because its report is finalized and immutable.
Markdown/PDF exports visibly label partial reports. Existing pre-upgrade
checkpoints remain readable and retain their former completion behavior. JSON exports
preserve exact selected passages, hashes, offsets, the review ledger, and investigation
steps. Markdown includes reviews and compact source references instead of dumping
large source excerpts. PDF retains the narrative and limitations; use JSON for the
complete machine-readable evidence package.

Markdown, PDF, and plain-text reports separate **Cited Sources** from **Uncited
Background** using citation markers in the report. Source numbers stay unchanged,
including gaps between numbers; JSON retains the original source inventory and
evidence. Being cited is not a verdict that a source supports a claim.

HTML reads fall back from placeholder document titles to metadata or headings.
Prose whitespace is normalized before the character limit is applied. Excessive
horizontal whitespace runs (80 or more characters) in preformatted text are
normalized with a warning; ordinary code spacing remains. Extracted text is not
a verbatim code copy. HTML truncation is explicitly reported, just like PDF
truncation; later sections may still be missing from a bounded read.

### Multi-round retrieval

Queries with explicit years or current/latest intent prioritize matching title,
publication-date, or URL-path periods over undated and mismatched candidates.
This period hint applies to ranked retrieval and the unpinned report window;
`ranker="none"` preserves retrieval order, and reviewed evidence stays pinned.
Calendar-year queries use the publication year for fiscal-year titles; explicit
fiscal-year queries match the title instead. These are routing
hints, not guarantees of relevance or recency; inspect the source and reporting
period. Undated sources are not assigned invented publication dates.

For multi-domain `site:` queries, up to two focused searches fill missing requested
domains. Their queries and failure counts are recorded, and one result per present
domain is placed ahead of repeated hits before applying the URL fetch limit. This
does not guarantee balanced coverage when the search provider has no matching results.

Direct source-read failures retain the fetch reason (such as `HTTP 403` or an
unsupported content type) in the attempt's `error`. No third-party rendering is
silently enabled. PDF failures (missing extra, encryption, no readable text,
malformed input, or a resource limit) are explicit rather than inferred evidence.

Multi-round research is an evidence-feedback loop, not a fixed list of searches.
In host mode the calling agent supplies the reasoning; Sibyl retains its state,
recommends outstanding follow-ups, and enforces the existing retrieval budget:

1. Retrieve initial evidence for a planned question.
2. Review the result with `assess_research`. If the evidence leaves a gap, use
   `status="unresolved"` and `follow_up_queries=["a focused replication query"]`.
3. Read `research_plan.suggested_queries`, then call `continue_research` with a
   chosen query and `rationale="Resolve the missing replication evidence"`.
   Choose `purpose="counterevidence"` when investigating contrary evidence.
4. Inspect the new passages and reassess. Finish only when the required reviews
   are ready, or explicitly deliver an incomplete report with its gaps.

Each attempt retains its round number, prior evidence revision, prior gaps,
prior review status, and optional host rationale in the checkpoint and JSON report.
Reviews accept at most four follow-up queries of up to 1,000 characters. Completed
searches are omitted from suggestions for that question (case/whitespace normalized);
failed attempts remain retryable and consume budget. Explicit repeats are allowed
for freshness checks. Empty suggestions do not prove exhaustive coverage.
Budget exhaustion returns no more suggestions and asks for review or explicit
limitations; it never silently grants a quality pass. Resuming a checkpoint does
not reset the budget or lose the earlier rounds.

Autonomous mode already uses a configured model to reflect on a draft, generate
follow-up queries, retrieve, and re-synthesize when usable evidence changes. It
allows at most two reflection rounds, also bounded by retrieval/model/time budgets;
fast mode skips reflection. No additional usable evidence stops the loop with a
limitation. Model reflection remains fallible and incurs configured-provider usage.
Offline tests mock model calls and verify the loop, not research quality.

### Comparing with ordinary web search

Engineering gates are not evidence of superior research quality. For a quality A/B,
use the same host model, dated questions, model effort, search provider, and overall
time/call budget. Compare ordinary search with the persistent deep workflow on
predeclared measures: important-question coverage, claim-to-source support, original
source discovery, counterevidence, correct abstention, and cost/latency. Record both
outputs before judging; keep evidence fixtures separate from the evaluation questions.
Repeat across technical explanations, current-product comparisons, and contested
research, with reviewers blind to which workflow produced each report where feasible.

The offline deep-research tests establish selection, provenance, and readiness
contracts only. No controlled same-model report-quality A/B has yet established a
Sibyl advantage over ordinary web search. Model-backed evaluations require explicit
operator approval; do not spend a subscription/API budget implicitly.

For `mode="autonomous"`, the optional report installation and configured model
are required. The server runs research asynchronously; poll `get_research`.
It decomposes the question, retrieves shared source bundles, drafts, performs
one bounded coverage-review round by default, and checks report blocks against
the cited evidence. Unsupported or unassessed blocks are visibly marked.
Model verification is fallible and does not establish source truth.

`continue_research(job_id=...)` with no query explicitly resumes an autonomous
job. Saved evidence, completed plan, and consumed budgets survive restart;
unfinished synthesis may run again and incur additional model charges.
There is no automatic restart of interrupted work and no exactly-once model call
guarantee. Source pages remain untrusted data, including during resume.

| Limit | Default / ceiling |
|---|---|
| Retrieval attempts | 8 per job, caller-selectable from 1 to 12 |
| Selected report sources | 12; one-shot configuration can select up to 24 |
| Simultaneous job workers | 2 |
| Active execution time | 600 seconds across continuations; host thinking time excluded |
| Autonomous model calls | 40 attempts, including retries and verification |
| Reserved completion tokens | 60,000 across attempts; not an exact charge or input-token limit |
| Checkpoint records | 64; explicit deletion required when full |
| Checkpoint payload | 1 MiB per job, plus 64 KiB reserved for failure recovery metadata |
| SQLite database | 16,384 pages; approximately 64 MiB with default page size |
| Progress history | Last 80 events per job |
| PDF fetch / parsing | 2 MiB; first 40 pages; at most 60,000 characters; 6-second worker deadline |
| PDF workers | 2 per event loop; 512 MiB memory threshold; see platform details in SECURITY.md |
| Export file | 16 MiB |
| Exports per output directory | 256 files / 128 MiB; explicit retention required |

A retrieval attempt can itself issue multiple search-provider requests. These
limits do not promise an exact dollar budget; external search/model pricing,
input tokens, provider retries, and already-dispatched requests still matter.
Cancelled requests may still be billed by the provider.

The private database lives in `SIBYL_OUTPUT_ROOT/.research/jobs.sqlite3`.
Only one runtime may own that directory at a time. Cancellation preserves the
checkpoint; `cancel_research(job_id=..., delete=true)` removes it and reclaims
history capacity, but does not remove exported files. SQLite deletion is not
guaranteed forensic erasure. See [Privacy](../PRIVACY.md).

JSON, Markdown, and PDF use the same `ResearchReport` contract. JSON retains
source passage IDs, timestamps, limitations, verification scope, and model-call
counters. PDF export requires `[report]`; text-PDF input separately requires `[pdf]`.
If PDF-only export fails, the MCP call returns a tool error. With `format="both"`,
a PDF failure is disclosed while a successful Markdown artifact is retained.
Input is extracted in an isolated child process without inherited credentials,
then split into the same hashed passages as HTML. Scanned pages have no OCR;
text order, columns, tables, and formulas still need source inspection.
The 40-page and character limits are visible limitations, not full-document claims.

Monitoring setup and recovery procedures are in [Operations](OPERATIONS.md).

## Recommended agent workflow

For a non-trivial question, one broad retrieval call is rarely enough. `gather_evidence()` makes the loop explicit and bounded:

1. Call `gather_evidence(question="...")`. Atomic questions may become `ready` immediately; dependent fact chains return `decompose_query` without wasting a broad retrieval.
2. Continue with `gather_evidence(loop_id="...", query="one atomic query")` and follow `next_action`. Repeated and still-compound follow-ups are rejected.
3. Inspect each returned `current_step.bundle`, including `content_origin`, `content_cluster_id`, sufficiency reasons, and passage citation IDs. Historical steps remain as compact summaries so evidence is not duplicated in the host context.
4. Stop after at most four retrieval calls. When the evidence covers the original question, call `gather_evidence(loop_id="...", finish=true, supporting_step_ids=["E1", "E2"])`.
5. Synthesize only when the loop returns `status="ready"`; selected supporting steps must individually have `recommended_action="synthesize"`.

The loop expires after ten minutes and never invokes MCP sampling or a hidden Sibyl model. The calling host remains responsible for semantic planning and synthesis; Sibyl enforces the retrieval budget and evidence-state checks. For one focused query, `gather_bundle()` remains the simpler primitive.

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
    "recommended_action": "synthesize",
    "refinement_searches": 0,
    "refinement_failures": 0
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
| `flashrank` | `pip install -e '.[rerank]'` | Optional local cross-encoder; falls back explicitly to lexical |
| `none` | Built in | Preserves retrieval order and returns `null` scores |

Matching `gather_evidence()`, `gather_bundle()`, and `gather_sources()` steps share in-flight work and reuse successful retrievals for 30 seconds within one MCP process. Failed retrievals are not cached.

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

When Jina Reader is used, the target URL is sent to that external service. It is disabled by default in every workflow. Set `render_thin_pages=true` for retrieval or `js_render: true`/`--js-render` for experimental report generation only when this disclosure is acceptable.

## Configured-model one-shot reports

The compatible `research()` tool and `sibyl` CLI run a blocking pipeline:

```text
plan → shared source bundles → draft → coverage review / follow-up → report-wide audit → report
```

They require an LLM provider key or a configured local/API-compatible backend. Sibyl auto-detects common provider environment variables:

Install the report capability first:

```bash
python -m pip install -e '.[report]'
```

| Provider | Environment variable |
|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| ZhipuAI / GLM | `ZHIPUAI_API_KEY` |

Run the MCP server with one-shot tools enabled:

Select a model explicitly through `SIBYL_MODEL` / `SIBYL_API_KEY` (and
`SIBYL_API_BASE` when needed), or `SIBYL_CONFIG` with role-specific providers.
Provider autodetection defaults exist for compatibility; they are not a claim
that those models are current or optimal. Measure a candidate model on the same
research tasks and budget before changing defaults.

```bash
claude mcp add sibyl -- \
  /absolute/path/to/sibyl/.venv/bin/sibyl-mcp --profile report
```

Or use the CLI:

```bash
sibyl research "Canadian housing market outlook" --depth 2
sibyl research "Changes to Canadian immigration policy" --language en --md --output reports/
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

Supply credentials through the host's secret/environment mechanism; do not put
real keys in command arguments or committed files. Model IDs above illustrate
role routing, not a recommendation or current availability guarantee. For YAML,
use provider-supported environment credentials or a private configuration file;
`${VARIABLE}` substitution is not performed by Sibyl's YAML loader.
The keyless tools remain available when no LLM provider is configured.

## Evaluation and reproducibility

The functional CI suite is network-free and runs on Python 3.11, 3.12, and 3.13. It includes unit tests, fixed retrieval-ranking cases, full retrieval-pipeline fixtures, SourceBundle contract checks, and a source-quality control baseline. Dependency security audits separately query OSV.

```bash
python -m unittest discover tests -v
python scripts/eval_retrieval.py --ranker lexical
python scripts/eval_retrieval_pipeline.py --ranker lexical
python scripts/eval_source_quality.py
python scripts/eval_live_retrieval.py --repeats 3 \
  --output evals/results/live-retrieval-YYYY-MM-DD.json
```

The first four commands check offline contracts without public-web or model calls;
transport tests use local loopback only. `eval_live_retrieval.py` runs 66
natural-language and adversarial questions against the public web without an LLM.
Metric version 2 measures answer coverage, safe trap handling, run-to-run stability,
answerable runs containing both `ok` and expected evidence, ready-state precision,
and p50/p95 latency. It records configured and actual provider paths. Save dated
results; public search and publisher availability vary between runs.

The latest formal keyless three-repeat run is retained in [`evals/results/live-retrieval-keyless-post-crossref-2026-07-21.json`](../evals/results/live-retrieval-keyless-post-crossref-2026-07-21.json). Answer coverage improved from 75.9% in the [original baseline](../evals/results/live-retrieval-2026-07-21.json) to 85.2%, trap safety remained 100%, and p95 latency was 11.0 seconds. A subsequent [105-request Tavily pilot](../evals/results/live-retrieval-tavily-pilot-2026-07-21.json) reached 88.9% answer coverage and 10.7-second p95 latency. Recomputed under metric version 2, Tavily produced synthesis-ready answer evidence on 70.4% of answerable cases with 92.7% ready-state precision; the gates are 75% and 95%. Sibyl therefore remains a public beta until a dated three-repeat run clears every threshold; offline checks or a single-repeat pilot are not release evidence.

The repository also contains an exploratory 30-question SimpleQA comparison: the measured host workflow answered 26/30 correctly and the configured one-shot workflow 5/30. This small, agent-graded, model/configuration-confounded experiment does not establish the quality of this new workflow or current models. Read the [method and caveats](https://github.com/chriswu727/sibyl/blob/main/docs/EVAL_HOST_CLAUDE.md). A fresh comparison must use the same model, source access, budget, and realistic research questions; report coverage, citation support, contradictions, freshness, cost, and latency separately.

The bounded evidence loop also has a [four-case keyless retrieval run](../evals/results/evidence-loop-keyless-2026-07-21.json) covering the previously missed World Cup river, CUDA founding year, Inception/Batman, and Qantas currency chains. All four fixed host plans reached `ready` and contained the expected answer. This is a single-repeat retrieval test with hand-authored atomic queries; it does not measure whether an arbitrary host can produce the plan or synthesize the final answer.

## Known limitations

- Keyless search engines and public websites can throttle, block, or change behavior.
- Tavily is an optional, credentialed, usage-billed search path; it does not guarantee readiness or make publisher pages deterministic.
- Retrieval relevance does not establish factual truth or source credibility.
- `quality_score` is intentionally unpopulated; no production credibility score is provided.
- Publication dates are extracted from explicit page metadata and may be wrong at the publisher.
- Near-duplicate clustering is content-based and can miss heavily rewritten syndication.
- Autonomous reports are only as capable and reliable as their configured model and retrieved evidence. Offline workflow tests do not measure live report accuracy.
- Live web results are not deterministic; use the offline fixtures for regression testing.

## Development

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[report,finance,extract,pdf,monitor]'
python scripts/check_docs.py
python -m unittest discover tests -v
```

Useful project documents:

- [Documentation index](README.md)
- [Operations and monitoring](OPERATIONS.md)
- [Active roadmap](QUALITY_BUILD_PLAN.md)
- [SourceBundle 1.6 consumer contract](https://github.com/chriswu727/sibyl/blob/main/docs/source-bundle-1.6.md)
- [Performance notes](https://github.com/chriswu727/sibyl/blob/main/docs/PERFORMANCE.md)
- [MCP architecture notes](https://github.com/chriswu727/sibyl/blob/main/docs/MCP_ARCHITECTURE.md)
- [Launch readiness and release gates](https://github.com/chriswu727/sibyl/blob/main/docs/LAUNCH_READINESS.md)
- [MCP installation reference](https://github.com/chriswu727/sibyl/blob/main/llms-install.md)
- [Privacy](https://github.com/chriswu727/sibyl/blob/main/PRIVACY.md)
- [Security](https://github.com/chriswu727/sibyl/blob/main/SECURITY.md)
- [Contributing](https://github.com/chriswu727/sibyl/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/chriswu727/sibyl/blob/main/CHANGELOG.md)
- [Release runbook](https://github.com/chriswu727/sibyl/blob/main/docs/RELEASING.md)

## License

[MIT](https://github.com/chriswu727/sibyl/blob/main/LICENSE)
