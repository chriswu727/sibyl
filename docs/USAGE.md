<!-- mcp-name: io.github.chriswu727/sibyl -->

<div align="center">

# Sibyl usage reference

This reference follows the `main` branch. See the [landing page](../README.md)
for release status and the [architecture guide](MCP_ARCHITECTURE.md) for current
execution limits.

**Deep research tools for AI agents — persistent investigations and reusable evidence.**

Sibyl is a local research MCP for an existing AI agent. The host plans and reasons; Sibyl retrieves evidence, preserves reviews and gaps, and exports optional host-authored reports. No internal model or subscription credentials are used.

[![Tests](https://github.com/chriswu727/sibyl/actions/workflows/tests.yml/badge.svg)](https://github.com/chriswu727/sibyl/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sibyl-research?color=blue)](https://pypi.org/project/sibyl-research/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-0f766e)](../pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chriswu727/sibyl/blob/main/LICENSE)

[Install](#quick-start) · [SourceBundle contract](https://github.com/chriswu727/sibyl/blob/main/docs/source-bundle-1.6.md) · [Evidence-loop contract](https://github.com/chriswu727/sibyl/blob/main/docs/evidence-loop-1.0.md) · [Privacy](https://github.com/chriswu727/sibyl/blob/main/PRIVACY.md) · [Security](https://github.com/chriswu727/sibyl/blob/main/SECURITY.md) · [Changelog](https://github.com/chriswu727/sibyl/blob/main/CHANGELOG.md)

</div>

## What Sibyl is

Sibyl combines a research workflow with an inspectable evidence boundary:

> Research produces evidence-linked conclusions and visible gaps. A report is optional; neither output is automatically true.

The core path is `gather_bundle()`: a keyless retrieval tool that returns a typed, versioned `SourceBundle`. It does not generate an answer. The bundle carries source and passage identities, hashes, offsets, retrieval and publication metadata, content-origin labels, near-duplicate clusters, relevance signals, and explicit failure states. For dependent questions, `gather_evidence()` adds a bounded, auditable sequence of atomic retrieval steps while leaving planning and synthesis to the host agent.

Persistent jobs maintain agent research state without requiring a report. SourceBundle retrieval is shared by MCP, the inspection CLI and the Python API.

| Mode | Tool | API key | Who reasons? | Best for |
|---|---|---:|---|---|
| Structured retrieval | `gather_bundle()` | No | Your agent | Pipelines, contracts, machine-readable citations |
| Readable retrieval | `gather_sources()` | No | Your agent | Conversational hosts and manual inspection |
| Bounded evidence loop | `gather_evidence()` | No | Your agent | Multi-step questions and auditable follow-ups |
| Host-assisted report | `start_research(mode="host")` | No | Your agent | Persistent research and Markdown/JSON delivery |

## Quick start

### 1. Install the current source checkout

This guide follows main, including the unreleased agent profile and research views.
These additions are not in 0.4.0b1. The older 0.3.0 package does not
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
python -m pip install -e '.[report]'   # PDF rendering only
python -m pip install -e '.[rerank]'   # local cross-encoder ranking
python -m pip install -e '.[pdf,monitor]'  # text-PDF reads and local operational endpoints
python -m pip install -e '.[all]'      # every optional capability
```

### 2. Add the research MCP server

For Claude Code:

```bash
claude mcp add sibyl -- /absolute/path/to/sibyl/.venv/bin/sibyl-mcp --profile agent
```

For clients that accept an MCP server configuration, replace the absolute path:

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "/absolute/path/to/sibyl/.venv/bin/sibyl-mcp",
      "args": ["--profile", "agent"]
    }
  }
}
```

No search or model key is required for the eight default tools, including host
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
sibyl-mcp --profile agent --list-tools
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

MCP report exports use `SIBYL_OUTPUT_ROOT` (default `reports/`).
`save_report(output_dir=...)` selects a relative subdirectory of that root, not
an absolute path. Stateful report operations reject overlapping calls.

The default `agent` profile exposes eight tools with compact structured research
briefs. The host must consume MCP `structuredContent`; text-only clients can use
`--profile keyless` for full responses and twelve compatibility tools.
`--profile auto` is an alias for agent and never inspects model credentials.
`report`, `finance` and `full` are no longer valid profiles.

| Tool | Result |
|---|---|
| `start_research` | Persistent host-led investigation |
| `continue_research` | Focused search or direct source read |
| `get_research` | Brief, exact passage, paged catalog, full state or job list |
| `assess_research` | Host conclusion with stable support/challenge links |
| `complete_research` | Optional immutable host-authored report |
| `save_report` | Markdown (default), JSON or optional PDF |
| `cancel_research` | Cancel or explicitly remove checkpoint |
| `gather_bundle` | Standalone SourceBundle 1.6 |

The keyless compatibility profile also includes `gather_evidence`,
`gather_sources`, `quick_search` and `read_url`. Neither profile calls models.
Read exact passages with `get_research(passage_id=...)` before assessing them.
`view="sources"` pages the complete saved catalog; use the current revision to
reject stale pagination. `view="full"` returns the full compatible snapshot.
See [Agent protocol](agent-protocol.md).

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
sixteen `preferred_domains` such as a relevant official documentation host. The
first two also receive bounded domain-focused discovery searches, unless the
query already contains an explicit `site:` constraint. Preferences prioritize
relevant candidates before fetching; they are not an allowlist and do not exclude
useful sources from other domains. All preferences still apply during final
source selection. Ranked retrieval prioritizes a relevant full-text representative
from each preferred domain within the existing period constraints; other slots
retain relevance and content/domain diversity. Snippets, metadata and community
archives do not receive that representative boost. `ranker="none"` keeps its
original ordering. Domain preference does not establish authority. Match
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

After reading the evidence, call `assess_research` for each question, including
atomic material claims:

```text
assess_research(job_id="...", revision=the_current_revision, question_id="Q1",
  status="disputed",
  conclusion="The findings differ under the recorded conditions.",
  counterevidence_note="Compared the original result and contrary evidence.",
  claims=[{
    "claim_id": "Q1.C1", "kind": "fact", "status": "disputed",
    "text": "The available sources disagree on the requested result.",
    "supporting_passage_ids": ["bundle/S1/P1"],
    "challenging_passage_ids": ["other/S1/P2"]
  }])
```

Facts link exact evidence. Inferences and recommendations link `basis_claim_ids`
and explain their rationale; recommendations also declare criteria with origin
`user` or `host_assumption`. Unknown information uses `status="unresolved"`
with an explanation in `rationale`. Each assessment replaces the question's claim
set; preserve IDs that other judgments depend on. The
[claim contract](research-claims.md) defines fields, limits and a complete example.

A supported review needs supporting evidence; disputed needs both sides;
unresolved preserves missing evidence. Claim links and transitive reasoning
premises supply the question-level evidence links automatically. Only cite passages
actually inspected. The server checks reference integrity, revision freshness and
structural completeness, **not semantic correctness**.

New retrieval makes the question's review stale. Changed premises and new evidence
also invalidate dependent judgments across questions. Reviewed evidence stays in
the report window; more than twelve distinct reviewed sources or conflicting
duplicate snapshots require narrowing the references instead of losing them.

The claim ledger is usable without a report. To finalize a new job, get the latest
revision and submit:

```text
complete_research(job_id="...", revision=the_current_revision,
                  claim_ids=["Q1.C1"], allow_incomplete=true,
                  limitations=["A specific unresolved question."])
save_report(job_id="...", format="json")
```

New MCP jobs use `claim_contract_version=2`. Omit the legacy summary/findings/analysis
arguments; the report renders the host-authored claims, their premises, current
source citations, reasoning, criteria and assumptions. Unknown/disputed claims
remain included even if not selected. A partial report can explicitly abstain
without inventing sources. Unknown claim IDs, dependency cycles and stale
revisions fail. An old checkpoint retains contract 1 and its prose-based completion.

Completion normally requires current question reviews, resolved claim gaps and
substantive counterevidence investigations. `allow_incomplete=true` permits delivery
with all remaining gaps and `report.status="partial"`; it is not a quality pass.
Job `status="completed"` means the report is immutable. Start another job for
a new edition.

Report JSON schema 1.1 adds the claim ledger and freshness flags, alongside exact
selected passages, hashes, offsets, reviews and investigation steps. Markdown/PDF
retain the reasoning, criteria, assumptions and limitations in the generated
analysis, presented once under Evidence-linked conclusions rather than repeated
in summary, findings and analysis. JSON keeps the compatible fields and full
machine-readable evidence package; old prose reports retain their layout.
PDF export tries available Arial Unicode, DejaVu Sans or Noto Sans fonts and
otherwise uses the core font. Font coverage and required rendering failures
are explicit errors, not saved PDFs with missing report text. Use Markdown/JSON
when a suitable font is unavailable. A combined PDF/Markdown export may be partial
with only the confirmed Markdown artifact; inspect the manifest failures.
`verification_scope="citation_targets_only"` means reference checks, not proof
of semantic support or factual truth.

Retrieval diagnostics keep `unique_domains` for all returned sources and expose
`substantive_domains` for relevant extracted-text sources used by readiness.
Crossref metadata and search-snippet origins do not count as substantive text,
independent content or full-text domain diversity. They remain inspectable leads.
Downloaded HTML may itself be an abstract or catalog entry; the host still needs
to judge the actual evidence scope rather than infer a full paper was read.

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

| Limit | Default / ceiling |
|---|---|
| Retrieval attempts | 8 per job, caller-selectable from 1 to 12 |
| Selected report sources | 12 |
| Simultaneous job workers | 2 |
| Active execution time | 600 seconds across continuations; host thinking time excluded |
| Checkpoint records | 64; explicit deletion required when full |
| Checkpoint payload | 1 MiB per job, plus 64 KiB reserved for failure recovery metadata |
| SQLite database | 16,384 pages; approximately 64 MiB with default page size |
| Progress history | Last 80 events per job |
| PDF fetch / parsing | 2 MiB; first 40 pages; at most 60,000 characters; 6-second worker deadline |
| PDF workers | 2 per event loop; 512 MiB memory threshold; see platform details in SECURITY.md |
| Export file | 16 MiB |
| Exports per output directory | 256 files / 128 MiB; explicit retention required |

A retrieval attempt can itself issue multiple search-provider requests. These
limits do not promise an exact dollar budget; external search pricing,
provider retries and already-dispatched requests still matter.
Cancelled requests may still be billed by the provider.

The private database lives in `SIBYL_OUTPUT_ROOT/.research/jobs.sqlite3`.
Only one runtime may own that directory at a time. Cancellation preserves the
checkpoint; `cancel_research(job_id=..., delete=true)` removes it and reclaims
history capacity, but does not remove exported files. SQLite deletion is not
guaranteed forensic erasure. See [Privacy](../PRIVACY.md).

JSON, Markdown, and PDF use the same `ResearchReport` contract. JSON retains
source passage IDs, timestamps, limitations, verification scope, and legacy model-call
counters when reading old records. New work makes no model calls. PDF export requires `[report]`; text-PDF input separately requires `[pdf]`.
For Markdown and JSON, call `save_report` twice with `format="md"` and
`format="json"`. The legacy `format="both"` means PDF plus Markdown only.
Each export gets an independent filename; use its returned path, not a guessed
path with another extension. If PDF-only export fails, the MCP call returns a
tool error. With `format="both"`, a PDF failure is disclosed while a successful
Markdown artifact is retained.
The agent profile returns an [export manifest](agent-protocol.md#export-manifest-10)
with confirmed artifacts and per-format failures; `report_status` remains separate
from export success. Failed calls also carry [structured recovery](agent-protocol.md#structured-recovery)
instead of requiring the host to infer state from a text error. The keyless
profile retains its text response format.
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

When Jina Reader is used, the target URL is sent to that external service. It is disabled by default in every workflow. Set `render_thin_pages=true` for retrieval or `js_render: true` for the compatibility read tool only when this disclosure is acceptable.

## Migration

This source preview removes internal model execution, LiteLLM, provider routing,
the `research/analyze/compare/swot/timeline` tools and finance/chart tools.
The `sibyl research` CLI and implicit `sibyl "question"` report command are gone;
use MCP for research or `sibyl gather` for evidence inspection.

Install in a new virtual environment to leave old transitive dependencies behind.
The `report` extra now contains only PDF rendering dependencies. No model key,
API base, subscription token or model profile is needed. Old model environment
variables are ignored, not used to enable capabilities.

`SIBYL_CONFIG` is optional and only accepts `search_engine`, `extractor`,
`jina_fallback` and `js_render` for the compatibility `quick_search/read_url` tools.
Unsupported legacy fields cause an actionable error without echoing values.
Persistent research uses its explicit tool arguments and safe defaults.
See [configuration example](../sibyl.yaml.example).

Existing checkpoints and completed reports remain readable/exportable. An old
autonomous checkpoint cannot resume model execution and is not silently converted
or deleted. Start a new host job to investigate further. Old host checkpoints
retain their prior completion rules. Historical model verdicts remain historical,
not new independent verification.

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
- Host conclusions depend on host reasoning and evidence quality; offline tests do not measure semantic report accuracy.
- Live web results are not deterministic; use the offline fixtures for regression testing.

## Development

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[report,extract,pdf,monitor]'
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
