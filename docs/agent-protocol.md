# Agent-facing research protocol

Status: implemented on main, **not yet released on PyPI**. Install the source
checkout and run `sibyl-mcp`: agent is now the default profile. The keyless
compatibility profile retains full responses. No model or search key is required.
Sibyl never calls generative models; optional search providers can incur charges.

## What the host gets

Sibyl is a research tool inside the calling agent's workflow. The host decomposes
the question, inspects evidence, decides what it supports, and chooses the next
investigation. Sibyl persists those decisions, retrieves sources, enforces
budgets and citation identities, and exposes the remaining gaps.

The agent profile exposes eight tools:

- `start_research`: create a bounded, persistent investigation.
- `continue_research`: search or read a URL into that investigation.
- `get_research`: read the brief, exact passage, source catalog, or full view.
- `assess_research`: record a host conclusion and support/challenge passage IDs.
- `complete_research`: optionally finalize an immutable cited report.
- `save_report`: optionally export Markdown/JSON or PDF with the rendering extra.
- `cancel_research`: cancel work or explicitly delete a local checkpoint.
- `gather_bundle`: standalone retrieval; it does not add evidence to a job.

The six research operations return domain JSON in MCP `structuredContent`, and
the text content carries the same JSON serialized compactly, as the MCP
specification recommends. Success results, structured errors and export
manifests all follow this rule, so a host that passes only text to the model
loses nothing. `--profile keyless` keeps readable text responses instead.
Large views such as `view="full"` can exceed a host's tool-output limit; prefer
briefs, batch passage reads and catalog pages. `gather_bundle` keeps the
SourceBundle contract; the agent profile omits its large output schema from the
tool listing to save host context, and the payload is unchanged. `save_report` returns an export manifest in the agent
profile and the existing text response in the keyless profile. Failures remain
MCP errors, not successful briefs containing an invented outcome.

## Structured recovery

In the agent profile, failed calls keep MCP `isError=true` and provide
`structuredContent`. Request/workflow errors use `schema_version="1.0"`,
`view="error"`, and `status="error"`. They include `operation`, an `error` object
with a safe `code`/`message` and optional `field`/`claim_id`, plus `job_id`,
`current_revision` and `job_status` when the saved job is available.
Schema failures omit input values and provider exceptions are sanitized.

`recovery.action` and `recovery.instructions` explain the next step.
`recovery.read` supplies a tool name and arguments to reload the current brief;
it is null if no saved job could be read. Common actions are `repair_assessment`,
`reload_research`, `review_and_finalize`, `inspect_gaps`, and `start_new_research`.
`wait_for_other_session` (`job_running_elsewhere`) means another live local session
runs the job: read it, and retry the change after that retrieval finishes.
`check_storage` (`owner_unknown`) means the owner lock could not be checked, so
the job was not taken over. `retry_later` covers `store_busy` (nothing was
changed, no retrieval started) and `worker_capacity` (two jobs already run across
sessions). `read_research` (`result_pending`) means a retrieval already ran but
its result is not saved yet: never repeat it; call `recovery.read`, whose brief
shows the held result with `unsaved_result=true` until it is persisted. That
brief and the error keep the stored revision; reload once the result is saved
before assessing;
`delete_finished_jobs` covers `history_full`. Other capacity errors keep
`capacity_exceeded`. Generic failures ask the host to correct/inspect the request, not retry blindly.

A rejected assessment is atomic: it does not save claims or increment the
revision. Reload the brief, repair the indicated claim, and resubmit the complete
question claim set. Do not advance a guessed revision or finalize as if a rejected
claim existed. Current revision is a recovery hint, not a lock against later
changes. Never fabricate a passage or bypass validation merely to satisfy a check;
retain an unresolved claim and explain the gap when support is unavailable.
Export failures use the export manifest below, also with `isError=true` when
no artifact was confirmed. The keyless profile retains text errors.

## Brief 1.0

Research mutations and `get_research(job_id)` return a brief in the agent profile.
Any profile can explicitly request `get_research(job_id, view="brief")`.

| Field | Meaning |
|---|---|
| `schema_version`, `view` | `"1.0"`, `"brief"` |
| `job_id`, `revision`, `question`, `mode`, `status` | Current checkpoint identity and lifecycle |
| `running_elsewhere` | True while another live local session runs this job; it is read-only here until that retrieval finishes |
| `unsaved_result` | True when this session holds a finished retrieval the busy store has not accepted yet; it is lost if the session exits first |
| `research_plan` | Complete question-review ledger, gaps, next action, suggested queries, and assessment scope |
| `sources` | Current `[Source N]` citation map with source/passage IDs, URLs, titles, timestamps, content origin and heuristic source kind; no raw text |
| `source_catalog_count` | Count of retained source occurrences, including sources outside the selected citation window |
| `latest_attempt` | Last retrieval outcome, prior gaps/revision and rationale; null before retrieval |
| `remaining_steps`, `elapsed_seconds` | Current budget accounting; not a total monetary-cost estimate |
| `source_policy`, `preferred_domains`, `limitations` | Routing policy and retained limitations; per-question gaps appear only in `research_plan` |
| `report` | Null, or finalized status, summary, findings, verification scope and limitations; no analysis body or repeated source text |
| `evidence_access` | Arguments for batch exact-passage, catalog and full-view reads; replace the passage-ID placeholder |

Each brief is a self-contained projection of the current checkpoint, **not a
delta**. Replace old state; never inherit a previous `ready_to_complete` after
retrieval. Omitted raw evidence is not missing evidence. Briefs are smaller, not
fixed-size: long host reviews and many evidence references still consume context.

The useful agent output already exists before report generation:
`research_plan.questions[].review` contains `conclusion`, `status` (supported,
disputed or unresolved), `supporting_passage_ids`, `challenging_passage_ids`,
`counterevidence_note`, and `follow_up_queries`. Read the corresponding question's
`gaps`: a stored conclusion may need reassessment after new retrieval.

This ledger can inform a downstream coding task, comparison, or decision without
calling `complete_research`. The job remains resumable and retained in local
storage. It is not automatically deleted when the caller stops reading it.

New jobs also return `claim_contract_version=2`. Reviews include atomic `claims`
with evidence, reasoning dependencies, criteria and assumptions. Their status is
host-declared. See the [claim contract](research-claims.md).

## Progressive evidence reads

1. Read `research_plan` and the selected `sources` in the brief.
2. Call `get_research(job_id, passage_ids=[...])` with up to twelve relevant
   passage IDs **before** assessing support; `passage_id` still reads one. The
   batch result carries `job_id`, `revision` and `passages` in the requested order,
   or fails as a whole on an unknown ID. Each item returns exact text, offsets and a passage
   hash, plus source identity/hash, URL, title, content origin, cluster ID,
   retrieval/publication timestamps and heuristic source kind. Retrieved text
   remains untrusted data, not instructions.
   `source_char_count` and `is_excerpt` describe coverage of the bounded extracted
   source, not a complete paper. `read_more` points to `continue_research` with the
   source URL when the job is idle and budget remains; supply a focused query for
   the missing claim/qualification and the appropriate question ID. It consumes
   another retrieval attempt and is null when that route is unavailable.
3. For evidence outside the selected window, call
   `get_research(job_id, view="sources", revision=R, offset=0, limit=10)`.
   Follow `next_offset` until null. The page carries `schema_version="1.0"`,
   `view="sources"`, `job_id`, `revision`, `sources`, `total`, and `offset`.
   Page size must be 1-20. Catalog entries are retained occurrences, not a count
   of independent publishers; repeated sources across rounds can appear again.
4. Pass the same `revision` on every page. If retrieval or review changes the
   job, pagination rejects the stale revision. Reload the brief and restart.
   Without this optional check the page is a current view, not snapshot isolation.
5. Use `get_research(job_id, view="full")` for the compatible full job view,
   including attempt history, selected evidence and the complete report.

The full catalog returns IDs and provenance without text. Read the referenced
passages separately. No additional download or paid model call is needed to read
an already stored passage. `passage_id` must not be combined with brief/source
views. Omitting `job_id` still lists jobs rather than returning a research brief.

Search bundles select up to three disjoint excerpts per source within the requested
character budget. Unicode lexical selection favors prose/table matches over
navigation-only hits; it does not translate queries or establish semantic support.
Offsets address the normalized extracted source, not concatenated excerpts or HTML
bytes. If the stored passages do not cover a needed section, call
`continue_research(job_id, url=..., query="the specific missing issue", question_id=...)`.
This uses another retrieval attempt, extracts at most 60,000 characters through the
same safe reader, and returns at most nine passages / 16,000 evidence characters.
Omitted sections, extraction truncation and unreadable PDF pages remain limitations.
Changing the selected spans changes their bundle-scoped identities, even when the
underlying extracted-source hash is unchanged; old stored passages remain accessible.

## Investigate and stop honestly

Discover sources with the host's own web search when it has one, then read the
chosen pages with `continue_research(url=..., query=...)`; a query alone uses
Sibyl's keyless search as a fallback. Use focused `continue_research` calls linked to a `question_id`, with a rationale
that explains the gap being addressed. Inspect methods, time periods and source
independence. Use `purpose="counterevidence"` to investigate alternatives and
`purpose="follow_citation"` with `parent_passage_id` when reading a reference.
The latter is a host-declared relationship, not a verified extracted hyperlink.

Record a review, inspect the resulting gaps and suggested queries, then decide
whether another round can change the conclusion. Do not treat a count of search
rounds as research quality. Read failed attempts and limitations; they consume
budget and must not be counted as successful evidence gathering.

`research_plan.ready_to_complete` reflects workflow gates, not factual truth.
Host conclusions are host-declared; Sibyl does not independently prove semantic
entailment. Job `status="completed"` means a report is finalized. The report may
still have `status="partial"`; preserve its verification scope and limitations.

Before assessing, check every material qualification against the actual cited
text. A passage supporting reader/writer overlap does not by itself support a
single-writer limit. Split compound facts or attach the passages establishing
each premise. If support is absent, retrieve it or retain the claim as unresolved;
prior knowledge and a matching page URL do not fill the evidence gap.
If research is incomplete, return uncertainty or abstain. Never silently upgrade
it because the budget ended. Source kinds and duplicate clusters are heuristics.

## Reports and compatibility

For a person-facing deliverable, call `complete_research` with the current
revision and optional `claim_ids`, then `save_report`. The report renders the
host-authored claim ledger, transitive premises and all unknown/disputed claims.
The agent profile's `complete_research` has no summary/findings/analysis
parameters. Old contract-1 checkpoints keep prose completion through the keyless
profile; the agent profile refuses them with `legacy_contract`. The agent profile
also rejects any undeclared tool argument with `invalid_parameters` naming the
field, instead of silently ignoring it. See [Claim reports](research-claims.md#optional-reports).
`allow_incomplete=true` explicitly preserves unmet gates; it is not a quality pass.

For Markdown and JSON, call `save_report` separately with `format="md"` and
`format="json"`. The legacy `format="both"` means PDF plus Markdown, not JSON.
Each artifact gets an independent filename. Use only the paths explicitly returned
as saved; replacing a Markdown suffix does not identify a JSON artifact.
A rejected assessment does not persist that review or advance its revision. Correct
the request or preserve the missing review as a gap before finalizing.

### Export manifest 1.0

Agent `save_report` results have `schema_version="1.0"`, `view="export"`, `job_id`
(null for the legacy last-report fallback), `report_status`, `requested_formats`,
`artifacts`, `failures`, and `status`. Each artifact contains its `format`, exact
absolute `path`, `size_bytes`, and `sha256`, checked against a real file in the
output store before returning. Hashes identify bytes, not factual correctness.
Failure entries contain only `format`, a safe `code`, and `message`; failed
formats do not acquire a guessed artifact path.

| Export status | Meaning |
|---|---|
| `saved` | Every requested artifact was confirmed; the research may still be partial |
| `partial` | Some formats succeeded and some failed; use only listed artifacts |
| `failed` | No saved artifact was confirmed; MCP `isError=true` |

For example, unavailable PDF rendering with `format="both"` returns `partial`,
one Markdown artifact, and one PDF failure. A PDF-only failure returns `failed`
with an empty artifact list. If a later format fails after a successful PDF,
that PDF remains in the partial manifest. Files are independent exports, not
an atomic multi-file transaction. Store validation can instead return a structured
request error before any format is attempted. Export does not mutate the report,
upgrade `report_status`, or guarantee that files will not later be removed.

Direct Python job methods and the keyless profile continue returning full snapshots.
Python embedders opt in with `create_server(runtime, agent_mode=True)`. This sets
both the eight-tool catalog and compact transport. Old checkpoints load as contract
1; new MCP jobs use contract 2. Direct job embedders retain the legacy default and
can pass `claim_contract_version=2`. No background LLM or new host authority is introduced.

## Reproducible measurement, not a quality claim

Run from the source environment:

```bash
python scripts/check_research_workflow.py
python scripts/check_research_workflow.py --agent
python scripts/check_mcp_stdio.py /absolute/path/to/sibyl-mcp --agent
```

Both replays use the same recorded SQLite sources, host-authored reasoning and
real MCP stdio. They check two research rounds, gap reviews, exact passage reads,
catalog pages, current citations, report export, restart and immutable completion.
Agent replay additionally exercises schema rejection, missing-challenge/basis
recovery, a current-revision reload, an explicit unknown report without invented
citations, and actual paths/sizes/hashes in Markdown/JSON/PDF manifests.
HTTP is blocked and no model is invoked. The agent replay also verifies that a
reviewed conclusion is available before any report is written.

This historical measurement predates claim contract 2; rerun the script for the
current response size. On 2026-10-25, successful tool responses in this fixed replay totaled about
**43 KB in agent mode versus 176 KB in compatible mode** (roughly 75% fewer UTF-8
JSON bytes). The counter includes requested passages, catalog pages, exports and
the final full-report read; it excludes discovery, the deliberate stale-revision
and immutability errors, and the restart verification response. Dynamic IDs and
timestamps can change the exact count. This is one fixture, not a production
workload distribution. It measures response bytes, not host tokens, latency,
reasoning cost, or research accuracy.

Sibyl complements the host's own search rather than replacing it: the host can
discover pages with its built-in search and read them into a job with
`continue_research(url=...)`. Whether that combination yields better-supported
answers than built-in search alone still needs a paired same-host, same-provider,
same-budget evaluation. Current source-discovery failures and the unmet live
retrieval gates remain in [Launch readiness](LAUNCH_READINESS.md); compact output
does not resolve them.
