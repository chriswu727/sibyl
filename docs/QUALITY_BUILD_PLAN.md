# Sibyl implementation plan

Updated 2026-10-26. The objective is a complete research MCP that an existing AI
agent can install, understand and use to investigate questions, make evidence-based
recommendations and produce optional reports. The [functional goals](REFACTOR_GOALS.md)
define the requirements; [launch readiness](LAUNCH_READINESS.md#release-gates)
governs publication.

Work toward that complete user experience through tested, reviewable changes.
Cases, contracts, retrieval fixes and host usability are parts of the same product
implementation. A case collection, a new schema or a passing test suite alone
does not complete the refactor.

## Current source implementation

The implementation below exists on main, not in a new published package.
[Architecture](MCP_ARCHITECTURE.md) is the current design; [migration](MIGRATION.md)
separates removed behavior from intentional old-record compatibility. The dated
paragraphs in this section are evidence history, not separate unfinished phases.

### Remaining implementation priorities

The 2026-10-27 audit found the code serves an evidence recorder, not the agreed
research-manager role, and the work now follows the
[phased refactor](REFACTOR_GOALS.md#phased-refactor):

0. Record the role and remove the keyless compatibility profile, contract-1 prose
   completion, never-written report fields and dead code.
1. Split the 890-line retrieval function into a bundle builder that also assesses
   host-supplied URLs, and keyless discovery as one caller; batch URL reads.
2. Return deterministic, reasoned search suggestions in every response.
3. Classify primary/secondary/aggregator sources per claim and base readiness on
   quoted claims instead of lexical query coverage (the source of false "ready"
   results in the live gate).
4. Check number/date/name consistency between claims and quotes, flag source
   disagreements and require assessed counterevidence.
5. Return deltas with inline passages, measured in real host use (the paired
   smoke cost about five times its baseline), and share one report document
   across formats.
6. Freeze the replacement retrieval gate, then run the approved paired host
   evaluation, choose a fresh version and follow the release runbook.

New model-backed runs still require fresh explicit consent. Do not infer
entailment from structural validation; the latest SQLite sample still fails
exact qualification support.

These are connected acceptance gaps in the same product, not permission to
restore internal models, add an unrelated UI or ship another beta. The design,
README, migration and operations documents must remain aligned throughout.

### Implemented contracts and retained measurements

New MCP jobs now connect atomic facts, reasoning and recommendation criteria
through the [claim contract](research-claims.md). The source implementation checks
references and reasoning cycles, detects changed/stale premises, retains unknowns
and disputes, and renders optional reports from the same ledger. Old contract-1
checkpoints can no longer be finalized. Focused query variants also preserve a leading
entity name when the question word occurs later in the query.

Recorded-source MCP replay covers this workflow and export/restart. Broader host
usability, reliable live discovery and a paired native-search comparison remain
acceptance gaps; the bounded host results below do not meet the complete-product criteria.

An [anonymous three-query smoke record](../evals/results/source-discovery-claims-2026-10-26.json)
retains source identities and diagnostics before/after final-source preference
changes. The final check returned original SQLite/Stripe documentation, but a
PostgreSQL 17 question still returned version 16 documentation. These sequential
single runs are exploratory; they do not clear the live gate or establish causality.

A [four-request Codex Luna low-effort smoke](../evals/results/host-agent-luna-low-2026-10-26.json)
completed one live SQLite investigation with partial Markdown/JSON reports, but
both unknown-case deliveries failed. The first guessed a nonexistent JSON path;
after export instructions were clarified, the next attempted the correct formats
but did not repair rejected claims or stale revisions. The fourth request exhausted
the approved allowance. CLI exit zero did not establish delivery success.
Explicit choice schemas and export semantics are implemented. Agent calls now
also return structured claim/revision recovery and confirmed artifact manifests,
with real stdio regression coverage for rejected assessments, reload, repair and
explicit unknown-report export. Validation rules remain intact and keyless text
responses are preserved. The subsequently authorized two-workflow retest below
measured improved delivery, not a complete semantic-quality pass. The original
four-request smoke remains a retained usability failure, not a paired comparison.

Body selection now chooses disjoint passages across the bounded extracted source
instead of splitting one head-biased window. Unicode lexical matches and
prose/table context reduce navigation-only matches; this is not translation or
semantic verification. Query-targeted URL reads inspect up to 60,000 extracted
characters with at most 16,000 evidence characters and explicit unread sections.
The [two-case real-MCP record](../evals/results/body-evidence-2026-10-26.json)
retains unchanged SQLite/AWS source hashes and before/after passage spans. Both
after runs include the declared writer-limit and retry-contract qualifications,
without host-supplied URLs or model calls. These are narrow reading regressions,
not a complete research, host-usability or stable-release gate pass.

The subsequent [frozen 66-case, three-repeat live gate](../evals/results/live-retrieval-body-selection-2026-10-26.json)
still fails: 81.5% answer coverage, 97.9% ready-state answer precision and 100%
trap-safe rate, but 57.4% answerable readiness and 51.5% repeat stability. The
[2026-10-27 GitHub-runner rerun](../evals/results/live-retrieval-github-runner-2026-10-27.json) also fails, with 72.2%
coverage, 30.2% answerable readiness, 47.0% stability and 90.7% ready-state
precision on a network that blocked more page reads. After hedging the general-web
fallback, the [second runner run](../evals/results/live-retrieval-hedged-2026-10-27.json) still fails (74.1%
coverage, 35.2% answerable readiness, 48.5% stability, 90.5% precision); its
provider outcomes show DuckDuckGo and Mojeek refusing almost every request.
Thresholds are unchanged. Diagnose discovery/readiness misses without treating the two
targeted reading fixes as a complete-product pass.

The first [host-source release gate run](../evals/results/host-source-retrieval-2026-10-27.json)
also fails (55.6% coverage, 9.3% answerable readiness; 98.5% stability, 100%
precision). After page reads identified Sibyl honestly (#66), the
[second run](../evals/results/host-source-retrieval-honest-ua-2026-10-27.json) passes
every threshold except answerable readiness (30.2%; 87.0% coverage, 100%
precision). The owner then made answerable readiness a reported diagnostic.

The first [paired host evaluation](../evals/results/paired-host-2026-10-27/manifest.json)
gave Sibyl the best citation support (78.6% against 69.1% for built-in search
and 71.6% for a deep-research workflow) and 6 of 6 factual answers, but failed
gate 2 on counterevidence (rubric 1.1 against 1.4). Next: make the Sibyl
workflow cost fewer retrievals (fold reading into fewer calls, let the update
carry what the host would otherwise fetch again) so a counterevidence read fits
the host's budget, and have the final answer carry the counterevidence status
Sibyl records. Then rerun gate 2 as a new dated run under the same protocol.

Claim-based human reports now present the generated claim narrative once, while
JSON retains compatible summary/findings/analysis fields and the full ledger.
PDF rendering checks font coverage and reports failures instead of silently losing
required text. Legacy prose layouts and saved research state remain unchanged.

Tracing three live misses found separate discovery failures rather than only a
passage-window problem. Contracted question words such as What's skipped the
existing quoted-title fallback, and focused patch queries split 0.7 into 0 7
while lexical ranking discarded both digits. Targeted regressions now preserve
those query identifiers and recognize contractions. Missing historical tenure and
unreadable/metadata-only papers still need investigation; these parsing fixes do
not clear the retained live gate or establish version-specific semantic support.
The [three-case tracing record](../evals/results/query-identifiers-2026-10-26.json)
retains the before/after queries, failed reads and after-run outcomes. None of
the three after runs contains a frozen gold/alias match; two nevertheless return
retrieval status ok. Do not present query preservation or those heuristic statuses
as successful answers. The 66-case gate has not been rerun after this parsing fix.

An additional controlled regression reproduced a readiness mismatch: three
distinct Crossref metadata records could be counted as three substantive sources
and return ok even when every page read failed. Retrieval now excludes known
metadata/snippet origins from substantive-source, independent-content and
substantive-domain gates, consistent with persistent-job accounting. Catalog
leads remain available. This stricter classification does not identify every HTML
abstract/catalog page, prove semantic support or clear the frozen live gate.

A renewed, explicitly bounded [two-workflow low-effort host retest](../evals/results/host-agent-structured-retest-2026-10-26.json)
delivered confirmed Markdown/JSON files for both tasks in 44.9/49.9 seconds. The
unknown-outcome task recovered from a rejected inference and preserved uncertainty.
Both reports remain partial. Manual review found that the SQLite single-writer
qualification was not supported by its cited excerpt, so this is not a two-case
research-quality pass. The same source's qualification lay outside the selected
passages for the host's broad query. Discovery, evidence coverage and exact
claim-to-passage semantic support remain unfinished quality work.

Exact passage reads now expose their bounded-source length, excerpt status and
a budgeted read-more hint, and host instructions explicitly require support for
each qualification. This makes the scope/recovery path visible; it does not
implement semantic verification or establish a fixed host outcome. The approved
two-workflow allowance is exhausted, and those subsequent changes have not been
retested with another model request.

## The finished user experience

A user installs Sibyl in an existing local MCP-capable agent and asks an ordinary
question or requests a recommendation. The agent can identify the user's needs,
investigate focused questions, read original sources, examine contrary evidence
and return a useful answer whose material factual premises have inspectable
support. The user can ask a follow-up and reuse the ongoing investigation.
A human-readable report is available when requested.

The agent should learn this workflow from tool descriptions and concise server
instructions. Users should not have to call tools manually or repeatedly explain
how to connect evidence to a conclusion. Sibyl stores evidence and research state;
all semantic reasoning and writing remain in the host.

## Start with a complete investigation

Begin implementation with a representative suitability/recommendation request
that exercises the whole path: user constraints, focused questions, source
discovery, passage reads, factual premises, counterevidence, a conditional
recommendation, follow-up investigation and optional export.

Use this investigation as a working example while connecting the missing pieces
of the existing code. Keep the eight-tool default and extend the current review
ledger and evidence store. Establish meaningful cases and expected outcomes
alongside the behavior being implemented, then retain each fixed failure as a
regression. Do not make an isolated test-package delivery the stopping point.

Use recorded sources for repeatable development and reserve separate held-out
questions for live quality measurements. Include technical suitability, factual
investigation and recommendations, with conflicting sources, stale information,
copied content, missing numbers, unclear preferences, unreadable documents,
prompt injection, provider failure and restart. Record whether sources were
automatically discovered or manually supplied.

## Connect conclusions and recommendations to evidence

The implemented claim contract extends question-level conclusions and passage
links with individual factual assertions, deductions and recommendation criteria.
The requirements below describe invariants to preserve while closing the remaining
quality gaps, not an instruction to build a second claim store.

Represent the minimum information needed by the host: stable statement identity,
statement text and type, support/challenge references, dependencies on other
statements, assessment status, reasoning and applicable assumptions or scope.
Keep missing or disputed information visible. Maintain the concrete additive
[schema and examples](research-claims.md); avoid redundant fields and uncalibrated
numerical confidence scores.

For recommendations, record the user's criteria and distinguish them from
host-supplied assumptions. Connect each material advantage, disadvantage and
comparison premise to its supporting evidence. If a price or performance figure
is unavailable, preserve the unknown and explain its effect on the recommendation.
If priorities are missing, the host should ask or offer conditional choices.

Apply the contract throughout Sibyl-managed conclusions and deliverables,
including analysis and recommendation rationale. Existing evidence IDs, revisions
and provenance remain the common reference system. Unknown references and stale
submissions fail clearly; new evidence marks affected assessments for review.
Structural validation and host-declared semantic support must stay distinguishable.

Implementation belongs in the existing job models, research state/review,
persistent handlers and result projections. Keep persistence in the job store.
Version new records explicitly, preserve old checkpoints and exports, and keep
old autonomous execution disabled. Historical records must not acquire a fresh
verification verdict merely because they can be loaded.

## Make useful original evidence easier to obtain

Improve source discovery and reading as actual investigations reveal failures.
Trace each miss through query generation, candidate selection, relevance ranking,
fetching and extraction before choosing the fix. Prioritize relevant original
documentation, papers and first-party disclosures while retaining useful contrary
evidence. Source labels remain heuristics rather than authority guarantees.

Preserve canonical source identity, exact passages, hashes/offsets, timestamps,
available publication metadata, content origin and extraction warnings.
Search snippets and abstracts are leads; a truncated document is only partially
read. Duplicate content must not become independent corroboration.

Keep direct reads and explicit citation-following in the same investigation and
budget. Add selective PDF page/range reading if observed tasks need evidence
beyond a truncated read. Any new reader must use the existing safe network and
bounded-parser boundary. Expose failed reads and the remaining evidence gap.
Keep paid providers and third-party rendering explicit.

Judge discovery separately from reading a host-supplied URL. Track the unresolved
weakness shown by the retained host suite, where seven of ten cases needed
manually supplied URLs. A successful manual rescue must not hide a discovery miss.

## Make the research loop clear to the host

After each action, return enough current state for the host to decide what to do:
what was found, which judgments have evidence, which questions remain unresolved,
what conflicts or limitations matter, and what retrieval budget remains.
The next query should address a specific gap. A repeated query or a larger source
count alone must not be treated as progress.

Keep compact self-contained briefs, per-question updates with inline passages
after each read or assessment, on-demand exact passages and the complete
saved-source catalog. Preserve evidence outside the selected report window.
Make failures and incomplete outcomes actionable, and let the host stop with
uncertainty when more retrieval is unlikely to resolve the question.

Support both focused lookups and persistent investigations without making a
simple factual query require a long form. Add or change a tool only when a real
user task cannot be expressed clearly through the current surface.

## Preserve useful answers and optional reports

The structured investigation must be useful before a report is finalized.
Keep conclusions, evidence links, recommendation criteria, assumptions, conflicts
and gaps available to downstream agent work.

Markdown, JSON and optional PDF should render from the same saved state.
Human-readable outputs can summarize evidence but must preserve material
qualifications and partial status. JSON retains the machine-readable evidence
package. Completed reports remain immutable; new editions need an explicit
research/update design rather than a silent rewrite.

## Verify the complete product during implementation

Exercise the actual MCP journey as the capabilities are connected. Retain
regressions for missing support, invalid IDs, stale judgments, contradictory
sources, incomplete reads, failed providers, cancellation, restart and exports.
Existing Python, type, lint, security, runtime and distribution checks stay in force.

Record real host/version configurations, including a subscription-backed workflow
and text-oriented compatibility where supported. Check installation, tool discovery,
structured results, natural research requests, follow-ups, explicit uncertainty
and optional export. Record prompt scaffolding and manual intervention so that
a fixture success is not mistaken for autonomous host usability.
Obtain explicit approval before consuming model/subscription usage.

Measure answer quality with the same host/model/effort, dated questions, provider
access and overall budget against ordinary search. Declare scoring, stopping and
retry rules before inspecting results. Where native-search access cannot be
matched, disclose the confound and limit the claim.

Assess original-source discovery, important-question coverage, material-claim
support, counterevidence, justified recommendations, appropriate abstention,
latency and actual usage. Keep subjective preferences separate from empirical
facts. Retain both outputs and failures; blind scoring to the condition where
practical. Smaller tool responses are not automatically lower token cost or better
answers. The existing dated 66-case, three-repeat retrieval gate remains applicable.

## What counts as done

The refactor is complete when the supported installation and host configurations
can perform the intended investigation end to end, and retained evidence shows:

- Material factual premises can be traced to exact evidence; inferences,
  recommendation criteria, assumptions and unknowns remain explicit.
- The host can find/read useful sources, investigate counterevidence, address gaps
  and continue after a follow-up without per-call human coaching.
- Discovery failures, conflicting evidence, stale claims, incomplete reads and
  exhausted budgets produce useful, honest outcomes.
- Structured results and optional reports preserve the same evidence and limitations.
- Existing records remain usable, and runtime, security and clean-install checks pass.
- Research-quality and real-host acceptance results meet their declared criteria,
  with any superiority claim limited to what the comparison establishes.
- English repository documentation, the sole Chinese README, GitHub About, package
  and Registry metadata, and the separately maintained product page agree with
  the actual candidate.
- Applicable [release gates](LAUNCH_READINESS.md#release-gates) are satisfied and
  published artifacts are verified through the [release runbook](RELEASING.md).

## Delivery rules

Continue toward the complete product through scoped commits and reviewable PRs.
An intermediate artifact or successful subsystem check is progress, not completion.
Keep development, live measurements and publication accurately documented.

Use English for maintained repository prose and metadata. README.zh-CN.md is the
only Chinese documentation; retain original-language evidence and multilingual
fixtures. Do not alter historical results, weaken gates or add hidden billable
actions. Keep internal model execution and subscription-token handling out of
Sibyl. Independent chat UI, model orchestration, financial-analysis tools, OCR and
automatic publication are outside the current requirements.
