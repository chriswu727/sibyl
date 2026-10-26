# Sibyl roadmap

Reviewed 2026-10-26. This is the active roadmap for the 0.4.0 beta line
and future stable release. Implemented features are documented in [Usage](USAGE.md);
release blockers are governed by [Launch readiness](LAUNCH_READINESS.md).
The [earlier implementation plan](https://github.com/chriswu727/sibyl/blob/d343267/docs/QUALITY_BUILD_PLAN.md)
is preserved in Git history, not a source of current defaults or quality claims.

## Implemented baseline

- Agent-first source interface (unreleased): eight focused tools, compact
  structured briefs, direct review-ledger consumption, exact evidence reads,
  and revision-checked source pages. Human-readable reports remain optional.
- Persistent host-assisted investigations with multi-round retrieval, question
  reviews, counterevidence requirements, explicit gaps, and partial reports.
- Shared source/passages contracts, hashes, offsets, provenance, date hints,
  duplicate-content clusters, and source inspection.
- Host-only execution: no LiteLLM, autonomous model workflow, subscription
  credential handling or generic model/finance tools. PDF rendering remains optional.
- Optional text-PDF reads through the safe fetch boundary, page markers,
  bounded parsing workers, and explicit extraction limitations.
- Private SQLite checkpoints, cancellation/resume, retained budgets on oversized
  checkpoints, bounded exports, and single-runtime ownership.
- Authenticated loopback monitoring, content-free metrics, structured stderr
  events, and documented recovery procedures. Monitoring is opt-in; an alert
  service and dashboard are not installed automatically.
- Consistent formatting/import checks and typed core job/state/review contracts.
- Recorded-source full MCP workflow replay and clean-wheel core,
  PDF/monitoring, and report/extraction verification.
- Bounded fault tests for abrupt process termination, failed writes before/after
  retrieval dispatch, transaction rollback, repeated cancellation, and export cleanup.
- HTML title fallback, whitespace normalization and truncation warnings; stable
  cited-source/background grouping across report renderers.
- A bounded runtime soak with real SQLite, optional monitoring/PDF workers,
  injected upstream failures, resource sampling, and stopped-store backup/restore.
- Preferred-domain discovery and relevance ranking before the bounded fetch limit.
- A ten-case live host research suite with adaptive follow-ups, inspected evidence,
  explicit discovery failures, and verified Markdown/JSON delivery.

## Implementation sequence

The [functional goals](REFACTOR_GOALS.md) define the target and responsibility
boundary. This is a proposal for the remaining work, not authorization to run
paid evaluations or an announcement that the features exist. Deliver each phase
as a focused, tested change; do not start by adding tools or replacing the entire
retrieval pipeline.

### P0 Freeze scope and acceptance cases

Status: product goals and this plan recorded; the new case set and rubric remain
to be implemented. Covers G1-G7.

Define a frozen representative set before tuning retrieval. Cover three user tasks:
technical suitability, factual investigation and evidence-based recommendations.
Include cases for missing primary sources, stale facts, contradictory accounts,
syndicated copies, unsupported precise numbers, unclear user preferences,
snippet/abstract-only evidence, truncated PDFs, prompt injection, provider failure,
budget exhaustion and follow-up questions after restart.

For each case record the user need, required research questions, expected evidence
types, failure/abstention behavior and how success is scored. Keep deterministic
fixtures separate from held-out live evaluation questions. Record whether each
source was automatically discovered or supplied by the host/operator.

Deliverables: versioned cases, scoring rubric, controlled baseline and an additive
contract proposal for facts/inferences/recommendations/unknowns. Map each proposed
field to an actual consumer need and a validation rule before implementing it.
Do not create a numerical confidence score without a calibrated meaning.

Exit: cases, scoring rules and scope are reviewed before tuning. Establish numeric
quality thresholds before inspecting treatment results; no retrospective lowering
of the existing 66-case live-retrieval thresholds.

### P1 Fix source discovery and reading failures

Status: pending. Covers G2 and G7; starts after P0.

Use the frozen failures to prioritize primary-source candidate selection, query
constraints, extraction and transparent fallbacks. Improve one failure class at
a time, using the existing search and safe-fetch boundaries. Separate discovery
success from successful reading of a manually supplied URL.

Deliverables: targeted retrieval fixes and offline regressions, preserved provenance
and extraction warnings, and a dated live result when authorized. Consider PDF
page/range reads only if the cases show that bounded extraction prevents useful
investigation; do not bundle OCR or browser automation into this phase.

Exit: targeted cases and existing lexical/pipeline/source-quality gates pass.
Show the effect on held-out cases and preserve failure records. A regression fix
does not by itself establish a stable-release quality pass.

### P2 Add the evidence and judgment contract

Status: proposed design, not implemented. Covers G3, G4 and G6.

Extend the existing review ledger with the minimum representation needed for
material factual assertions, host inferences, recommendation criteria and explicit
unknowns. Link factual premises to inspected passages; preserve contrary evidence,
assumptions and relevant dates/scope. Recommendation weights and preferences must
remain visibly distinct from source facts.

Deliverables: additive schema, compatibility/migration fixtures, deterministic
validation, revised brief/export presentation and host instructions. Reuse the
current evidence store and stable IDs. Old host checkpoints remain readable; old
autonomous execution remains disabled.

Exit: nonexistent references, stale revisions and malformed records are rejected.
Missing support is exposed or the assertion omitted, never silently called verified.
Cross-format exports preserve qualifications. Tests distinguish structural
validation from semantic support; no internal judge model is introduced.

### P3 Validate the host research loop and usability

Status: existing eight-tool workflow is implemented; real-host acceptance remains
pending. Covers G1, G3, G5 and G6; builds on P1/P2.

Improve result guidance where frozen cases show the host cannot identify the next
useful action. Preserve compact self-contained briefs, on-demand passages,
question-level gaps, failure reasons and remaining budgets. Do not add a tool
unless an observed task cannot be handled clearly by the current surface.

Select and record two real local MCP host/version configurations, including a
subscription-backed workflow. Verify setup, discovery, structured-content handling,
natural-language research requests, focused follow-ups, explicit abstention,
optional reports and restart. Test text-oriented compatibility separately.
Obtain explicit approval before consuming model/subscription usage.

Deliverables: setup matrix, sanitized interaction traces, reproducible defects and
corresponding fixes. Record required prompt scaffolding and manual intervention.
An MCP client fixture is necessary but is not proof of natural agent usability.

Exit: supported hosts can use the workflow without per-call human coaching;
unsupported hosts and needed configuration are documented. No universal
compatibility claim, hidden billing or silent interpretation of a failed call.

### P4 Measure research quality against ordinary search

Status: not established. Covers the differentiation claim across G2-G6.

Compare the same host/model/effort, dated questions, provider access and overall
time/call budget with and without the Sibyl workflow. Where native-search provider
access cannot be matched, report that confound and limit causal claims.
Declare prompts, stopping rules, retries, cache policy, scoring and budget first.

Measure original-source discovery, material-claim support, important-question
coverage, counterevidence, disclosed uncertainty, justified recommendations,
appropriate abstention, latency and actual usage. Keep factual premises separate
from subjective preference judgments when scoring recommendations. Retain both
outputs and failures; blind scoring to the condition where practical.

Deliverables: paired results, evidence-linked scoring, uncertainty/scope statement
and explicit manual-URL-rescue counts. A self-scored small sample is exploratory,
not independent proof. Do not claim that fewer response bytes imply lower token
cost or better reasoning.

Exit: predeclared comparative criteria are met before making a scoped superiority
claim. If not, fix measured failures or narrow positioning. The separate dated
66-case, three-repeat live retrieval gate still applies before stable publication.

### P5 Validate and publish the release candidate

Status: pending quality and host-acceptance work. Covers G1, G6 and G7.

Retain Python 3.11-3.13 CI, lint/format, typed core, documentation checks, dependency
audits, offline retrieval gates, real stdio replay, bounded fault/soak checks and
clean-wheel core, PDF/monitoring and report/extraction profiles. No failing security
check is waived by product simplification.

Verify package/Registry manifests, source versus release instructions, GitHub About,
English README, the sole Chinese README and the product-page description against
the actual candidate. Coordinate the separately maintained website rather than
silently assuming it matches the repository. Keep histories and published records
accurate; a source commit does not update an existing PyPI distribution.

Exit: all applicable [release gates](LAUNCH_READINESS.md#release-gates) are met,
then follow [Releasing](RELEASING.md). After publishing, verify actual installation
and public artifacts. No tag/version bump or promotional expansion merely because
a documentation change or offline suite passed.

## Working rules

- Execute P0 first; do not substitute a feature list for acceptance cases.
- Keep runtime and storage safeguards in force throughout every phase.
- Separate code changes, live research measurements and publication.
- Use explicit usage approval for paid search/model/subscription evaluations.
- Preserve failed results and unknowns; do not optimize a benchmark by weakening it.
- Keep maintained repository prose and metadata in English. README.zh-CN.md is
  the only Chinese documentation; original evidence and multilingual fixtures
  retain their language.
- Update this plan's status only with retained implementation/verification evidence.

## Subsequent enhancements

These are candidates, not implemented capabilities or release commitments:

1. Selective PDF page/range reads so a host can investigate beyond a truncated
   extraction without repeatedly downloading/parsing the same full input.
2. Extracted citation-link traversal with explicit parent/child provenance;
   current citation lineage is host-declared.
3. Incremental research editions that distinguish retained from refreshed
   evidence; current completed reports remain immutable.
4. Optional OCR and layout-aware tables/formulas with explicit resource and
   confidence limits; no automatic model downloads.
5. A research-progress interface built on appropriately scoped state access;
   current monitoring deliberately excludes research content.

Select these based on observed research failures and user needs. Keep the local
runtime, network policy, evidence contracts, and persistence boundaries clear;
see [Architecture](MCP_ARCHITECTURE.md) for where each extension belongs.
