# Sibyl host agent research goals

Recorded 2026-10-26; the product role was revised on 2026-10-27 (see
[product role](#product-purpose) and [phased refactor](#phased-refactor)). This is
the functional target for the current refactor, not a description of features
that have all shipped. The implementation baseline
is [314e721](https://github.com/chriswu727/sibyl/commit/314e721), which removed
internal model execution. The [implementation plan](QUALITY_BUILD_PLAN.md)
connects the requirements into a complete product; [launch readiness](LAUNCH_READINESS.md#release-gates)
continues to govern publication.

## Product purpose

Sibyl is a local research MCP for an AI agent the user already uses. It manages
that agent's whole research loop so the result is better than the agent would
reach alone: it structures the investigation, tells the agent what to search
for next and why, reads the sources the agent finds, assesses those sources,
checks the agent's claims against the exact evidence, and keeps conclusions,
gaps and unknowns. A human-readable report is an optional projection of the same
research state, not a long document or an independently generated answer.

Sibyl never runs a model, so every part of that management is deterministic:
templates, rules, source classification and text checks over stored evidence.
An MCP server also cannot call the host's tools. Sibyl therefore guides the host's
own web search through its responses; it does not execute that search. Its
keyless search is a fallback for hosts without one: public search pages often
refuse keyless requests, most visibly from datacenter networks.

The intended advantage over the host's ordinary search is evidence-directed
discipline: original sources found through targeted searches, exact passages
instead of model-summarized pages, explicit counterevidence, claims checked
against quotes, and gaps that drive the next search. That advantage is a
hypothesis until the paired evaluation measures it. More sources, more rounds
or more prose do not by themselves establish better research.

## Responsibility and subscription boundary

The host agent understands the request, runs web searches with its own tools,
judges semantic support, chooses tradeoffs and writes the final answer. Sibyl
proposes a question plan, names the next searches with their purpose and reason,
reads and selects evidence from the sources the host supplies, classifies and
assesses those sources, checks claims deterministically, persists the research
state, enforces resource limits and exports reports built from recorded claims.
Sibyl's suggestions are guidance; the host may override them, and Sibyl records
what was actually searched and read.

A subscription-backed agent can call Sibyl through a compatible local MCP
integration. Sibyl must not read subscription tokens, reuse login sessions,
call a generative-model API, invoke hidden MCP sampling or create a second agent.
Host usage remains subject to the host's own account, limits and privacy terms.
Optional search/rendering services have separate disclosure and billing boundaries;
an agent subscription does not pay for them. Anonymous retrieval remains the
no-key default, with its availability limitations stated explicitly.

## Evidence and truthfulness policy

The product goal is to prevent unsupported content from being presented as
established fact. It is not a guarantee of perfect truth or zero hallucinations.
A web page can be wrong, stale or incomplete; a model can misread it. A valid URL
does not establish support, and support from a source does not establish truth.
Sibyl cannot enforce what a host writes outside Sibyl's own result/export path.

The target research contract distinguishes four kinds of statement:

- **Sourced fact:** a factual assertion linked to the exact inspected passages,
  including the relevant date, entity and scope. Report it as a source-backed
  assertion, not independently verified reality.
- **Inference:** a host deduction that identifies its evidence, assumptions and
  reasoning, rather than presenting the deduction as a quotation from a source.
- **Recommendation:** a host judgment against the user's needs and explicit
  comparison criteria, with evidence-backed advantages, disadvantages and tradeoffs.
- **Unknown or disputed:** a missing, conflicting, stale or unreadable basis that
  must remain visible instead of being filled with a plausible invention.

Every material factual premise in a recommendation should have inspectable support.
This applies throughout the answer, analysis and recommendation rationale, not
only to the summary or a list of findings.
Missing prices, dates, measurements or compatibility details stay unknown. Scores
or weights are user preferences or disclosed host judgments, not invented empirical
measurements. “Best for this workload under these constraints” is different from
“objectively best.” If the user's priorities are unclear, ask or state conditional
recommendations instead of inventing preferences.

Conflicting evidence must retain both sides and their scope. Copied articles are
not independent corroboration. An absent search result is not evidence that an
event or drawback does not exist. Freshness metadata and primary-source labels
are useful signals, not proof of currentness, authority or accuracy.

### What can be enforced

Deterministic checks can reject missing or unknown evidence references, stale
revisions, malformed records and undeclared workflow gaps. They can validate an
exact excerpt against stored text and preserve extraction limitations. Hashes
identify stored content; they do not authenticate the publisher's claims.

Deterministic checks can also flag lexical inconsistencies between a claim and
its quotes (a number, date or name in the claim that no quote contains) and
disagreements between sources on the same number or date. Such flags are warnings
about text, not semantic verdicts. Semantic entailment, source credibility and
recommendation quality remain host judgments evaluated separately. Claim-to-evidence records distinguish
structural validation from host-declared semantic support. It must not acquire a
“verified” label merely by passing citation checks. The source implementation now
provides an additive [claim contract](research-claims.md), explicit migration
behavior and claim-derived report rendering. No internal judge model is part of this plan.

## Target user experience

The user asks a normal research question in their existing agent. They should not
need to learn tool names, manually allocate source IDs or fill out a research form.
The host should discover the workflow from MCP descriptions and instructions,
with an optional short research policy rather than a mandatory long pasted prompt.

For example, “Is SQLite WAL appropriate for this multi-process application?”
should lead to focused questions, original documentation, inspected passages,
limitations/counterevidence and a conditional conclusion. A follow-up about a
network filesystem should reuse the ongoing job and reassess affected judgments.
Completed reports remain immutable; revising a finalized report currently requires
a new job, not a silent rewrite.

The default response to the user is a useful answer with supporting sources and
uncertainty. A report is produced when requested. Empty retrieval, access blocks,
unreadable PDFs and exhausted budgets must be useful outcomes rather than
completed-looking answers.

## Functional requirements

### G1 Installation and onboarding

Keep a small core installation, local stdio transport and no required model/search
key. Document source preview versus published packages, the host's structured
content requirement, absolute executable/storage paths, optional extras and a
startup check. Before the first stable release, remove the text-oriented keyless
compatibility profile and the contract-1 prose completion path; old checkpoints
stay readable and exportable. Keyless search remains available as an explicit
fallback tool.

Acceptance: a fresh installation can discover tools, start/read/cancel a job and
complete a recorded evidence workflow without model SDKs or secret provisioning.
Real-host usability must be tested separately from a synthetic MCP client.

### G2 Source discovery and reading

Guide the host to topic-appropriate original sources: every research response
names the next searches (gap, primary source, counterevidence, date or language
variant, site-specific) with the gap or reason that produced them. Sources the
host supplies receive the same passage selection, clustering and assessment as
sources Sibyl finds itself, and several URLs can be read in one call. Preserve
source identity, canonical URL, retrieval time,
available publication metadata, content origin, exact passages and extraction
limits. Distinguish search snippets, metadata, abstracts and full text.

Acceptance: frozen discovery failures gain targeted offline regressions; live
evaluation distinguishes discovered sources from host-supplied URL rescues.
Blocked, truncated or unreadable sources are never represented as fully read.
Do not solve retrieval failures by silently enabling paid providers or bypassing
network controls. PDF page/range reading is conditional on observed failures,
not a prerequisite for every investigation.

### G3 Persistent evidence-directed research

Keep focused questions, retrieval attempts, rationale, evidence, reviews and gaps
in a bounded job. The host can investigate contrary evidence, compare conditions,
follow a supplied source URL and revise a judgment after new evidence arrives.
The next retrieval should address a named gap rather than merely increment a
round counter, and Sibyl should propose that retrieval rather than leave the host
to invent it. Repeated searches and duplicate publishers must remain visible.

Acceptance: a replay preserves the path from gap to retrieval to revised review
across process restart. Stale judgments cannot silently retain an old readiness
decision. Cancellation and failures retain evidence and consumed budgets.

### G4 Claim support and evidence-based recommendations

Extend the existing question-review ledger only where needed to represent the
statement distinctions above, material claim-to-passage links, assumptions,
conflicting evidence and recommendation criteria. Add deterministic checks for
claim-to-quote consistency of numbers, dates and names; surface source
disagreements; mark facts that rest only on secondary or background sources; and
require counterevidence to be read and assessed, or declared not found, rather
than merely searched for. Reuse stable evidence IDs and
revision protection instead of creating a separate report-only evidence store.

Acceptance: controlled cases reject nonexistent references and stale submissions;
unsupported material claims are omitted or explicitly unresolved in Sibyl-managed
deliverables. Recommendations expose which factual premises support each tradeoff
and which criteria depend on user preference. Semantic support is scored through
evaluation, not inferred from schema validity or keyword overlap.

### G5 Agent-facing results and stopping behavior

Return a compact result: what changed, the passages just read, current gaps and a
ranked list of next actions (search, read, assess or stop) with reasons. The full
brief with every review stays available on request.
Read exact passages and the full source catalog on demand. Preserve failure details
and partial status; smaller output must not mean hidden contrary evidence.

Acceptance: the host can choose a useful next action from the result, read the
underlying evidence, stop with uncertainty and reuse a conclusion without writing
a report. Measure actual host usage before claiming token or latency savings.
Readiness is a workflow signal, not a truth score.

### G6 Human reports and reuse

Keep host-authored Markdown/JSON and optional PDF, grounded in the same saved
research state. Preserve provenance, conclusions, material limitations and explicit
fact/inference/recommendation distinctions. JSON is the full machine-readable
evidence package; human formats may summarize but must not erase qualifications.

Acceptance: cross-format fixtures preserve citations and partial/disputed status,
and finalized reports cannot mutate. Reading or exporting a historical report
must not make its old verdicts look like a fresh verification.

### G7 Safety and operational continuity

Retain the existing public-network boundary, prompt-injection warnings, quotas,
cancellation, private local checkpoints, revision checks, bounded parsing and
optional authenticated monitoring. No automatic upload, external publication or
deletion of reports. Research content is not encrypted at rest or scrubbed of all
personal data; disclose these limits.

Acceptance: existing runtime, fault, security and distribution gates remain in
force. No silent provider change, new billable action or authority expansion.
Model/subscription-backed evaluations require explicit approval for usage.

## Current baseline versus remaining work

| Area | Current source status; historical measurements are dated separately |
|---|---|
| Internal model removal and eight-tool default | Implemented on main; not a new PyPI release |
| Structured briefs, exact passage reads and source pagination | Implemented |
| Persistent reviews, counterevidence workflow and explicit gaps | Implemented; host-declared judgments |
| Citation target and revision validation | Implemented; not semantic verification |
| Markdown/JSON and optional PDF export | Implemented |
| Original-source discovery reliability | Measured weakness; improvement required |
| Explicit claim/inference/recommendation contract | Implemented in the source preview; host-declared semantic support |
| Natural tool use inside real subscription-backed hosts | Two bounded low-effort workflows delivered partial reports; broader usability and exact claim support remain unproven |
| Better research than built-in search | Hypothesis; paired evaluation not established |
| Search guidance to the host | Deterministic gap-driven suggestions with kind and reason, named in the agent instructions; effect on research quality unmeasured |
| Full assessment of host-supplied URLs | Implemented; up to five URLs per read share the bundle builder's ranking, clustering and sufficiency |
| Source quality per claim | Cited source count, kind hints, domains and weak-support flags per fact; background-only support blocks completion; URL heuristics, not credibility verdicts |
| One report document for every format | Markdown, text and PDF render one shared document; JSON serializes the same record |
| Compact per-question updates with inline passages | Implemented for reads and assessments; 43% fewer response bytes on the offline replay; real-host token use unmeasured |
| Claim/quote consistency; source disagreement | Numbers must appear in a fact's quotes; names absent from cited text and supported facts with different numbers are flagged; counterevidence read/linked status per question; string comparisons, not entailment |

The retained ten-case host suite needed host-supplied URLs in seven cases. The
dated live retrieval gates still have failures. See [launch evidence](LAUNCH_READINESS.md)
for the runs and their scope. Neither architecture simplification nor offline
tests erase those limitations.

The [latest host retest](../evals/results/host-agent-structured-retest-2026-10-26.json)
improved delivery and error recovery but retained a material SQLite citation-support
gap. Later excerpt-scope hints have offline coverage, not another model test.
The [architecture](MCP_ARCHITECTURE.md) specifies the implemented design;
[migration](MIGRATION.md) inventories intentional legacy data compatibility.

## Phased refactor

Agreed on 2026-10-27 after an audit of the current source. Each phase lands as
separate reviewed pull requests with tests; none waives a release gate.

0. **Direction and cleanup.** Record this role; remove the keyless compatibility
   profile, contract-1 prose completion, never-written autonomous-era report
   fields and dead code; expose keyless search as an explicit fallback tool.
1. **Bundle builder.** Split the retrieval pipeline so any set of pages, including
   host-supplied URLs, gets passage selection, clustering and assessment; keyless
   discovery becomes one caller. Allow batched URL reads.
2. **Search guidance.** A deterministic module turns gaps, source kinds, claims and
   question structure into ranked search suggestions with reasons, returned in
   every response and described in the agent instructions.
3. **Source assessment.** Classify primary, secondary, aggregator and background
   sources; carry source quality to each claim; base readiness on quoted claims
   rather than lexical query coverage.
4. **Claim checks.** Number/date/name consistency between claims and quotes,
   source-disagreement flags and a counterevidence gate that requires assessed
   counterevidence or an explicit "none found" unknown.
5. **Compact responses.** Delta results with inline passages, measured with real
   host usage; one shared report document for Markdown, JSON and PDF.
6. **Gates.** Freeze the replacement retrieval gate before running it, then run
   the approved paired host evaluation.

## Non-goals for this refactor

Do not add an independent chat UI, internal research/judge model, model routing,
multi-agent orchestration, financial prediction/analysis tools, SWOT/timeline
generators, automatic publication or a hosted multi-tenant service. Do not promise
exhaustive research, zero hallucinations, universal MCP-host compatibility or
production readiness without evidence. OCR, browser automation, automatic reference
graph traversal and refreshed report editions remain separate future decisions.

## Repository language and publication

English is canonical for README.md, design and operational documents, tool
descriptions, maintained code comments/messages, metadata, the GitHub About
description, and maintainer-authored issues, PRs and release notes.
README.zh-CN.md is the only maintained Chinese documentation file. Keep its
capabilities, installation steps, release status and limitations aligned with the
English README. Do not maintain a second translated planning tree.

This is a documentation language policy, not a restriction on research languages.
Preserve original-language evidence, proper names, multilingual test inputs and
matching literals where changing them would alter behavior or historical results.
Do not translate or delete raw evidence to make a language scan pass.

Publication requires working installation, retained engineering/security gates,
honest channel metadata and the applicable research-quality gates. Any claim of
superiority must be scoped to a predeclared paired evaluation with the same host
model, provider access and budget; installation success alone is not such evidence.
