# Sibyl host agent research goals

Recorded 2026-10-26. This is the functional target for the current refactor,
not a description of features that have all shipped. The implementation baseline
is [314e721](https://github.com/chriswu727/sibyl/commit/314e721), which removed
internal model execution. The [implementation plan](QUALITY_BUILD_PLAN.md)
connects the requirements into a complete product; [launch readiness](LAUNCH_READINESS.md#release-gates)
continues to govern publication.

## Product purpose

Sibyl is a local research MCP for an AI agent the user already uses. It helps that
agent investigate a question through public sources, inspect exact evidence,
pursue counterevidence, preserve conclusions and gaps, and optionally export a
human-readable report. The primary output is reusable research state, not a long
document or an independently generated answer.

The intended advantage over a single search call is continuity and accountability:
each investigation connects a question, retrieved evidence, host judgment,
remaining gaps and the reason for the next retrieval. More sources, more rounds
or more prose do not by themselves establish better research.

## Responsibility and subscription boundary

The host agent understands the request, decomposes questions, selects queries,
reads evidence, judges support, chooses tradeoffs and writes the final answer.
Sibyl retrieves and stores evidence, exposes structured research state, checks
reference integrity, enforces resource limits and exports host-authored reports.

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

Semantic entailment, source credibility and recommendation quality remain host
judgments evaluated separately. Claim-to-evidence records distinguish
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
startup check. Maintain the eight-tool default and the text-oriented keyless
compatibility surface until measured host usability justifies a change.

Acceptance: a fresh installation can discover tools, start/read/cancel a job and
complete a recorded evidence workflow without model SDKs or secret provisioning.
Real-host usability must be tested separately from a synthetic MCP client.

### G2 Source discovery and reading

Improve the ability to find topic-appropriate original sources before spending
the fetch budget. Preserve source identity, canonical URL, retrieval time,
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
round counter. Repeated searches and duplicate publishers must remain visible.

Acceptance: a replay preserves the path from gap to retrieval to revised review
across process restart. Stale judgments cannot silently retain an old readiness
decision. Cancellation and failures retain evidence and consumed budgets.

### G4 Claim support and evidence-based recommendations

Extend the existing question-review ledger only where needed to represent the
statement distinctions above, material claim-to-passage links, assumptions,
conflicting evidence and recommendation criteria. Reuse stable evidence IDs and
revision protection instead of creating a separate report-only evidence store.

Acceptance: controlled cases reject nonexistent references and stale submissions;
unsupported material claims are omitted or explicitly unresolved in Sibyl-managed
deliverables. Recommendations expose which factual premises support each tradeoff
and which criteria depend on user preference. Semantic support is scored through
evaluation, not inferred from schema validity or keyword overlap.

### G5 Agent-facing results and stopping behavior

Return a compact, self-contained current brief: questions, reviews, gaps, relevant
evidence IDs, latest retrieval outcome, remaining budget and actionable next steps.
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

| Area | Status at the recorded baseline |
|---|---|
| Internal model removal and eight-tool default | Implemented on main; not a new PyPI release |
| Structured briefs, exact passage reads and source pagination | Implemented |
| Persistent reviews, counterevidence workflow and explicit gaps | Implemented; host-declared judgments |
| Citation target and revision validation | Implemented; not semantic verification |
| Markdown/JSON and optional PDF export | Implemented |
| Original-source discovery reliability | Measured weakness; improvement required |
| Explicit claim/inference/recommendation contract | Implemented in the source preview; host-declared semantic support |
| Natural tool use inside real subscription-backed hosts | Requires a recorded compatibility/usability evaluation |
| Better research than built-in search | Hypothesis; paired evaluation not established |

The retained ten-case host suite needed host-supplied URLs in seven cases. The
dated live retrieval gates still have failures. See [launch evidence](LAUNCH_READINESS.md)
for the runs and their scope. Neither architecture simplification nor offline
tests erase those limitations.

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
