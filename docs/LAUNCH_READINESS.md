# Sibyl launch readiness

Historical engineering baseline reviewed 2026-10-21 against
[`d343267`](https://github.com/chriswu727/sibyl/commit/d343267).
This record separates implemented behavior, verified engineering checks, and
unmet product-quality gates. See the [roadmap](QUALITY_BUILD_PLAN.md) for execution
order and [Releasing](RELEASING.md) for publication mechanics.

## Host-only scope update — 2026-10-26

The current source preview removes internal model execution and the finance
surface. Default MCP now exposes eight agent tools; the host owns planning,
semantic judgment and writing. Human reports remain optional. This change is not
a new PyPI publication or proof of better retrieval.

The internal-model evaluation gate is retired because that execution path no
longer exists. Its replacement is a paired same-host, same-budget research
comparison against ordinary search, with explicit spend authorization. Live
retrieval thresholds remain unchanged. The historical records below describe
the architecture and checks at their dated commits, not current capabilities.

## Historical beta decision — not current release authority

On 2026-10-21 the maintainer approved an opt-in **0.4.0b1 prerelease**, focused
on the keyless host-assisted MCP workflow. Configured-model autonomous research
remains experimental. This is an explicit scope decision, not a passing result
for the live research-quality gates. The original thresholds and failed runs
remain unchanged. Stable 0.4.0 is not cleared by this decision.

Beta publication requires the engineering, security, clean-install, and honest
documentation checks below. See [channel status](RELEASING.md#publication-channels)
for publication progress; version metadata alone does not prove availability.

## Historical engineering evidence — 2026-10-21

[PR #22](https://github.com/chriswu727/sibyl/pull/22) delivered bounded PDF input,
authenticated local monitoring, checkpoint recovery, and artifact quotas.

| Check | Verified result at the baseline |
|---|---|
| Local regression suite | 408 tests passed on Python 3.12 |
| PR CI | 9 successful jobs: 3 Python test/build jobs and 6 dependency-audit jobs |
| Supported CI matrix | Python 3.11, 3.12, 3.13 |
| Lexical and retrieval-pipeline fixtures | 8/8 each |
| Contextual source-quality baseline | 5 correct, 1 incorrect, 2 abstentions; committed gate passed |
| Real MCP stdio | Keyless lifecycle and 12-tool catalog passed |
| Real monitoring transport | Bearer authentication, invocation metrics, and shutdown cleanup passed |
| Live source-read integration | One arXiv PDF produced 10 passages with page markers and a 16,000-character truncation warning |

The live PDF check is a narrow integration result, not a complete-paper or
research-quality benchmark. Offline model tests use mocks. Dependency audits
cover resolved core and report/finance/extract/pdf/monitor installations, not
every possible optional environment or unknown vulnerability.

Behavior at that baseline included persistent host investigations, per-question reviews,
counterevidence requirements, explicit partial reports, and optional autonomous
research. Host completion checks citation targets and workflow structure, not
semantic entailment. Autonomous model audits are fallible. Monitoring is read-only
and local; no external collector, dashboard, or notifications are deployed by default.

### Engineering follow-up, 2026-10-21

The engineering-readiness follow-up adds a repository-wide format/import/error
gate, static checking for five core job/state/review modules, and a complete
[recorded-source MCP replay](examples/sqlite-wal/README.md). Local verification
passed 424 tests on Python 3.12, including real process-kill recovery, write-failure
injection, revision rollback, repeated capacity/cancellation, and artifact cleanup.
Three isolated wheel profiles passed on macOS/Python 3.12: core, PDF/monitoring,
and report/finance/extraction. Checks include real stdio, authenticated monitoring,
PDF input/output, report export and restart. Each temporary environment is removed.

The publication workflow now uses the same wheel checks and rejects an undated
changelog. The source test matrix remains Python 3.11–3.13; clean-wheel execution
in CI runs on Python 3.12. This does not establish all-platform compatibility,
full-repository strict typing, sustained-load reliability, or new research accuracy.
The original live-quality gates remain unmet. Numerical quality thresholds and
model-spend approval requirements are unchanged.

### Live host self-evaluation, 2026-10-21

A [four-attempt idempotency investigation](examples/idempotency-live/README.md)
used the production keyless MCP server with live public sources. Two search
rounds did not produce useful primary evidence; two host-supplied official URL
reads succeeded. Review, budget exhaustion, explicit partial completion, and
Markdown/JSON export worked. The report retained unresolved gaps and failed
search limitations. AWS title/extraction quality and unrelated background in
the reference inventory remain follow-up issues. This single host-authored
exercise is not a successful discovery benchmark, an independent review, or
the required autonomous model-backed report evaluation.

### Model-free retrieval gate, 2026-10-21

The [new retained run](../evals/results/live-retrieval-keyless-2026-10-21.json)
used baseline `29147979bb396c863d1edd196027918974a52673`, metric version 2,
66 cases with three repeats, concurrency 2, lexical ranking, and the explicit
keyless search provider. Default thresholds were not changed. No retrieval
code changed during the run. Local engineering checks and one small host
investigation shared the machine; this is not a controlled latency benchmark.

| Metric | Observed | Frozen gate | Result |
|---|---:|---:|---|
| Answer coverage | 83.3% | 80% | Pass |
| Trap-safe outcomes | 91.7% | 80% | Pass |
| Repeat stability | 62.1% | 90% | Fail |
| Answerable ready rate | 63.6% | 75% | Fail |
| Ready-state answer precision | 97.2% | 95% | Pass |
| p95 retrieval latency | 10.6 s | 30 s | Pass |

The command exited 1 with `passed=false`. This run does not clear publication.
Answer hits use gold/alias text matching, not semantic report grading. A trap
can count safe because Sibyl abstained or because a refutation alias appeared;
91.7% is not a guarantee against false answers. Read the per-case raw records.
The autonomous model-backed gate has not been run without spend authorization.

### Search-budget follow-up, 2026-10-21

The [post-fix run](../evals/results/live-retrieval-keyless-budget-fix-2026-10-21.json)
uses the same 66 cases, three repeats, concurrency, metric implementation, and
unchanged thresholds. The source baseline above plus the search/constraint
fixes is identified by these SHA-256 values:

- `sibyl/search.py`: `6718e54fac1c33b78c384adf5c3288e77419b94b5adbf3f4f035e90129f32c6a`
- `sibyl/queries.py`: `6c244e042388c5920c24464fcea2ed92a15adacf81db474053dd36701886f949`

| Metric | Before | After | Frozen gate |
|---|---:|---:|---:|
| Answer coverage | 83.3% | 83.3% | 80% |
| Trap-safe outcomes | 91.7% | 91.7% | 80% |
| Repeat stability | 62.1% | 68.2% | 90% |
| Answerable ready rate | 63.6% | 66.0% | 75% |
| Ready-state answer precision | 97.2% | 97.3% | 95% |
| p95 retrieval latency | 10.6 s | 12.5 s | 30 s |

The follow-up also exited 1 with `passed=false`: readiness and stability remain
below their gates. These sequential public-web runs are not a controlled A/B
experiment; observed differences cannot be attributed solely to the code fix.
No cache was added, no sufficiency rule was relaxed, and no paid provider or
autonomous model was invoked. A narrow provider probe observed a DuckDuckGo
challenge, Mojeek HTTP 403, and a transient Yahoo HTTP 500 followed by successful
Yahoo responses; it does not quantify provider availability across the suite.

Engineering verification passed 430 tests, Ruff, five-module mypy, all existing
offline evaluation gates, the PDF-enabled research replay, real MCP/monitoring
smoke checks, and all three clean-wheel profiles. An OSV dependency audit of the
local environment found no known vulnerabilities; the editable project itself
was skipped. This is not a proof that the code has no vulnerabilities.
Temporary wheel environments and the type-check cache were removed.

The confirmed fixes reserve time for fallback engines, bound the configured
provider before fallback, await child-task cleanup on cancellation, and preserve
explicit search constraints. Reliable provider availability and the outstanding
report-quality evaluation still require a release decision or further work.

### Extraction and runtime follow-up, 2026-10-21

HTML title selection now rejects placeholder titles and falls back to metadata
or headings. Prose and excessive preformatted whitespace are normalized before
bounded extraction, with warnings for modified preformatted text and truncation.
A fresh direct read of the AWS idempotency article returned its correct title;
the 16,000-character read still truncated the long article. This is not a rerun
of the full retrieval benchmark or a claim of complete source coverage.

Markdown, PDF, and plain-text reports now separate cited sources from uncited
background, retaining original citation numbers and unchanged JSON evidence.
The renderer fixture passed PDF text checks and a visual review. Partial reports
also retain their content and warning in plain-text output.

The [bounded runtime soak](../evals/results/runtime-soak-2026-10-21.json) passed
300 measured rounds with two workers, repeated injected upstream failures,
timeouts/cancellation, live loopback monitoring, 31 malformed-PDF worker calls,
and stopped-store backup/restore. No tasks or child processes survived shutdown;
descriptor and disk counts stayed bounded. See [Performance](PERFORMANCE.md#bounded-runtime-recovery-check)
for measurements and scope. These checks do not clear the unmet live-quality gates.

Local verification passed 443 tests, Ruff, the five-module mypy gate, existing
offline retrieval/source-quality gates, real MCP/monitoring smoke checks, and
the PDF-enabled research replay. Fresh wheel/sdist builds passed metadata checks.
All three clean-wheel installation profiles passed. The local dependency audit
found no known vulnerabilities, and dependency consistency checks passed.

### Discovery follow-up, 2026-10-21

Preferred domains now participate in bounded discovery searches, and candidates
are ranked before the fetch limit rather than only after page retrieval. The
[retained 66-case, three-repeat run](../evals/results/live-retrieval-discovery-2026-10-21.json)
used unchanged metric-v2 thresholds, keyless search, and concurrency two.
Its retrieval implementation SHA-256 was
`b2d00c4e49006f1738c28e35b65483fcaf4fefd510d57ec932c4347dfa177f5b`.

| Metric | Observed | Frozen gate | Result |
|---|---:|---:|---|
| Answer coverage | 87.0% | 80% | Pass |
| Trap-safe outcomes | 91.7% | 80% | Pass |
| Repeat stability | 60.6% | 90% | Fail |
| Answerable ready rate | 60.5% | 75% | Fail |
| Ready-state answer precision | 99.0% | 95% | Pass |
| p95 retrieval latency | 11.1 s | 30 s | Pass |

The command exited 1; publication is not cleared. These sequential live runs
are not controlled A/B experiments. Higher answer coverage and lower readiness
can coexist because retrieved answer text alone does not satisfy every source
sufficiency rule. No thresholds were weakened and no cache was introduced to
make repeat outcomes identical. The generic benchmark supplies no preferred
domains; the separate host suite exercises that part of the discovery change.

The [ten-case live host suite](examples/host-suite/README.md) completed real MCP
investigations with inspected evidence, adaptive follow-ups, reviews, and verified
Markdown/JSON exports. Nine reports are structurally complete and one intentionally
partial. Seven cases needed host-supplied source URLs; this is not a discovery
success rate or an independent semantic evaluation. The optional autonomous
model-backed gate remains separate and unrun.

Local verification passed 452 tests, scoped Ruff formatting/lint, the five-module
mypy gate, offline evaluation gates, real MCP/monitoring checks, the PDF-enabled
workflow replay, package metadata checks, and a clean core-wheel installation.
The clean-wheel environment was removed after verification. The other wheel
profiles are covered by the preceding engineering baseline and CI; they were not
rerun locally for this discovery-only change.

The initial PR CI run exposed a timing-dependent soak assertion. Non-timeout
cycles now have a separate deadline budget; timeout cycles accept either the
inner job's failed-attempt response or the outer runtime's `TimeoutError`.
Cancellation and unrelated errors remain strict, with two additional regressions.
No worker, evidence, budget, file, descriptor, or shutdown assertions were removed.
A new 300-round local run with monitoring and 31 PDF-worker calls passed in
28.944 seconds, retaining 146,779 traced Python bytes and seven final descriptors.

## Historical live evidence

The retained formal runs are from July 2026 and do not measure the current
persistent research workflow. Keep their original outcomes and dates intact.

| Metric | Keyless, three repeats | Tavily pilot, one repeat | Existing gate |
|---|---:|---:|---:|
| Answer coverage | 85.2% | 88.9% | 80% |
| Trap-safe outcomes | 100% | 100% | 80% |
| Answerable ready rate, metric v2 | 66.0% | 70.4% | 75% |
| Ready-state precision, metric v2 | 93.9% | 92.7% | 95% |
| Repeat stability | 62.1% | Not measured | 90% |
| p95 retrieval latency | 11.0 s | 10.7 s | 30 s |

The keyless run fails readiness, precision, and stability. The Tavily pilot fails
readiness and precision; a single repeat cannot measure stability. The pilot
consumed 105 basic search requests. These results do not justify a new quality
claim or automatically authorize another paid-provider run.

Retained evidence:

- [Original keyless baseline](../evals/results/live-retrieval-2026-07-21.json).
- [Three-repeat post-Crossref run](../evals/results/live-retrieval-keyless-post-crossref-2026-07-21.json).
- [Tavily pilot](../evals/results/live-retrieval-tavily-pilot-2026-07-21.json).
- [Four-case fixed-plan evidence-loop run](../evals/results/evidence-loop-keyless-2026-07-21.json):
  4/4 ready with answer evidence, one repeat with hand-authored queries; not host
  planning or report synthesis validation.
- [Exploratory host-model comparison](EVAL_HOST_CLAUDE.md): small, agent-graded,
  model/configuration-confounded; not a current same-model comparison.

The original live JSON uses metric version 1. Its `ready_bundle_rate` counts
`ok` outcomes without requiring answer evidence. Metric v2 instead evaluates
answerable cases with both `ok` and expected answer evidence, and measures the
precision of ready states. The v2 figures above are derived from retained case
records, not new runs. Do not compare the two metric definitions as equivalent.

Anonymous-provider throttling, inconsistent source coverage, specialized facts,
and dependent questions contributed to the historical failures. Fixes must be
measured on new dated runs; neither extra features nor different branding prove
those failures are resolved.

## Release gates

The current target is a complete stable host-only release, not another beta.
The historical beta exception below applies only to the already published
0.4.0b1 artifact. It does not waive any gate for current `main` or authorize
republishing changed source under that version.

### Opt-in 0.4.0b1 beta (historical scope)

1. Release-commit CI, regression, lint, typed-core, dependency-audit, offline
   retrieval, MCP lifecycle, and authenticated monitoring checks pass.
2. Fresh release distributions pass metadata validation and all three clean-wheel
   profiles: core, PDF/monitoring, and report/finance/extraction.
3. Public documentation identifies the beta and experimental autonomous mode,
   accurately describes supported interfaces, and retains failed live metrics.
4. Publish a GitHub prerelease and an exact-version PyPI package. Verify public
   artifacts and the Registry manifest before reporting those channels complete.
   Directory reindexing and the website are tracked separately.

The retained ten-case host suite is evidence for the default workflow, not an
independent quality benchmark. No new paid model evaluation is authorized or
required for this scoped beta. Experimental status does not waive security or
dependency checks for the optional packages.

### Current stable host-only release

Do not publish a stable host-only version or claim the research-quality gates passed until
all five gates below are satisfied:

1. A new dated 66-case, three-repeat live retrieval run passes every committed
   threshold. Freeze thresholds before inspecting results.
2. A paired host-agent evaluation compares this workflow with ordinary search
   using the same model, provider access and budget, a declared rubric, and
   retained outputs. Record evidence support, gaps, abstention, cost and latency.
   Explicit model-quota/API-spend approval is required. A host-authored fixture
   alone does not establish research-quality superiority.
   Verify natural host recovery, exact support for material qualifications,
   counterevidence, follow-ups and honest unknown delivery. Existing partial
   exports and the unsupported SQLite qualification do not clear this gate.
3. The release wheel passes clean core and report-extra installation smoke tests;
   also exercise the new PDF/monitoring extras and check installed dependencies.
4. Required GitHub CI passes on Python 3.11–3.13 at the release commit.
5. README, package metadata, Registry metadata, and the product page agree on
   capabilities, maturity, installation, version, and limitations.

A green release workflow automates packaging and publishing; it does not enforce
all research-quality gates. The maintainer must review this checklist before
publishing a GitHub Release. The version validator checks metadata agreement,
not research quality.

Current clean-wheel profiles are core, PDF/monitoring and report/extraction;
there is no finance extra or internal-model evaluation path. Use a fresh,
not-yet-published version only after these gates pass. The historical beta
checklist and measurements retain their original architecture below/above;
they are not instructions to reintroduce removed capabilities.

### Low-effort real-host smoke, 2026-10-26

The [retained four-request record](../evals/results/host-agent-luna-low-2026-10-26.json)
used Codex CLI 0.163.0-alpha.10.2, GPT-6 Luna with low effort, the agent MCP profile
and anonymous retrieval under explicit subscription-quota approval. One initial
request failed on harness approval configuration; the corrected SQLite workflow
delivered real Markdown/JSON artifacts with explicit partial status. Both attempts
at the unknown-outcome task failed the requested report delivery. A host-invented
JSON link exposed ambiguous export semantics; clearer instructions did not solve
claim-error/revision recovery in the last run. The server rejected malformed/stale
submissions rather than inventing completed reports.

All four requests ended within one minute, but that is not a latency distribution.
No native-search comparator, cross-host run, exact monetary cost or new release
verdict was established. The approved four-request allowance is exhausted.
This is functional evidence and a retained usability failure, not a stable gate pass.

The current source now adds structured recovery for rejected claims and stale
revisions, plus confirmed per-format export manifests. Offline real-stdio
regressions cover reload/repair and explicit unknown-report export without
relaxing validation. This is an interface-contract improvement, not a new host
measurement: the failed Luna runs above have not been rerun, and their approval
does not authorize more model usage. Stable quality gates remain open.

### Current-source live gate, 2026-10-26

The [retained 66-case, three-repeat run](../evals/results/live-retrieval-body-selection-2026-10-26.json)
used the body-selection candidate merged in #37, anonymous keyless search, no
model calls and the unchanged frozen datasets/thresholds. It exited 1 with
`passed=false`. Its package version string is still 0.4.0b1; the record identifies
the actual source commit, not a newly published beta artifact.

| Metric | Result | Gate | Outcome |
|---|---:|---:|---|
| Answer coverage | 81.5% | 80% | Pass |
| Trap-safe rate | 100% | 80% | Pass |
| Repeat stability | 51.5% | 90% | Fail |
| Answerable ready rate | 57.4% | 75% | Fail |
| Ready-state answer precision | 97.9% | 95% | Pass |
| p95 retrieval latency | 11.0 s | 30 s | Pass |

This uses gold/alias matching, not semantic report scoring. Dated sequential runs
do not isolate changes in public search results or establish a causal improvement
or regression. The narrow reading regressions below passed, but current discovery
and readiness stability still block a stable release. Do not lower thresholds,
discard unsuccessful runs, or treat report/export improvements as clearing this gate.

Subsequent [three-case query tracing](../evals/results/query-identifiers-2026-10-26.json)
fixed contracted-question fallback and decimal-version preservation, but all
three after runs still lack a frozen gold/alias match. Two return heuristic status
ok, which is not answer correctness. No replacement full-gate pass or host-model
quality verdict has been established after those parsing changes.

## Claims supported now

### Claude Code paired smoke, 2026-10-27

The [retained record](../evals/results/host-claude-code-paired-2026-10-27.json)
used Claude Code 2.1.311 with claude-sonnet-5-5 at medium effort on source
[`7b7ed95`](https://github.com/chriswu727/sibyl/commit/7b7ed9568d464fc5797fc157550d0f9dc531974c),
under explicit approval. The two tasks from the previous retest were each run once
with Sibyl plus the host's built-in WebSearch/WebFetch, and once with built-in
search only. Each run was isolated, limited to 2 USD and 600 seconds, and had no
shell or file access. No human coached a run.

- SQLite WAL recommendation: with Sibyl the host read three sqlite.org pages,
  every supported fact carried a quote found verbatim in its passage, and it
  surfaced the WAL-reset corruption bug that matches two writers in separate
  processes. It left the exact `synchronous=FULL` guarantee unresolved. The
  baseline fetched the same WAL page but did not mention that bug.
- NVIDIA 2030 unknown: both runs refused to invent a realized share. With Sibyl
  the host used built-in search for discovery and Sibyl URL reads, as intended,
  but chose two weak sources and reported no forecast figure. The baseline
  reported four forecasts, of which a spot check found three in the cited pages
  and one not.
- Cost: the Sibyl runs used 4.5 to 5.6 times the CLI-reported cost, 2.5 to 3 times
  the time and 3 to 4 times the turns. One run recovered from `stale_revision`
  and from passing legacy prose fields to `complete_research`; the agent profile
  now hides those fields and rejects undeclared arguments.

This is one win and one loss from two tasks, one run each, scored without
blinding by a model of the host's family, on one host. It is not the declared
paired comparison or a release gate, and supports no superiority claim.

### Structured-recovery host retest, 2026-10-26

The [retained two-workflow record](../evals/results/host-agent-structured-retest-2026-10-26.json)
used Codex CLI 0.163.0-alpha.10.2, GPT-6 Luna low, anonymous search and the same
frozen source commit for both tasks, under fresh approval for two workflows of
at most 120 seconds each. No human supplied URLs or coached calls during a run.
Both delivered confirmed Markdown/JSON artifacts in 44.9 and 49.9 seconds, with
partial report status retained. The unknown case repaired a rejected supported
inference and did not invent a realized future result or saved paths.

Delivery is not semantic correctness. The SQLite recommendation's cited passage
supports reader/writer overlap but not its added single-writer qualification.
That qualification exists elsewhere in the same official document, outside the
selected excerpts for this broad query. Other inspected limits were supported,
but this material support gap prevents a research-quality pass. The record keeps
tool errors, hashes, usage counters, limitations and the unsupported passage ID.

Subsequent excerpt-scope metadata and read-more guidance have offline coverage,
not a new host result. The approved two-workflow allowance is exhausted. This is
not a paired comparator, cross-host test, full semantic benchmark or replacement
for the failed 66-case gate; stable release remains blocked by quality criteria.

### Body-selection regressions, 2026-10-26

The [two-case model-free real-MCP record](../evals/results/body-evidence-2026-10-26.json)
repeats the frozen SQLite WAL and Stripe/AWS retry queries without supplied URLs.
The inspected SQLite and AWS extracted-source hashes are unchanged between runs.
Before the change, their selected passages covered characters 0-4000 and omitted
the declared single-writer / retry-contract qualifications. After body-aware
multi-span selection, the SQLite excerpt includes the one-writer limit alongside
reader/writer overlap; AWS includes resource-lifetime retention and parameter
mismatch behavior. The record retains span offsets, hashes and phrase-match scope.

Selection remains lexical, not semantic verification. This small sequential live
check does not establish complete-source reading, a successful host-authored
investigation, or superiority over native search. URL reads now extract up to
60,000 characters but return at most 16,000 evidence characters; omitted text and
parser limits remain visible. The separate stable research-quality gates remain.

### Supported scope

- A local Deep Research MCP for multi-round investigations and traceable reports.
- Keyless host-assisted research with explicit evidence reviews and limitations.
- Versioned passage provenance, duplicate-content signals, and bounded text-PDF reads.
- Authenticated operational monitoring and tested local runtime controls.
- Host-only execution without internal model calls or subscription credential access.

Not established: exhaustive web coverage, production-grade anonymous-search
availability, guaranteed citation truth, automatic peer-review/authority
verification, or superiority over a strong model using ordinary web search.
