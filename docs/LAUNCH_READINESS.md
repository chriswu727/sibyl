# Sibyl launch readiness

Reviewed 2026-10-21 against engineering baseline
[`d343267`](https://github.com/chriswu727/sibyl/commit/d343267).
This record separates implemented behavior, verified engineering checks, and
unmet product-quality gates. See the [roadmap](QUALITY_BUILD_PLAN.md) for execution
order and [Releasing](RELEASING.md) for publication mechanics.

## Decision

Sibyl remains a public beta. The source checkout identifies itself as 0.4.0
development; the latest published release is 0.3.0. Do not publish 0.4.0 or present
current development APIs as available from PyPI until the gates below are met.
Updating documentation or passing CI does not waive those gates.

## Current engineering evidence

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

Current behavior includes persistent host investigations, per-question reviews,
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

### Current model-free retrieval gate, 2026-10-21

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

Local verification passed 450 tests, scoped Ruff formatting/lint, the five-module
mypy gate, offline evaluation gates, real MCP/monitoring checks, the PDF-enabled
workflow replay, package metadata checks, and a clean core-wheel installation.
The clean-wheel environment was removed after verification. The other wheel
profiles are covered by the preceding engineering baseline and CI; they were not
rerun locally for this discovery-only change.

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

Do not publish v0.4.0, update the official MCP Registry, or begin broad directory
promotion until all five gates are satisfied:

1. A new dated 66-case, three-repeat live retrieval run passes every committed
   threshold. Freeze thresholds before inspecting results.
2. The optional report path completes a bounded model-backed evaluation with
   declared model/configuration, rubric, cost, and retained results. Explicit
   model-quota/API-spend approval is required. Include the host research workflow
   in the representative research-quality baseline.
3. The release wheel passes clean core and report-extra installation smoke tests;
   also exercise the new PDF/monitoring extras and check installed dependencies.
4. Required GitHub CI passes on Python 3.11–3.13 at the release commit.
5. README, package metadata, Registry metadata, and the product page agree on
   capabilities, maturity, installation, version, and limitations.

A green release workflow automates packaging and publishing; it does not enforce
all research-quality gates. The maintainer must review this checklist before
publishing a GitHub Release. The version validator checks metadata agreement,
not research quality.

## Claims supported now

- A local Deep Research MCP for multi-round investigations and traceable reports.
- Keyless host-assisted research with explicit evidence reviews and limitations.
- Versioned passage provenance, duplicate-content signals, and bounded text-PDF reads.
- Authenticated operational monitoring and tested local runtime controls.
- Optional configured-model research with explicit costs and audit limitations.

Not established: exhaustive web coverage, production-grade anonymous-search
availability, guaranteed citation truth, automatic peer-review/authority
verification, or superiority over a strong model using ordinary web search.
