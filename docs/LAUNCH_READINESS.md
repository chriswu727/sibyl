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
promotion until all six gates are satisfied:

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
6. At least three external beta users independently install Sibyl, complete a
   representative task, and report whether evidence states and citations were clear.

A green release workflow automates packaging and publishing; it does not enforce
all human/live-quality gates. The maintainer must review this checklist before
publishing a GitHub Release. The version validator checks metadata agreement,
not research quality or beta completion.

## Claims supported now

- A local Deep Research MCP for multi-round investigations and traceable reports.
- Keyless host-assisted research with explicit evidence reviews and limitations.
- Versioned passage provenance, duplicate-content signals, and bounded text-PDF reads.
- Authenticated operational monitoring and tested local runtime controls.
- Optional configured-model research with explicit costs and audit limitations.

Not established: exhaustive web coverage, production-grade anonymous-search
availability, guaranteed citation truth, automatic peer-review/authority
verification, or superiority over a strong model using ordinary web search.
