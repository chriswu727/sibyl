# Sibyl roadmap

Reviewed 2026-10-21. This is the active roadmap for the unreleased 0.4.0
development line. Implemented features are documented in [Usage](USAGE.md);
release blockers are governed by [Launch readiness](LAUNCH_READINESS.md).
The [earlier implementation plan](https://github.com/chriswu727/sibyl/blob/d343267/docs/QUALITY_BUILD_PLAN.md)
is preserved in Git history, not a source of current defaults or quality claims.

## Implemented baseline

- Persistent host-assisted investigations with multi-round retrieval, question
  reviews, counterevidence requirements, explicit gaps, and partial reports.
- Shared source/passages contracts, hashes, offsets, provenance, date hints,
  duplicate-content clusters, and source inspection.
- Optional configured-model research with bounded reflection and visible
  source-support audit limitations; no hidden host sampling or subscription reuse.
- Optional text-PDF reads through the safe fetch boundary, page markers,
  bounded parsing workers, and explicit extraction limitations.
- Private SQLite checkpoints, cancellation/resume, retained budgets on oversized
  checkpoints, bounded exports, and single-runtime ownership.
- Authenticated loopback monitoring, content-free metrics, structured stderr
  events, and documented recovery procedures. Monitoring is opt-in; an alert
  service and dashboard are not installed automatically.
- Consistent formatting/import checks and typed core job/state/review contracts.
- Recorded-source full MCP workflow replay and clean-wheel core,
  PDF/monitoring, and report/finance/extraction verification.
- Bounded fault tests for abrupt process termination, failed writes before/after
  retrieval dispatch, transaction rollback, repeated cancellation, and export cleanup.
- HTML title fallback, whitespace normalization and truncation warnings; stable
  cited-source/background grouping across report renderers.
- A bounded runtime soak with real SQLite, optional monitoring/PDF workers,
  injected upstream failures, resource sampling, and stopped-store backup/restore.
- Preferred-domain discovery and relevance ranking before the bounded fetch limit.
- A ten-case live host research suite with adaptive follow-ups, inspected evidence,
  explicit discovery failures, and verified Markdown/JSON delivery.

## Before release

### 1. Establish a current research-quality baseline

The [ten-case host baseline](examples/host-suite/README.md) is complete: frozen
questions/rubric, live production MCP, evidence inspection, gap-led follow-ups,
reviews, and ten verified Markdown/JSON deliveries. Nine reports are structurally
complete and one is intentionally partial. Seven investigations required
host-supplied URLs; do not count them as successful automatic discovery. The
record is host-authored and self-scored, not independent evaluation.

The remaining model-backed evaluation concerns the optional autonomous path,
not a requirement to repeatedly rebuild the host suite. It needs an explicit
model/configuration and authorized API budget. Any future comparison with
ordinary search must hold model, source access, and budget constant.

### 2. Fix measured retrieval and report failures

Prioritize false-ready states, missing query anchors, concentrated or duplicated
sources, ineffective follow-ups, and conclusions that overlook PDF truncation.
Turn each reproducible failure into an offline regression. Preserve unknowns
and unsupported claims rather than weakening the readiness checks.

Acceptance: targeted regressions pass, followed by the committed 66-case,
three-repeat live retrieval gate with its existing thresholds. The small research
baseline supplements that gate; it does not replace or waive it.

### 3. Retain the verified runtime boundary

Keep the bounded runtime soak passing. The current fixture covers repeated
upstream failure, cancellation, timeouts, descriptor/task/file growth, worker
cleanup, and backup/restore with the runtime stopped. It is not an RSS benchmark
or a long-running public-provider availability test.

Longer-duration scenarios and additional metrics are follow-up enhancements,
not a reason to reopen the completed runtime work. Possible metrics include fetch outcomes, parser
failures, investigation rounds, and insufficient-evidence rates. Keep questions,
URLs, job IDs, and evidence out of metric labels.

Acceptance: reproducible fault scenarios, documented recovery outcomes and
resource measurements. Optional live scraping, alerts, or a dashboard require an
operator-selected deployment and retention policy; a runbook example is not a
deployed monitoring service.

### 4. Review dependencies and distribution compatibility

Review dependency proposals individually. Major MCP SDK and data-library updates
need dedicated compatibility tests; a green dependency audit alone is insufficient.
Keep the implemented core, PDF/monitoring, and report/finance/extraction clean-wheel
checks passing for each release candidate. Extend adversarial input tests without claiming prompt-injection
immunity or treating a parser process as a security sandbox.

Acceptance: supported Python matrices, clean-wheel checks, audit results, and
documented upgrade behavior. No release/version changes merely to clear a queue.

### 5. Complete release validation and publish

Retain the test method, results, and limitations. Satisfy every release gate before
dating the changelog, tagging, publishing PyPI/Registry metadata, or expanding
directory promotion. Follow [Releasing](RELEASING.md).

Acceptance: recorded evaluation, all required checks, and consistent GitHub,
PyPI, Registry, Glama, and product-page descriptions. Local engineering checks
do not establish live research accuracy.

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
