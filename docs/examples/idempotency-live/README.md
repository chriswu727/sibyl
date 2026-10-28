# Live self-evaluation: API idempotency

Date: 2026-10-21. Code baseline: `29147979bb396c863d1edd196027918974a52673`.
This is one live host-assisted investigation over the real keyless MCP stdio
server, not an autonomous-model benchmark.
The [unaltered Markdown export](report.md) is deliberately a **partial report**.

Question: do idempotency keys make retries safe indefinitely, and how do API
scope and retention limits compare?

## Method and observed result

The host used `start_research` with a four-attempt budget, primary-source policy,
and preferred domains `aws.amazon.com` and `docs.stripe.com`. No model provider
or search API credentials were supplied. The host was the Codex agent conducting
this evaluation; it planned and authored the report in the existing conversation.
There was no separate Codex/Claude process, autonomous model call, or paid search.

| Round | Action | Observed outcome |
|---|---|---|
| 1 | Search `API idempotency keys retries retention expiration limitations Stripe AWS` | Four background encyclopedia pages; `insufficient_evidence` |
| 2 | Search `Stripe idempotent requests 24 hours site:docs.stripe.com` after an unresolved review | No sources; `insufficient_evidence` |
| 3 | Host supplies Stripe's official idempotent-requests URL as counterevidence | Direct read succeeds; the API v1 retention caveat is present |
| 4 | Host supplies the AWS Builders Library idempotency URL | Direct read succeeds; useful request-identity passages but an incomplete bounded extract |

The host reassessed the evidence as unresolved, then explicitly requested
`allow_incomplete=true`. Sibyl finalized the job with `status="completed"` while
preserving `report.status="partial"`, `verification_scope="citation_targets_only"`,
the unresolved review, and the failed-search limitations. Markdown and JSON
exports succeeded. No writes to Stripe or AWS APIs were attempted.

## Assessment

- **Discovery: failed for this question.** Both useful URLs came from the host,
  not Sibyl's search results. Domain preference is not an allowlist or a promise
  that matching sources will be found. Do not count the final report as a search win.
- **Evidence reading: useful but incomplete.** Stripe's relevant passage was
  readable. The AWS extract included large code/whitespace regions before the
  later discussion and retained the placeholder title `references-details-empty`.
- **Workflow and delivery: passed this narrow live exercise.** Failed rounds,
  review updates, budget exhaustion, explicit incomplete completion, and exports
  were preserved without inventing a complete comparison.
- **Report quality: useful partial answer, not a release-quality report baseline.**
  The conclusion separates Stripe API v1 from API v2 and does not fabricate an
  AWS retention interval. Uncited encyclopedia background still appears in the
  export's reference inventory; readers must not mistake it for supporting evidence.

This self-assessment is qualitative and host-authored. No semantic grader,
experimental API test, independent reviewer, or model-comparison control was used.
Do not infer a success rate, cost estimate for autonomous mode, or superior
research performance from one investigation.

## Follow-up before stronger quality claims

Investigate provider-specific search failures and irrelevant fallback candidates;
improve source-title selection and bounded extraction around code-heavy pages;
distinguish cited evidence from uncited background in report presentation. Repeat
this question and a broader frozen suite after fixes. Keep the failed run intact.

### Engineering follow-up

The subsequent extraction fix resolves the placeholder title, normalizes excessive
whitespace, and reports HTML truncation. A fresh AWS direct read confirmed the
title fix; the bounded read still did not cover the entire article. New report
exports separate cited sources from uncited background without changing citation
numbers. The report above is the original export and has not been rewritten.
These targeted fixes do not establish that search discovery now succeeds.
