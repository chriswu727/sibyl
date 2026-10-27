# Focused live report examples

These are concise presentations of two host-authored reports from the
[ten-case suite](README.md), with source links rather than reproduced source
excerpts. The [retained record](../../../evals/results/host-research-2026-10-21.json)
includes the complete findings, review ledger, source hashes, and limitations.
Markdown and JSON exports were verified locally for all ten cases.

## Semiconductor sales: preserve the revision

The inspected SIA release reports an initial 2024 global sales total of
$627.6 billion and 19.1% growth, but its March 3, 2025 correction raises the
year-end total to **$630.5 billion**. The revised total must not be replaced by
the old figure still present in the headline and body. The report does not infer
a revised growth percentage from an unverified denominator.
[SIA release and correction](https://www.semiconductors.org/global-semiconductor-sales-increase-19-1-in-2024-double-digit-growth-projected-in-2025/).

The WSTS spring and fall forecasts projected $611 billion and $627 billion,
respectively. They are separate forecast vintages, not final annual observations.
SIA uses WSTS data, so the sources are not independent measurement systems.
[Spring forecast](https://www.wsts.org/76/103/WSTS-Semiconductor-Market-Forecast-Spring-2024),
[fall forecast](https://www.wsts.org/76/103/WSTS-Semiconductor-Market-Forecast-Fall-2024).

Discovery limitation: search found forecasts but missed the year-end release;
the host supplied the SIA URL. The report reflects the correction visible in
that inspected page, not a guarantee that no later revisions exist.

## Idempotency: safe retries have a scope

An idempotency key is not an indefinite duplicate-suppression guarantee.
Stripe API v1 allows pruning after keys are at least 24 hours old; reuse after
pruning creates a new request. Parameter matching also matters. The report
does not extend this API v1 retention statement to API v2.
[Stripe idempotent requests](https://docs.stripe.com/api/idempotent_requests).

AWS Powertools likewise defines an idempotent repeat in terms of an unexpired
stored request. Its retention behavior is not a universal contract for every
AWS API.
[AWS Powertools idempotency](https://docs.aws.amazon.com/powertools/python/3.4.0/utilities/idempotency/).
The Builders Library explains why caller-provided request identity matters:
identical parameters can describe distinct intended operations.
[AWS retry design](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/).

Discovery limitation: AWS material was found through search; Stripe required a
host-supplied URL. This was a documentation investigation, not an execution of
payment or infrastructure operations.
