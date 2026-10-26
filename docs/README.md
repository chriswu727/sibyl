# Sibyl documentation

These documents follow main. The host-only architecture, default agent profile
and research views are unreleased;
the published 0.4.0b1 beta does not contain them. PyPI 0.3.0 is an older stable package.
See [installation](../llms-install.md) and [channel status](RELEASING.md). Sibyl is a local
deep-research MCP tool for AI agents, with multi-round investigations, traceable
evidence, gap tracking, and optional reports; its current maturity is public beta.

## Start and use

| Document | Purpose |
|---|---|
| [Product overview](../README.md) | Core workflow, capabilities, quick start, and limits |
| [MCP installation](../llms-install.md) | Source installation, host configuration, optional extras |
| [Usage reference](USAGE.md) | Eight default tools, compatibility profile, research jobs, configuration, and export |
| [Agent guide](../AGENTS.md) | Evidence-handling rules and the host research procedure |
| [Agent protocol](agent-protocol.md) | Compact briefs, evidence reads, source pagination, and compatibility |
| [Claim contract](research-claims.md) | Facts, reasoning, recommendation criteria, dependency freshness and report generation |
| [Complete workflow example](examples/sqlite-wal/README.md) | Recorded-source MCP replay, gap investigation, and exported report |
| [Configuration example](../sibyl.yaml.example) | Compatibility retrieval settings; no model credentials |

## Operate and extend

| Document | Purpose |
|---|---|
| [Operations](OPERATIONS.md) | Authenticated local monitoring, metrics, recovery, retention |
| [Architecture](MCP_ARCHITECTURE.md) | Module ownership, lifecycle, persistence, extension points |
| [Security](../SECURITY.md) | Network, parser, storage, authentication boundaries and reporting |
| [Privacy](../PRIVACY.md) | Local retention, provider disclosures, credentials, telemetry policy |
| [Performance](PERFORMANCE.md) | Resource controls and reproducible measurement |
| [SourceBundle 1.6](source-bundle-1.6.md) | Structured evidence consumer contract |
| [SourceBundle fixture](source_bundle_1_6.example.json) | Machine-validated serialization example |
| [EvidenceLoop 1.0](evidence-loop-1.0.md) | Short-lived atomic retrieval workflow, distinct from persistent jobs |

## Evaluate and release

| Document | Purpose |
|---|---|
| [Implementation plan](QUALITY_BUILD_PLAN.md) | Complete user workflow, connected capability changes and product completion criteria |
| [Refactor goals](REFACTOR_GOALS.md) | Functional target, evidence policy, recommendation requirements, non-goals and language policy |
| [Launch readiness](LAUNCH_READINESS.md) | Dated engineering evidence and unmet release gates |
| [Live host self-evaluation](examples/idempotency-live/README.md) | Failed source discovery, successful direct reads, and an explicit partial report |
| [Ten-case live host suite](examples/host-suite/README.md) | Adaptive investigations, cited findings, source-discovery limitations, and verified exports |
| [Historical host experiment](EVAL_HOST_CLAUDE.md) | Exploratory results and methodological limitations |
| [Contributing](../CONTRIBUTING.md) | Development checks and documentation requirements |
| [Release runbook](RELEASING.md) | Gated packaging, publication, and post-release validation |
| [Changelog](../CHANGELOG.md) | Development changes and previous releases |
| [Registry metadata](../server.json) | Next-version publication manifest; not proof of a published release |

Current behavior belongs in usage/operations documentation, planned behavior in
the roadmap, and dated measurements in readiness/evaluation records. Passing
engineering checks does not establish semantic correctness or report-quality
superiority. No new model evaluation or package publication is implied by a
documentation update.
