<!-- mcp-name: io.github.chriswu727/sibyl -->

# Sibyl

**A research MCP for your AI agent: investigate across sources, inspect evidence,
track counterevidence and gaps, and export a cited report when needed.**

Your agent does the thinking. Sibyl gives it a persistent research workspace.
Connect it to a local MCP-capable agent you already use, including one backed by
a subscription. Sibyl does not call another generative model, read subscription
credentials, or require a second model API key. The host's own usage rules still apply.

[Installation](#quick-start) · [Agent protocol](docs/agent-protocol.md) ·
[Documentation](docs/README.md) · [Quality limits](#quality-and-limitations)

> **Source preview, not a new PyPI release.** This README describes the host-only
> architecture in this checkout. Published beta 0.4.0b1 and stable 0.3.0 still
> contain the older architecture. Use the source installation below to try this
> version. It is not yet proven superior to your agent's built-in search.

## Research for the agent, reports for the person

A search call returns leads. Sibyl keeps the investigation connected:

- Break a question into research questions and retain the plan across calls.
- Search and read original pages into one evidence store.
- Inspect exact passages with stable identities, hashes and provenance.
- Record supported, disputed or unresolved conclusions with evidence links.
- Investigate counterevidence and pursue specific gaps in subsequent rounds.
- Reuse the structured review ledger directly, or export Markdown, JSON or PDF.

The host chooses queries, reads sources and judges what they support. Sibyl
retrieves, preserves state, checks reference integrity and enforces budgets.
It does not decide truth for the host or run an autonomous researcher.

The default MCP surface has **eight tools**, compact structured briefs and
on-demand passage reads. A research result does not need to become a long report
before another agent can use it.

## Quick start

Requires Python 3.11+ and a host supporting local stdio MCP and
`structuredContent`. Install this source checkout in a fresh environment:

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/sibyl-mcp --list-tools
```

The last command should show `profile: agent` and eight tools. Add this to your
host's MCP configuration, replacing both paths with absolute local paths:

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "/absolute/path/to/sibyl/.venv/bin/sibyl-mcp",
      "env": {
        "SIBYL_OUTPUT_ROOT": "/absolute/path/to/private-sibyl-research"
      }
    }
  }
}
```

On Windows, use `.venv\\Scripts\\sibyl-mcp.exe`. Configuration syntax varies by host;
the example is for clients accepting `mcpServers` JSON. Text-only integrations
can use `"args": ["--profile", "keyless"]` for the twelve-tool compatibility surface
and full text responses. This is not a remote HTTP MCP endpoint.

No search key is required. Optional Tavily search has separate credentials and
provider charges; subscription access to your agent does not pay for Tavily.
See [installation and optional integrations](llms-install.md).

Try asking your agent:

> Use Sibyl to investigate whether SQLite WAL fits a multi-process application.
> Inspect the original documentation, investigate limitations and counterevidence,
> and return evidence-linked conclusions with unresolved questions. Write a report
> only if I ask for one.

## Research workflow

| Tool | Purpose |
|---|---|
| `start_research` | Create a bounded job with focused sub-questions |
| `continue_research` | Search or read a URL into the job |
| `get_research` | Read the brief, an exact passage, source catalog or full state |
| `assess_research` | Record the host's conclusion, support, challenges and gaps |
| `complete_research` | Optionally finalize a cited host-authored report |
| `save_report` | Export that report; Markdown is the default |
| `cancel_research` | Cancel work or explicitly delete a checkpoint |
| `gather_bundle` | Retrieve standalone evidence without creating a job |

Start with `question` and `sub_questions`. Link each retrieval to `question_id`;
use `purpose="counterevidence"` for an actual investigation of alternatives.
The brief intentionally omits raw text: read
`get_research(job_id, passage_id=...)` before assessing support.

Each assessment preserves stable supporting/challenging passage IDs, a conclusion
and optional `follow_up_queries`. Inspect `research_plan` for remaining gaps,
continue with a focused query and rationale, then reassess. Use
`get_research(view="sources", revision=..., offset=0, limit=10)` to page evidence
outside the selected citation window.

Reports are optional. To deliver one, submit `complete_research` with the current
revision and `[Source N]` citations, then `save_report(job_id=..., format="md")`.
Citation validation checks targets, **not whether the text entails the claim**.
Incomplete research must retain its gaps; `allow_incomplete=true` yields a partial
report. A completed job means the report is finalized, not that its claims are true.

[Detailed workflow and limits](docs/USAGE.md#persistent-research-jobs) ·
[Machine-facing contract](docs/agent-protocol.md)

### See the investigation

[![Offline two-round research replay](assets/research-replay.svg)](docs/examples/sqlite-wal/README.md)

[Read the report](docs/examples/sqlite-wal/report.md) ·
[Inspect the evidence](docs/examples/sqlite-wal/sources.json) ·
[Reproduce the replay](docs/examples/sqlite-wal/README.md)

This is a recorded-source MCP replay with host-authored reasoning, not a live
accuracy benchmark. It checks gap reviews, counterevidence, exports and restart.
The [ten-case live host suite](docs/examples/host-suite/README.md) separately
documents discovery failures, host-supplied URLs and explicit abstention.

## Optional capabilities

```bash
.venv/bin/python -m pip install -e '.[pdf]'      # text-PDF input; POSIX only
.venv/bin/python -m pip install -e '.[report]'   # PDF output, no model SDK
.venv/bin/python -m pip install -e '.[monitor]'  # authenticated loopback monitoring
```

Markdown and JSON export need no extra. `[extract]` supplies Trafilatura for
compatibility reads; `[rerank]` supplies optional local FlashRank ranking.
The default lexical ranker needs no model download. See
[usage](docs/USAGE.md) and [operations](docs/OPERATIONS.md).

For inspection outside an agent, use `sibyl gather "a focused question" --format json`
or the async Python `sibyl.gather_bundle` API. Neither generates an answer.

## Upgrading from the old architecture

LiteLLM, model-provider configuration, autonomous execution, generic model analysis,
SWOT/timeline generation and finance/chart tools have been removed.
The `report` extra now means PDF rendering only. Use `agent` (default) or
`keyless`; old `report`, `finance` and `full` profiles are rejected.

Install in a fresh virtual environment: upgrading a package does not uninstall
its old transitive dependencies. Remove old model settings from `SIBYL_CONFIG`.
Do not give Sibyl your host's login token. Existing checkpoints remain readable
and completed reports exportable; old autonomous jobs cannot resume execution.
Start a new host job to continue investigating. See [migration](docs/USAGE.md#migration).

## Quality and limitations

The practical difference is **persistent, evidence-directed research**, not a
claim to have a better model. Whether it produces better answers than built-in
search still needs a paired same-host, same-provider, same-budget evaluation.

The [2026-10-21 live retrieval run](evals/results/live-retrieval-keyless-budget-fix-2026-10-21.json)
tested 66 questions three times. It reached 83.3% answer coverage and 97.3%
ready-state answer precision, but failed answerable readiness (66.0% vs 75%) and
repeat stability (68.2% vs 90%). These use gold/alias evidence matching, not
semantic report accuracy. Seven of ten live host-suite cases needed host-supplied
URLs. Compact responses and model removal do not fix those discovery weaknesses.

Public search can throttle; original pages can block extraction. Source-kind,
recency and duplicate-cluster labels are heuristics. PDFs are bounded text reads,
not OCR or guaranteed table extraction. Retrieved text can contain prompt
injection: the host must treat it as untrusted evidence, never instructions.

State is private local SQLite storage under `SIBYL_OUTPUT_ROOT/.research`, with
explicit deletion and bounded quotas, not automatic expiry or encryption.
One trusted local host owns a runtime. Public-network checks, revision checks
and controlled exports reduce risk; they do not turn Sibyl into a sandbox.

[Release evidence](docs/LAUNCH_READINESS.md) · [Roadmap](docs/QUALITY_BUILD_PLAN.md) ·
[Security](SECURITY.md) · [Privacy](PRIVACY.md)

## Contributing

Use [CONTRIBUTING.md](CONTRIBUTING.md) for offline tests, real MCP replay,
clean-wheel checks and dependency audits. Paid evaluations require explicit
authorization. [Architecture](docs/MCP_ARCHITECTURE.md) describes module ownership.

[Changelog](CHANGELOG.md) · [MIT license](LICENSE)
