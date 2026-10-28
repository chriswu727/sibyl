# Ten live research investigations

Run date: 2026-10-21. These cases used the production keyless MCP server over
stdio, with the existing Codex conversation planning, inspecting, and writing.
No separate model process, autonomous model API, or paid search was invoked.

The [questions and rubric](../../../evals/host_research_cases.json) were fixed
before collection. The [retained result](../../../evals/results/host-research-2026-10-21.json)
contains findings, source URLs and hashes, passage IDs, reviews, attempts,
tool sequences, limitations, and export checks. Scores are host self-assessments,
not independent grading, a blind benchmark, or a product accuracy estimate.

## Observed delivery

All ten investigations finalized and exported both Markdown and JSON. The
exported findings and report status were checked against the completed MCP job.
Nine reports have `report.status="ok"`; the future-share question deliberately
retains `partial`. This status checks workflow requirements and citation targets,
not independently verified semantic correctness.

| Case | Retrieval rounds | New source URL supplied by host | Report status |
|---|---:|---|---|
| API idempotency | 4 | Stripe documentation | ok |
| SQLite WAL | 5 | Both official SQLite documents | ok |
| PostgreSQL isolation | 2 | No | ok |
| FlashAttention | 2 | No | ok |
| RAG limitations | 6 | Both paper abstracts | ok |
| GPTQ and AWQ | 4 | Original GPTQ abstract | ok |
| 2024 semiconductor sales | 4 | SIA year-end release | ok |
| Prompt-injection boundary | 3 | OWASP control guidance | ok |
| SSRF and DNS | 3 | No; direct read expanded a search-discovered source | ok |
| Exact realized 2030 market share | 3 | Historical NVIDIA financial release | partial |

Only three cases completed using search-discovered sources alone. Seven needed
host-supplied URLs. This is useful evidence for the research workflow and direct
reader, but **not a successful general source-discovery gate**. The separate
[66-case retrieval run](../../LAUNCH_READINESS.md#discovery-follow-up-2026-10-21)
still fails repeat stability and answer readiness.

## What changed during investigation

The initial two queries were not treated as a sufficient research plan forever.
After inspecting evidence, the host recorded unresolved reviews and proposed
focused follow-ups. Failed searches stayed in the record. Subsequent direct
reads were labeled as host-supplied sources rather than automatic discoveries.

- **A mismatched paper was rejected.** A GPTQ query returned a paper about
  integer-only vision-transformer kernels. It was not cited as GPTQ evidence;
  the original GPTQ abstract was obtained separately.
- **A dated revision changed the market answer.** The SIA page retains its
  original $627.6 billion announcement but adds a March 3, 2025 correction to
  $630.5 billion. Earlier WSTS forecasts were kept separate from observed results.
  See the [focused report examples](reports.md).
- **Source scope stayed visible.** RAG conclusions use abstracts, not a claimed
  full-paper review. Copies of the original FlashAttention paper are not counted
  as independent works, and its sparse extension is not conflated with dense
  exact attention.
- **A future outcome remained unknown.** Historical NVIDIA revenue was not
  converted into an invented realized 2030 share. The finalized report remains
  explicitly partial.

## Rubric and limits

Each dimension is scored 0–2: coverage, support, source quality, counterevidence,
and delivery. The per-case record explains the source and investigation limits.
There is no aggregate accuracy score: human-in-the-loop source selection and
host-provided URLs materially shape these results. The suite does not establish
autonomous-model quality, search reliability, or superiority over a host's normal
web search.

To repeat initial collection from a source checkout:

```bash
.venv/bin/python scripts/collect_host_research.py --output /tmp/sibyl-host-suite-new
```

The output directory must be new. Collection saves full evidence and per-call
traces locally, including tool errors; it does not author, assess, or complete
reports automatically. Inspect the evidence, then use the documented MCP
review/continuation/completion/export workflow. The public result intentionally
omits full scraped passages and local filesystem paths.
