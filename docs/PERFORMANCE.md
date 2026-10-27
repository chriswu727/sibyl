# Performance and measurement

This guide describes the current local MCP runtime. There is no current
end-to-end latency SLA or validated report-quality speedup for the 0.4.0
development workflow. Historical timings do not establish current-model results.

## Where time is spent

Host investigations alternate between retrieval and host reasoning. Sibyl measures
tool execution, not the time the host spends thinking or waiting for a user.
All planning, synthesis and semantic judgment happen in the host. Sibyl makes
no generative-model calls; host subscription/API usage is outside its runtime.

Search-provider latency, throttling, publisher access, parsing, and model output
all affect completion time. A fast response can be an insufficient-evidence result;
measure usability and correctness separately from speed.

## Current controls

| Layer | Bound / behavior |
|---|---|
| MCP admission | 8 concurrent calls; busy requests fail without an unbounded queue |
| Tool deadline | 60 seconds per call; 600 seconds accumulated active work per job |
| Persistent job workers | 2; active execution time accumulates across continuations |
| Bundle cache | 30-second TTL, 64 entries, single-flight work shared among consumers |
| Search batch | 8-second collection deadline; cancellation awaits all provider tasks |
| General-web fallback | 7-second budget shared across DuckDuckGo, Mojeek, and Yahoo; each attempt reserves time for remaining providers |
| Configured Tavily search | At most 2 seconds before keyless fallback uses the remaining general-web budget |
| Page fetch | DNS-pinned HTTP; 2 MiB wire and decompressed body caps |
| HTML extraction | Offloaded with `asyncio.to_thread`; cancellation cannot kill the thread |
| PDF extraction | 2 workers per event loop; 6-second worker deadline; first 40 pages |
| PDF memory | Linux address-space cap; macOS sampled RSS kill threshold; see Security |
| Persistence | Bounded SQLite jobs and serialized payloads; no remote database dependency |

The direct Python API does not inherit MCP admission or whole-operation deadlines.
Search deadlines include provider pacing and queue waits. These are cancellation
deadlines, not a guarantee that an external service responds; cancellation cleanup
or event-loop stalls can add time. A timed-out request can still have been processed
by its provider. Empty or blocked search results do not establish that no evidence
exists. No CAPTCHA bypass or automatic provider-key provisioning is performed.

Embedding applications own those limits. Optional PDF export libraries can
block the event loop; monitoring shares that loop and cannot diagnose a stalled
process from inside it. Use an external process/scrape timeout as well.

## Measure a change

1. Record commit, Python/platform, installation extras, provider/model, parameters,
   query set, date, concurrency, and whether caches are warm or cold.
2. Run deterministic ranking/pipeline regressions before live measurements.
3. For retrieval, use `scripts/eval_live_retrieval.py` and retain the dated JSON.
   Report p50/p95, failures, actual provider paths, answer evidence, and repeat
   stability. A single successful source read is an integration check, not a benchmark.
4. For reports, score citation support, gaps, contradictions, and abstention along
   with elapsed time and provider usage. Hold model/source/budget choices constant
   in comparisons. Obtain explicit approval before consuming model quota.
5. For resource work, record peak memory, outstanding processes/tasks, open
   sockets, and disk usage across repeated runs and cancellation.

The [operations endpoints](OPERATIONS.md) expose invocation latency histograms,
outcomes, capacity, cache counts, and store state. Counters reset at restart;
rejected calls contribute zero-duration observations. A completed invocation is
not a quality verdict. Host thinking is not a Sibyl
MCP latency observation.

## Bounded runtime recovery check

Run `python scripts/check_runtime_soak.py --rounds 300 --monitor --pdf` with the
monitor/PDF extras installed. The harness uses temporary storage and no public
network or model calls. It exercises two concurrent workers with injected read
errors, timeouts, and cancellation, rejecting a third worker without consuming
its budget. It also checks artifact deletion, real authenticated monitoring,
malformed-PDF worker cleanup, and stopped-store backup/restore.

The [2026-10-21 result](../evals/results/runtime-soak-2026-10-21.json) passed 300
measured rounds plus three warm-up rounds in 28.947 seconds on macOS/Python 3.12.
The fixture retained three files totaling 16 KiB, with nine open descriptors
throughout the measured rounds and seven after shutdown. No new tasks or child
processes remained. Retained traced Python allocations grew by 123,360 bytes;
the sample list itself is retained. This is not process RSS or a guarantee of
zero leaks over hours of real-provider traffic. Temporary files are removed on exit.

CI runs 100 rounds with monitoring/PDF on Python 3.12; the core unit suite runs
a short six-round profile. Use `--output new-result.json` to retain raw samples;
the output path must not already exist.

## Tuning without changing the claim

Prefer focused questions and explicit follow-ups when broad retrieval misses an
anchor. Raising concurrency can increase provider throttling. Raising source or
output budgets increases cost and may not increase independent support. Preserve
content clusters, citations, and limitation markers when changing extraction.

There are no internal model, fast-mode or effort/depth settings. Adjust the host
reasoning budget separately; never describe reduced tool response bytes as reduced
host tokens or better answers without measuring them. See the [usage reference](USAGE.md).

## Historical evidence

The July 2026 three-repeat keyless retrieval run recorded an 11.0-second p95 but
failed readiness/precision/stability gates. Its scope and raw artifacts are linked
from [Launch readiness](LAUNCH_READINESS.md#historical-live-evidence).

Earlier v0.2.2 engineering notes reported model-specific latency reductions and
qualitative A/B observations. The [original notes](https://github.com/chriswu727/sibyl/blob/d343267/docs/PERFORMANCE.md)
remain in Git history. They mix successive configurations and are not a current
benchmark, proof that reasoning modes do not help, or a recommendation of one
model over another. The [roadmap](QUALITY_BUILD_PLAN.md) defines the next measured
research-quality and reliability work.
