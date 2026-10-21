# Sibyl operations

Run Sibyl as one trusted local stdio MCP process. Monitoring is an optional,
read-only view of that same runtime, not a remote MCP API. This runbook covers
startup, health interpretation, alerts, recovery, and local retention.

This describes unreleased `main`; start with [Installation](../llms-install.md).
For unimplemented monitoring extensions, see the [roadmap](QUALITY_BUILD_PLAN.md).

## Start with monitoring

Install from the current source checkout:

```bash
python -m pip install -e '.[pdf,monitor]'
```

Generate a token using `python -c 'import secrets; print(secrets.token_urlsafe(32))'`
and store it in the MCP host's secret settings as `SIBYL_MONITOR_TOKEN`. Do not put
the token into Git, shell command arguments, URL query strings, or screenshots.
The token must contain 32–256 printable ASCII characters without spaces. Add
`--monitor-port 8766` to the host's Sibyl command:

```json
{
  "command": "/absolute/path/to/sibyl/.venv/bin/sibyl-mcp",
  "args": ["--profile", "keyless", "--monitor-port", "8766", "--log-level", "INFO"]
}
```

The host must pass the token through its environment/secret mechanism. Its
configuration syntax varies by host. For manual stdio operation, the equivalent
command is `sibyl-mcp --profile keyless --monitor-port 8766 --log-level INFO` with
the token already in the environment. Do not start a second runtime against the
same research directory just to monitor the first.

The listener binds only to `127.0.0.1`. Use that exact host, not `localhost`.
Port conflicts or invalid credentials fail startup. Port `0` selects an ephemeral
port; the `monitor_started` INFO event on stderr reports the bound port without
the token. Omitting `--monitor-port` opens no monitoring socket. Token rotation
requires restarting the process. Stdout remains exclusively MCP protocol data.

Every request requires `Authorization: Bearer <token>`. Foreign Host headers,
Origin headers, query strings, bodies, and non-GET methods are rejected. Do not
publish this endpoint through a proxy, tunnel, or public interface.

| Route | Success | Meaning |
|---|---|---|
| `/healthz` | 200 | The HTTP/event loop is responding and the runtime is open |
| `/readyz` | 200 | Open, below active-call capacity, and initialized store summary readable |
| `/status` | 200 | Content-free runtime snapshot; 503 if the initialized store is unhealthy |
| `/metrics` | 200 | Prometheus exposition, including a separate store-health gauge |

Readiness returns 503 while busy or when the initialized store cannot be read.
An uninitialized lazy store is not tested or created by monitoring; inspect
`jobs.initialized` / `sibyl_store_initialized`. Readiness does not test publisher
availability, model credentials, disk writeability, evidence completeness, or
semantic correctness. An unhealthy store does not make the liveness endpoint
fail. A blocked event loop can prevent all four endpoints from responding.

## Metrics and logs

`/status` includes active/capacity counts, cache entry/in-flight counts, aggregate
job states, partial-report count, database bytes, uptime, per-tool outcomes and
latency totals, and the last 64 tool events. It includes no questions, arguments,
source URLs, passages, job IDs, filesystem paths, or provider keys. Tool labels
are bounded; there are no query/job-ID labels. Metrics reset on process restart.

`sibyl_tool_calls_total{tool,outcome}` distinguishes completed, failed, timeout,
cancelled, and rejected invocations. `sibyl_tool_duration_seconds` is a histogram
including rejected calls at zero duration. A completed invocation can return
insufficient evidence or a partial report: it is not a successful-research score.
Schema validation before runtime admission is outside these invocation counters.
Autonomous work is visible in job-state counts; its internal model operations are
not separate MCP invocation metrics.

At INFO, runtime events are JSON lines on stderr. The host owns log rotation and
retention. Sibyl does not create an unbounded log file or contact a telemetry
collector. Other library warnings may also appear on stderr.

A locally managed Prometheus can scrape with a separate owner-readable credential
file containing only the token (example path below; create it through your secret
manager and restrict permissions):

```yaml
scrape_configs:
  - job_name: sibyl
    scrape_interval: 15s
    scrape_timeout: 3s
    static_configs:
      - targets: ['127.0.0.1:8766']
    authorization:
      type: Bearer
      credentials_file: /private/path/sibyl-monitor-token
```

Prometheus must run in the same network namespace. A Docker container's loopback
is not the host's loopback. This repository does not automatically install a
metrics database, dashboard, alert manager, or background daemon.

Suggested alert expressions (tune thresholds to workload):

- `up{job="sibyl"} == 0` for two minutes: process, authentication, or scrape failure.
- `sibyl_store_healthy == 0` for one minute: checkpoint store needs investigation.
- `increase(sibyl_tool_calls_total{outcome="timeout"}[10m]) > 3`: repeated deadlines.
- `increase(sibyl_tool_calls_total{outcome="rejected"}[10m]) > 10`: admission pressure.
- `sibyl_research_jobs{state="interrupted"} > 0`: inspect retained jobs before resuming.

These are examples, not enabled notifications. A missing metric is not zero;
check scrape health and store initialization before interpreting job counts.

## Recovery and retention

Shutdown closes the listener, cancels owned work, and releases the research lock.
After a crash, unfinished jobs become interrupted and unfinished retrieval attempts
become failed. Consumed attempts remain budgeted. Inspect `get_research(job_id)`
before explicit continuation. Autonomous continuation can incur provider charges;
it does not run automatically or provide exactly-once external execution.

If new evidence exceeds the 1 MiB checkpoint payload ceiling, the prior evidence
is retained, the discarded round becomes failed, and consumed budgets remain.
A 64 KiB recovery metadata reserve records this transition. The MCP operation
reports the capacity error. Start a narrower investigation rather than repeatedly
retrying the same oversized evidence. Real disk-full/corruption errors still
require operator repair; no recovery routine can guarantee a write to a failed disk.

The store retains at most 64 jobs and 80 progress events per job. Use
`cancel_research(job_id, delete=true)` to explicitly remove unwanted checkpoints.
This does not delete exported reports. Before filesystem backup or restore, stop
the MCP runtime and copy the entire `.research` directory as one consistent unit.
Keep backups private. Do not remove an ownership lock to bypass an active owner.

Each selected export directory accepts at most 256 `sibyl_*` artifacts totaling
128 MiB; each exported file is capped at 16 MiB. Explicitly archive/delete old
exports when full. Limits are per directory, not a global disk quota; temporary
rendering output, distinct subdirectories, databases, and backups are separate.
PDF input is in-memory and does not cache raw downloads or install OCR models.
Use filesystem/container quotas if an absolute disk or memory ceiling is required.

## Verify an installation

```bash
python -m unittest discover tests -v
ruff check sibyl tests scripts
ruff format --check sibyl tests scripts
mypy
python scripts/check_mcp_stdio.py "$(command -v sibyl-mcp)"
python scripts/check_mcp_stdio.py "$(command -v sibyl-mcp)" --monitor
python scripts/check_research_workflow.py
```

The smoke checks use real stdio and loopback HTTP, create/delete a temporary
checkpoint, and verify the listener closes. They do not make paid model calls or
establish research-report quality. CI additionally runs offline retrieval gates
and clean-install dependency audits. Review [Security](../SECURITY.md) for parser
resource limits, platform differences, and non-sandboxed boundaries.

Fault regression tests additionally terminate a real checkpoint-owning process,
inject writes failing before/after retrieval dispatch, abort SQLite updates, cycle
worker capacity/cancellation, and fail atomic artifact publication. They check
budget retention, revision consistency, lock release, and staging cleanup. These
bounded tests do not simulate physical disk damage or establish a sustained-load SLA.
