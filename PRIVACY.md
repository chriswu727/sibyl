# Privacy

Sibyl is a local command-line and stdio MCP server. The project does not run a
Sibyl cloud service, user account system, analytics endpoint, or telemetry
collector.

This policy describes unreleased `main`. See the [documentation index](docs/README.md)
for current installation and [Operations](docs/OPERATIONS.md) for local retention controls.

## Data Sibyl handles

Sibyl may process research queries, search results, public URLs, downloaded page
content, extracted passages, publication metadata, and citation identifiers. The
host can submit assessments and reports, which Sibyl stores without model calls.

`gather_evidence` keeps its question, atomic follow-up queries, and retrieved
public evidence in the local MCP process for up to ten minutes so the host can
complete a bounded workflow. At most 64 loop records are retained; they are not
written to disk and disappear when the process exits.

Persistent research is separate: `start_research` stores questions, plans,
retrieval attempts, selected public evidence, progress events, counters, and
reports in `<output root>/.research/jobs.sqlite3`. The output root is
`SIBYL_OUTPUT_ROOT`, or the per-user data directory when unset. Every local
session of the same user using that root can list and read these records; they
are not separated by project. These records survive
restart and have no automatic time-based expiry. At most 64 jobs are retained,
with a 1 MiB payload limit per job plus 64 KiB reserved for recovery metadata;
new jobs are rejected when history is full.
`cancel_research(job_id=..., delete=true)` removes a checkpoint but does not
delete exported reports, backups, host conversation history, or provider records.
SQLite deletion is not a guarantee of forensic erasure. The database uses
owner-only file permissions, not encryption at rest.

MCP reports are written below the operator-selected
output root; calling agents can select only relative
subdirectories for report export. CLI/Python callers may explicitly select their
own output directory. Sibyl does not upload artifacts to the project maintainer.

MCP retrieval results may also remain in a per-runtime memory cache for 30 seconds
(at most 64 entries). Cancelling the final consumer stops its in-flight retrieval;
server shutdown clears transient runtime state but retains explicit research
checkpoints. Diagnostic runtime logs contain
tool names and aggregate completion/failure/cancellation counts, not tool arguments,
page bodies, or provider exception details. They are local logs, not telemetry.

Optional local monitoring exposes only operational counters and the last 64
content-free tool events through authenticated loopback endpoints. It does not
send telemetry anywhere. An operator-configured Prometheus scraper can persist
those metrics under its own retention policy. Logs at `--log-level INFO` include
tool name, outcome, duration, and active-call count; they go to stderr, not MCP stdout.
PDF input bytes are passed in memory to a resource-limited child without inherited
environment credentials. No raw PDF files or OCR models are saved automatically.

## Where data can go

- Keyless retrieval sends search terms to DuckDuckGo, Mojeek, or Yahoo Search;
  Google News, Reddit, Wikipedia, Semantic Scholar, and—for academic or DOI
  questions—Crossref; then requests pages from the selected publishers.
- When `SIBYL_SEARCH_PROVIDER=tavily` is explicitly configured, general-web
  search terms are sent to Tavily instead. Tavily receives the request metadata
  associated with that traffic and applies its own privacy and retention terms.
  Failed or empty Tavily requests fall back to the keyless search chain.
- Jina rendering is disabled by default. When `render_thin_pages`, `js_render`,
  or the `--render-thin-pages` CLI flag is explicitly enabled, the destination URL is
  sent to Jina Reader for extraction.
- Sibyl does not send prompts or evidence to a generative-model provider and does
  not read the host subscription login. Stored public evidence and user queries
  can themselves contain sensitive text; review before retaining.
- The MCP host may send Sibyl tool inputs and outputs to its own configured model
  provider. The host and provider privacy terms apply independently.
- Installing through PyPI, `pip`, or `uvx` contacts those distribution services.

Public web pages can contain personal or sensitive information. Review evidence
and generated artifacts before retaining or sharing them. Do not use Sibyl to
send private target URLs to third-party search, rendering, or model services.

## Credentials

Sibyl reads optional search/rendering credentials from environment variables,
not model or subscription credentials. Retrieval YAML accepts no credentials. It does not intentionally include credentials in reports or
send them to the project maintainer. Keep configuration files outside source
control and follow the selected provider's key-rotation guidance.

`CROSSREF_MAILTO` is optional and, when set, is sent to Crossref with academic
metadata requests. Use an address you are comfortable disclosing to that
service.
