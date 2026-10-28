# Security policy

## Supported versions

Security fixes are developed on `main` and delivered in subsequent releases.
This policy follows unreleased host-only `main`. Published 0.4.0b1 and 0.3.0
contain older architectures and must not be assumed to have every control below.
Use the source-checkout instructions in the README to evaluate this implementation.
Include the package version and checkout commit when reporting a problem.

The [documentation index](docs/README.md) links installation, operations, and
release status. A development version string is not proof that a fix is on PyPI.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could put users, credentials,
or network services at risk. Use the repository's
[private vulnerability report](https://github.com/chriswu727/sibyl/security/advisories/new),
or email `yichenwujob@gmail.com` if GitHub is unavailable. Include:

- the affected Sibyl version, operating system, and Python version;
- the MCP profile and client in use;
- reproduction steps and expected impact;
- logs or a minimal fixture with credentials and private content removed.

You should receive an acknowledgement within seven days. Please allow time for
validation and a coordinated fix before publishing details.

## Trust boundary

Sibyl runs locally with the permissions of the user who starts it. It makes
outbound requests to search services and public web pages. MCP reports
are confined to the output root: `SIBYL_OUTPUT_ROOT`, or the per-user data
directory when unset.
`save_report(output_dir=...)` accepts only a relative subdirectory of that root.
The CLI and direct Python rendering API retain explicit operator-selected paths.

Retrieved pages are untrusted data. They can contain prompt injection or
misleading instructions intended for an MCP host. Sibyl extracts text but does
not make that text trustworthy; hosts must treat source content as evidence, not
as executable instructions.

Job identifiers are random, but they are not an authentication boundary; do not
expose one Sibyl stdio process to mutually untrusted users.

The direct fetch path allows only public HTTP(S) destinations on ports 80 and
443. It rejects credential-bearing URLs and non-global addresses, validates and
pins DNS results, revalidates redirects, and caps decompressed responses at 2
MiB. Compressed bytes are also limited to 2 MiB, and gzip/zlib-deflate expansion is
bounded before allocation. Unsupported or stacked encodings fail closed.
These controls reduce SSRF and resource-exhaustion risk but do not establish
that a public site is benign or accurate.

Third-party Jina rendering is opt-in for keyless retrieval. Optional rendering, parsing,
and local ranking integrations expand the dependency and network trust
boundary. Install only the extras you need and review their policies separately.

## Execution and storage controls

- Each MCP runtime admits at most eight simultaneous calls. Busy admission fails
  immediately; there is no unbounded task queue. Default deadlines are 60 seconds
  per MCP operation; research jobs also have a 600-second cumulative active budget.
- Matching retrievals share one task. Cancelling one consumer does not interrupt
  another; cancelling the last consumer cancels the factory. Completed results
  have a 30-second TTL and a 64-entry cap. Shutdown cancels runtime-owned tasks.
- Stateful report operations are mutually exclusive. Reports, loops, and caches
  are not shared between independently created server runtimes.
- Persistent research admits two running jobs per store across all sessions, bounds retrieval attempts and active execution time, and requires explicit continuation after
  interruption. Checkpoints reserve attempts before dispatch. These are not exact
  search-provider billing limits; already-dispatched requests can be billed.
- Research state uses a private SQLite database below the output root, shared by
  the same user's local sessions. Running jobs are owned by one runtime's
  instance lock; other sessions get read-only access until the owner finishes or
  exits. This is not isolation between projects. The store holds 64 records, a 1 MiB payload ceiling per job (plus 64 KiB reserved
  for failure recovery metadata), 80 progress events per
  job, and a 16,384-page database limit. A stale revision cannot finalize a host
  report against different citation numbering. Finished reports are immutable.
- Host report validation checks citation targets, not semantic entailment.
  No internal model performs semantic audits. Lexical overlap never establishes support.
- Tool text is bounded recursively: 10,000 characters per string, 64,000 combined,
  128 elements per collection, and eight nesting levels.
- Artifact names are unique, writes are staged then atomically published, and
  files have mode `0600` on POSIX (on Windows they inherit the output directory's
  ACLs). Relative traversal, rooted or drive-relative Windows paths, and
  symlinked subdirectories are rejected. The root is resolved at startup/use from
  trusted operator settings.
  Do not place it in a directory writable by an untrusted local user: filesystem
  checks are not a defense against a hostile process racing directory changes.
- Exports are capped at 16 MiB per file, 256 files and 128 MiB per selected output
  directory. A private publication lock serializes quota checks and renames.
  These are not a filesystem-wide disk quota: different subdirectories, temporary
  rendering files, checkpoints, and files written by other programs are separate.
  No automatic deletion of user reports occurs.
- Provider exceptions are not returned verbatim at the MCP request boundary.
  Common credential query parameters and fragments are removed/redacted in
  displayed URL metadata; source text and structured evidence remain unchanged
  to preserve hashes and offsets. This is not general-purpose secret or PII
  detection. Never use sensitive queries or private signed URLs as research input.
- Docker uses an allowlisted build context and a non-root runtime user. Never
  mount credentials or host directories that the selected tools do not need.

## PDF input and monitoring

Text-PDF input is optional (`[pdf]`) and uses the same DNS-pinned, size-bounded
fetch path as HTML. Parsing runs under `python -I` in a separate process with an
empty environment, discarded stderr, at most two workers per event loop, a
6-second wall deadline, and CPU limits (5-second soft / 6-second hard). Cancellation
kills and reaps the child. Linux limits address space to 512 MiB. macOS cannot
enforce that address-space limit; a parent samples resident memory every 100 ms
and kills above 512 MiB. Sampling can overshoot and is not a hard allocation cap.
Other POSIX hosts require working resource limits; non-POSIX hosts reject PDF reads.
Parsing reads at most 40 pages and emits at most 60,000 characters, often fewer
under the caller's extraction limit. Encrypted, malformed, and image-only PDFs
return explicit failures; mixed text/image documents disclose unreadable pages.

The worker does not execute PDF actions or provide OCR. Process separation is
resource containment, **not an OS security sandbox**: it still has the launching
user's filesystem/network permissions. Use an unprivileged account/container and
OS-level isolation for adversarial inputs. Keep the optional parser up to date.

Monitoring is off unless `--monitor-port` is supplied. It binds only to IPv4
loopback, requires a 32–256 printable-character bearer token on every route, and
rejects foreign Host headers, browser Origin headers, query strings, request
bodies, duplicate headers, and non-GET methods. There is no CORS or state mutation.
The HTTP parser is h11 with an 8 KiB incomplete-event limit, 16-connection/task
admission, and a two-second keepalive timeout. This is not comprehensive local
DoS protection. Do not forward or reverse-proxy this port onto a public interface.
Anyone holding the token can read operational state. The status and metrics
surfaces omit questions, source URLs, passage text, job IDs, paths, and credentials.
See [Operations](docs/OPERATIONS.md) for lifecycle and readiness semantics.

Async deadlines are cooperative. Cancellation cannot forcibly stop an already
running native library or thread. Optional PDF export work can briefly block the
event loop, and third-party provider SDKs have their own resource behavior. Use
OS/container resource limits for stronger memory, CPU, disk, and network isolation.
The direct Python API does not inherit MCP admission limits; embedding applications
must bound their own concurrency and whole-operation deadlines.

## Dependency maintenance

CI audits clean core and report/extract/pdf/monitor installations on Python 3.11,
3.12, and 3.13 against OSV, on pull requests and weekly. Audit failures block a
green workflow. Dependabot tracks
Python requirements and GitHub Actions. Release validation audits its environment.
Actions are commit-pinned and publication uses GitHub OIDC rather than a stored
PyPI token. Optional FlashRank is not part of the default audit matrix; operators
using `[rerank]` should audit that installed environment separately.

A clean audit means no known matching advisory was found at that time—not that
every dependency or execution path has been proven secure.

Python 3.11 is the minimum supported runtime. The report extra requires
Pillow 13.0.0 or newer to exclude known-vulnerable older versions; Pillow 13 no
longer supports Python 3.10. The core installation does not require Pillow.

When upgrading an existing source-checkout environment, update transitive
dependencies too. Run from the checked-out repository:

```bash
python -m pip install --upgrade --upgrade-strategy eager -e .
python -m pip check
```

Include your selected extras in that install command. Fresh-install CI results
do not certify an old environment whose transitive dependencies remain pinned.
