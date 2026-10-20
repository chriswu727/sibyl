# Security policy

## Supported versions

Security fixes are developed on `main` and delivered in subsequent releases.
The runtime hardening documented here is not yet in the published 0.3.0 package;
use the source-checkout instructions in the README to evaluate it. Include the
package version or commit when reporting a problem.

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
outbound requests to search services and public web pages. MCP reports and charts
are confined to the operator-selected `SIBYL_OUTPUT_ROOT` (default: `reports/`).
`save_report(output_dir=...)` accepts only a relative subdirectory of that root.
The CLI and direct Python rendering API retain explicit operator-selected paths.

Retrieved pages are untrusted data. They can contain prompt injection or
misleading instructions intended for an MCP host. Sibyl extracts text but does
not make that text trustworthy; hosts must treat source content as evidence, not
as executable instructions.

Evidence-loop identifiers use random UUIDs and expire after ten minutes. They
prevent accidental cross-talk between concurrent local workflows, but they are
not an authentication boundary; do not expose one Sibyl stdio process to
mutually untrusted users.

The direct fetch path allows only public HTTP(S) destinations on ports 80 and
443. It rejects credential-bearing URLs and non-global addresses, validates and
pins DNS results, revalidates redirects, and caps decompressed responses at 2
MiB. Compressed bytes are also limited to 2 MiB, and gzip/zlib-deflate expansion is
bounded before allocation. Unsupported or stacked encodings fail closed.
These controls reduce SSRF and resource-exhaustion risk but do not establish
that a public site is benign or accurate.

Third-party Jina rendering is opt-in for keyless retrieval. Optional LLM,
finance, and ranking integrations expand the dependency and network trust
boundary. Install only the extras you need and review their policies separately.

## Execution and storage controls

- Each MCP runtime admits at most eight simultaneous calls. Busy admission fails
  immediately; there is no unbounded task queue. Default deadlines are 60 seconds
  for retrieval/data/export and 600 seconds for model-assisted tools.
- Matching retrievals share one task. Cancelling one consumer does not interrupt
  another; cancelling the last consumer cancels the factory. Completed results
  have a 30-second TTL and a 64-entry cap. Shutdown cancels runtime-owned tasks.
- Stateful report operations are mutually exclusive. Reports, loops, and caches
  are not shared between independently created server runtimes.
- Persistent research admits two job workers, bounds retrieval/model attempts
  and reserved completion tokens, and requires explicit continuation after
  interruption. Checkpoints reserve attempts before dispatch. These are not exact
  provider billing limits; input tokens and already-dispatched work can be billed.
- Research state uses an owner-locked private SQLite database below the output
  root, with 64 records, a 1 MiB payload ceiling per job, 80 progress events per
  job, and a 16,384-page database limit. A stale revision cannot finalize a host
  report against different citation numbering. Finished reports are immutable.
- Host report validation checks citation targets, not semantic entailment.
  Automated report audits are fallible. Lexical overlap never establishes support;
  unassessed or unsupported blocks remain visibly marked.
- Tool text is bounded recursively: 10,000 characters per string, 64,000 combined,
  128 elements per collection, and eight nesting levels.
- Artifact names are unique, writes are staged then atomically published, and
  files have mode `0600`. Relative traversal and symlinked subdirectories are
  rejected. The root is resolved at startup/use from trusted operator settings.
  Do not place it in a directory writable by an untrusted local user: filesystem
  checks are not a defense against a hostile process racing directory changes.
- Provider exceptions are not returned verbatim at the MCP request boundary.
  Common credential query parameters and fragments are removed/redacted in
  displayed URL metadata; source text and structured evidence remain unchanged
  to preserve hashes and offsets. This is not general-purpose secret or PII
  detection. Never use sensitive queries or private signed URLs as research input.
- Docker uses an allowlisted build context and a non-root runtime user. Never
  mount credentials or host directories that the selected tools do not need.

Async deadlines are cooperative. Cancellation cannot forcibly stop an already
running native library or thread. Optional PDF/chart work can briefly block the
event loop, and third-party provider SDKs have their own resource behavior. Use
OS/container resource limits for stronger memory, CPU, disk, and network isolation.
The direct Python API does not inherit MCP admission limits; embedding applications
must bound their own concurrency and whole-operation deadlines.

## Dependency maintenance

CI audits clean core and report/finance/extract installations on Python 3.10,
3.11, and 3.12 against OSV, on pull requests and weekly. Audit failures block a
green workflow. Dependabot tracks
Python requirements and GitHub Actions. Release validation audits its environment.
Actions are commit-pinned and publication uses GitHub OIDC rather than a stored
PyPI token. Optional FlashRank is not part of the default audit matrix; operators
using `[rerank]` should audit that installed environment separately.

A clean audit means no known matching advisory was found at that time—not that
every dependency or execution path has been proven secure.

When upgrading an existing source-checkout environment, update transitive
dependencies too. Run from the checked-out repository:

```bash
python -m pip install --upgrade --upgrade-strategy eager -e .
python -m pip check
```

Include your selected extras in that install command. Fresh-install CI results
do not certify an old environment whose transitive dependencies remain pinned.
