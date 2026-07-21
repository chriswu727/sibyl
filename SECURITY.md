# Security policy

## Supported versions

Security fixes are made on the latest published Sibyl release. Upgrade before
reporting a problem so the report reflects the current code.

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
outbound requests to search services and public web pages and can write reports
or charts under a user-selected output directory.

Retrieved pages are untrusted data. They can contain prompt injection or
misleading instructions intended for an MCP host. Sibyl extracts text but does
not make that text trustworthy; hosts must treat source content as evidence, not
as executable instructions.

The direct fetch path allows only public HTTP(S) destinations on ports 80 and
443. It rejects credential-bearing URLs and non-global addresses, validates and
pins DNS results, revalidates redirects, and caps decompressed responses at 2
MiB. These controls reduce SSRF and resource-exhaustion risk but do not establish
that a public site is benign or accurate.

Third-party Jina rendering is opt-in for keyless retrieval. Optional LLM,
finance, and ranking integrations expand the dependency and network trust
boundary. Install only the extras you need and review their policies separately.
