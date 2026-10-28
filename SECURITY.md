# Security policy

## Supported versions

Security fixes are developed on `main`. The current checkout is an unreleased
alpha; published 0.4.0b1 and 0.3.0 are the retired research server and receive
no fixes. Include the package version and checkout commit when reporting a
problem.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could put users or network
services at risk. Use the repository's
[private vulnerability report](https://github.com/chriswu727/sibyl/security/advisories/new),
or email `yichenwujob@gmail.com` if GitHub is unavailable. Include the Sibyl
version, operating system, Python version, reproduction steps and expected
impact, with private content removed. You should receive an acknowledgement
within seven days.

## Trust boundary

`sibyl` runs locally with the permissions of the user who starts it, usually
through an agent's shell tool. It reads a ledger file the agent wrote, fetches
the URLs that ledger cites, and writes the check result, optional page texts
and the report to paths the caller chose. It runs no code from pages or ledgers.

Retrieved pages are untrusted data and may contain prompt injection aimed at the
agent. Sibyl extracts text and compares strings; it does not make that text
trustworthy. The skill tells the agent never to follow instructions in page
text, but that is guidance to the agent, not an enforcement mechanism.

Because the agent chooses the URLs, the fetch path is the main exposure. It
allows only public HTTP(S) destinations on ports 80 and 443, rejects URLs with
credentials and non-global addresses, validates and pins DNS results,
revalidates each of at most five redirects, and caps compressed and
decompressed responses at 2 MiB, failing closed on unsupported or stacked
encodings. These controls reduce SSRF and resource-exhaustion risk; they do not
make a public site benign or accurate. Every request identifies itself as
`Sibyl/0.5 (+https://github.com/chriswu727/sibyl)`, and a refused page is not
retried.

## PDF input

Text-PDF reading is optional (`[pdf]`) and POSIX-only. Parsing runs under
`python -I` in a separate process with an empty environment, discarded stderr,
at most two workers per event loop, a 6-second wall deadline and CPU limits
(5-second soft, 6-second hard). Linux limits address space to 512 MiB; on macOS
a parent samples resident memory every 100 ms and kills above 512 MiB, which can
overshoot. Parsing reads at most 40 pages and 200,000 characters. Encrypted,
malformed and image-only PDFs fail explicitly; no OCR is performed. Process
separation is resource containment, **not an OS sandbox**: the worker keeps the
launching user's filesystem and network permissions.

## Dependency maintenance

CI audits clean core and `[pdf]` installations on Python 3.11, 3.12 and 3.13
against OSV on pull requests and weekly, and Dependabot tracks Python
requirements and GitHub Actions. Actions are commit-pinned and publication uses
GitHub OIDC rather than a stored PyPI token. A clean audit means no known
matching advisory was found at that time, not that every path is secure.
