# Sibyl

[English](README.md) | [Simplified Chinese](README.zh-CN.md)

**Check your AI agent's research claims against the pages they quote.**

Your agent researches with its own search and fetch tools and records each claim,
with the URLs and verbatim quotes it relies on, in a small JSON ledger. Sibyl
then reads every cited page itself and reports:

- quotes that do not occur in the page text (after normalizing case, quotation
  marks, dashes and whitespace; `...` skips words inside one passage);
- pages it could not read, which leave their quotes unverified;
- numbers in a claim that none of its supporting quotes contains;
- names in a claim that do not appear near its quotes;
- claims supported only by one site, and claims about the same subject that
  state different numbers;
- inferences that rest on a claim with any of the blocking problems above.

A passed check renders a Markdown report with the findings, disputed and
unresolved claims, the counterevidence search, gaps, and the hash of the text
read from each source.

Sibyl ships as an **agent skill** (a `SKILL.md` that teaches the workflow) and the
`sibyl` command it calls. It does not search, rank, or call any model.

> **Alpha, source only.** This is a rewrite of Sibyl's earlier research MCP
> server, which is retired; see [Why the rewrite](#why-the-rewrite). Published
> 0.4.0b1 and 0.3.0 on PyPI are that old server.

## Install

Python 3.11 or later:

```bash
git clone https://github.com/chriswu727/sibyl.git
python -m pip install -e './sibyl[pdf]'   # [pdf] reads text PDFs (POSIX only)
sibyl skill install                        # writes ~/.claude/skills/sibyl/SKILL.md
```

`sibyl skill install --dir <path>` writes the skill elsewhere for hosts that read
skills from another directory, and `sibyl skill show` prints it. The agent needs
a shell tool to run `sibyl`.

## Use

Ask your agent for researched, cited answers; the skill tells it when to use the
ledger. To try it by hand:

```bash
sibyl template > research/topic.json      # start from the example
# edit claims, URLs and quotes
sibyl check research/topic.json           # exit 0 passed, 1 blocking issues, 2 invalid ledger
sibyl report research/topic.json          # writes research/topic.md
```

`sibyl check` writes `topic.check.json` beside the ledger: every source's read
status and text SHA-256, every quote's position, and every issue.
`--save-pages DIR` keeps the text read from each page. `sibyl report` refuses a
ledger changed since its check, and a failed check unless `--allow-issues` is
given; then the report marks every failing claim.

The [ledger format](docs/ledger-format.md) describes every field and check.

## What a passed check means

Every quote occurred in the text Sibyl extracted from its page when it checked,
and the claims' numbers and names matched their quotes. It does **not** mean the
quote supports the claim, that the quote is in context, that the source is
right, or that the answer is complete. The conclusions remain the agent's.
Pages change: the recorded hash identifies what was read, not what the page
says now.

Sibyl reads only public HTTP(S) pages and identifies itself honestly; pages that
refuse automated clients, need a login or render only with JavaScript cannot be
read, and a PDF is read only up to 40 pages. No OCR is performed.

## Why the rewrite

Sibyl began as a research MCP server that planned searches, ranked sources and
guided the agent through a persistent investigation, when agents' own research
was weaker. In two runs of a preregistered paired evaluation on 20 tasks, agents
using that server were not reliably better than the same agent with only its
built-in search: the differences stayed within run-to-run noise, at about twice
the cost ([records](evals/README.md)). What the agent could not do for itself was
check its own quotes deterministically, so that is what Sibyl keeps.

## Security and privacy

Page text is untrusted data and may contain instructions aimed at the agent;
the skill tells the agent never to follow them. See [SECURITY.md](SECURITY.md)
and [PRIVACY.md](PRIVACY.md). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).

MIT license.
