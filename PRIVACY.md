# Privacy

Sibyl is a local command-line tool. The project runs no Sibyl service, account
system, analytics endpoint or telemetry collector.

## Data Sibyl handles

- **The ledger** the agent writes: a question, claims, URLs, quotes and notes.
  Sibyl reads it and never sends it anywhere.
- **Cited pages.** `sibyl check` requests every URL in the ledger from its
  publisher, identifying itself as `Sibyl/0.5 (+https://github.com/chriswu727/sibyl)`.
  Each publisher sees that request like any other visit, including your IP
  address. Do not cite private or signed URLs you would not open in a browser.
- **Files it writes**, all at paths you or your agent choose: the check result
  (`NAME.check.json`) with page titles, hashes, quote positions and issues; page
  texts when `--save-pages` is given; the Markdown report; and the skill file
  when `sibyl skill install` is run. Nothing is deleted automatically.

Sibyl does not call a generative model, read model or subscription credentials,
or need any API key. Your agent, its host and its model provider handle the
research conversation and any tool output under their own terms.

Installing through PyPI or `pip` contacts those distribution services. Public
pages can contain personal information; review saved page texts and reports
before sharing them.
