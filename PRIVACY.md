# Privacy

Sibyl is a local command-line and stdio MCP server. The project does not run a
Sibyl cloud service, user account system, analytics endpoint, or telemetry
collector.

## Data Sibyl handles

Sibyl may process research queries, search results, public URLs, downloaded page
content, extracted passages, publication metadata, and citation identifiers. The
optional report mode also processes model prompts and generated reports. Finance
tools process the ticker symbols and trend keywords supplied by the user.

Reports, Markdown files, PDFs, and charts are written only to paths selected by
the user or calling agent. Sibyl does not upload those artifacts to the project
maintainer.

## Where data can go

- Keyless retrieval sends search terms to DuckDuckGo or Mojeek, Google News,
  Reddit, Wikipedia, and Semantic Scholar, then requests pages from the selected
  publishers.
- Jina rendering is disabled by default. When `render_thin_pages`, `js_render`,
  or the `--js-render` CLI flag is explicitly enabled, the destination URL is
  sent to Jina Reader for extraction.
- Optional report and analysis tools send prompts and selected source text to
  the LLM provider configured by the user. A local or API-compatible backend can
  be used instead.
- Optional finance tools contact Yahoo Finance and Google Trends.
- The MCP host may send Sibyl tool inputs and outputs to its own configured model
  provider. The host and provider privacy terms apply independently.
- Installing through PyPI, `pip`, or `uvx` contacts those distribution services.

Public web pages can contain personal or sensitive information. Review evidence
and generated artifacts before retaining or sharing them. Do not use Sibyl to
send private target URLs to third-party search, rendering, or model services.

## Credentials

Sibyl reads provider credentials from environment variables or a user-selected
configuration file. It does not intentionally include credentials in reports or
send them to the project maintainer. Keep configuration files outside source
control and follow the selected provider's key-rotation guidance.
