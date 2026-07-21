# Install Sibyl MCP

Sibyl is a local stdio evidence-retrieval MCP server for AI agents. Its default
keyless profile searches, extracts, deduplicates, ranks, and returns inspectable
source bundles; the host agent plans and writes the answer. It does not require
a search or model API key.

## Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/) for isolated `uvx` execution

## Keyless MCP configuration

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "uvx",
      "args": ["--from", "sibyl-research", "sibyl-mcp"]
    }
  }
}
```

Equivalent commands:

```bash
claude mcp add sibyl -- uvx --from sibyl-research sibyl-mcp
codex mcp add sibyl -- uvx --from sibyl-research sibyl-mcp
```

Verify the published package:

```bash
uvx --from sibyl-research sibyl-mcp --version
uvx --from sibyl-research sibyl-mcp --list-tools
```

The default output should identify the `keyless` profile and list
`gather_evidence`, `gather_bundle`, `gather_sources`, `quick_search`, and
`read_url`.

## Optional production search

Tavily can replace the anonymous general-web path while the other public
sources remain available:

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "uvx",
      "args": ["--from", "sibyl-research", "sibyl-mcp"],
      "env": {
        "SIBYL_SEARCH_PROVIDER": "tavily",
        "TAVILY_API_KEY": "${TAVILY_API_KEY}",
        "CROSSREF_MAILTO": "${CROSSREF_MAILTO}"
      }
    }
  }
}
```

Tavily is explicit opt-in and may incur provider usage charges. A failed or
empty Tavily search falls back to the keyless chain. Crossref is queried for
academic and DOI-oriented questions without a key; `CROSSREF_MAILTO` is
optional.

## Optional experimental report profile

Install the report extra and provide a model credential:

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "uvx",
      "args": ["--from", "sibyl-research[report]", "sibyl-mcp", "--profile", "report"],
      "env": {
        "DEEPSEEK_API_KEY": "${DEEPSEEK_API_KEY}"
      }
    }
  }
}
```

Do not put a literal secret in a committed MCP configuration. Use the host's
secret or environment-variable mechanism.
