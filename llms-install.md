# Install Sibyl MCP

Sibyl is a local stdio deep-research MCP server. Its default keyless profile
supports persistent host-assisted research, inspectable evidence, and cited
Markdown/JSON reports. The host agent plans and writes; no second model key is
required. Configured-model autonomous research and PDF export are optional.

## Requirements

- Python 3.10 or newer
- Git and a local virtual environment

This guide targets unreleased `main`. PyPI 0.3.0 predates the current evidence
API and profile flags. Install the checkout before using this configuration:

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python -m venv .venv
.venv/bin/python -m pip install -e .
```

## Keyless MCP configuration

Replace `/absolute/path/to/sibyl` with the checkout's absolute path.

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "/absolute/path/to/sibyl/.venv/bin/sibyl-mcp",
      "args": ["--profile", "keyless"]
    }
  }
}
```

Verify the source installation:

```bash
.venv/bin/sibyl-mcp --version
.venv/bin/sibyl-mcp --profile keyless --list-tools
```

The default output should identify the `keyless` profile and list
`gather_evidence`, `gather_bundle`, `gather_sources`, `quick_search`, and
`read_url`, `start_research`, `get_research`, `continue_research`,
`complete_research`, `cancel_research`, and `save_report`.

## Optional production search

Tavily can replace the anonymous general-web path while the other public
sources remain available:

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "/absolute/path/to/sibyl/.venv/bin/sibyl-mcp",
      "args": ["--profile", "keyless"],
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

Install the report extra from the checkout with
`.venv/bin/python -m pip install -e '.[report]'` and provide a model credential:

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "/absolute/path/to/sibyl/.venv/bin/sibyl-mcp",
      "args": ["--profile", "report"],
      "env": {
        "DEEPSEEK_API_KEY": "${DEEPSEEK_API_KEY}"
      }
    }
  }
}
```

Do not put a literal secret in a committed MCP configuration. Use the host's
secret or environment-variable mechanism.
