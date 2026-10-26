# Install Sibyl MCP

Sibyl is a local stdio deep-research MCP tool for AI agents. Its default keyless profile
supports persistent host-assisted research, inspectable evidence, and cited
Markdown/JSON reports. The host agent plans and writes; no second model key is
required. Reports are optional outputs; the agent can consume the question-review
ledger directly. Configured-model autonomous research is experimental. PDF export is optional.

## Agent-first source installation (unreleased)

The `main` branch adds `--profile agent`, with eight focused tools, compact
structured briefs and on-demand evidence reads. It is not included in 0.4.0b1.
Install a source checkout with `python -m pip install -e .`, then set
`"args": ["--profile", "agent"]` in the configuration below. This mode requires
a host that reads MCP `structuredContent`. Keep `keyless` for text-only clients.
See [Agent protocol](docs/agent-protocol.md) for the contract and research workflow.

## Requirements

- Python 3.11 or newer
- A local virtual environment

This guide targets the published 0.4.0b1 beta. Pin the exact version to opt in;
PyPI 0.3.0 predates the current evidence API and profile flags.

```bash
python -m venv .venv
.venv/bin/python -m pip install 'sibyl-research==0.4.0b1'
```

## Keyless MCP configuration

Replace `/absolute/path/to/sibyl` with the directory containing your `.venv`.

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

Verify the installed package:

```bash
.venv/bin/sibyl-mcp --version
.venv/bin/sibyl-mcp --profile keyless --list-tools
```

The default output should identify the `keyless` profile and list
`gather_evidence`, `gather_bundle`, `gather_sources`, `quick_search`, and
`read_url`, `start_research`, `get_research`, `continue_research`,
`assess_research`, `complete_research`, `cancel_research`, and `save_report`.

## Optional PDF input and monitoring

For text-PDF input and local monitoring, install `sibyl-research[pdf,monitor]==0.4.0b1`. PDF ingestion
needs a POSIX host and does not perform OCR. Monitoring is disabled by default;
see [Operations](docs/OPERATIONS.md) before enabling `--monitor-port` with a
host-supplied `SIBYL_MONITOR_TOKEN`. It is not a remote MCP endpoint.

## Optional configured search

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

The `${VARIABLE}` strings in these JSON examples are placeholders. Environment
interpolation depends on the MCP host and is not performed by Sibyl. Configure
the host's documented secret mechanism; do not pass these strings literally.

## Optional experimental report profile

Install the report extra with
`.venv/bin/python -m pip install 'sibyl-research[report]==0.4.0b1'` and provide a model credential:

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

Use [Usage](docs/USAGE.md) for all profiles and tools, [Operations](docs/OPERATIONS.md)
for monitoring, and [Launch readiness](docs/LAUNCH_READINESS.md) for release status.
