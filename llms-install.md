# Install Sibyl MCP

Sibyl is a local stdio MCP server for web research. Its default keyless profile
searches and returns inspectable evidence; it does not require a search or model
API key.

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
`gather_bundle`, `gather_sources`, `quick_search`, and `read_url`.

## Optional report profile

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
