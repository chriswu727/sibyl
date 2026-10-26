# Install Sibyl MCP

Sibyl adds persistent evidence research to an existing MCP-capable AI agent.
The host plans, reasons and writes under its own account and usage rules.
Sibyl never reads subscription tokens or calls a generative model.

## Source preview

The host-only architecture and default eight-tool agent surface are unreleased.
PyPI 0.4.0b1 and stable 0.3.0 contain the previous architecture. Use a fresh source
environment; upgrading an old environment does not remove old dependencies.

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/sibyl-mcp --version
.venv/bin/sibyl-mcp --list-tools
```

Requires Python 3.11+. The version string alone does not identify this unreleased
change; record the checkout commit when reporting problems.

## Host configuration

For hosts accepting `mcpServers` JSON, replace the paths:

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "/absolute/path/to/sibyl/.venv/bin/sibyl-mcp",
      "env": {
        "SIBYL_OUTPUT_ROOT": "/absolute/path/to/private-sibyl-research"
      }
    }
  }
}
```

Windows uses `.venv\\Scripts\\sibyl-mcp.exe`. Your host must launch local stdio
servers and consume MCP `structuredContent`. For text-only hosts add
`"args": ["--profile", "keyless"]`; it returns full text and twelve compatibility
tools. The default `agent` surface has eight:
`start_research`, `continue_research`, `get_research`, `assess_research`,
`complete_research`, `save_report`, `cancel_research`, `gather_bundle`.

Do not configure a model provider or pass subscription credentials to Sibyl.
The host sends tool inputs/outputs through its own model connection; its privacy
policy and usage limits still apply. No hidden MCP sampling is performed.

## Optional integrations

Install from the same checkout:

```bash
.venv/bin/python -m pip install -e '.[pdf,report,monitor]'
```

`pdf` reads text PDFs on POSIX, without OCR. `report` only renders PDF output;
Markdown/JSON need neither extra. `monitor` enables authenticated loopback-only
operational endpoints when explicitly configured; see [Operations](docs/OPERATIONS.md).
It is not an HTTP MCP transport.

Anonymous search needs no key. To opt into Tavily, set
`SIBYL_SEARCH_PROVIDER=tavily` and supply `TAVILY_API_KEY` through your host's secret
or environment mechanism. This is separately billed by Tavily. One research step
can make multiple provider requests. Failed or empty searches fall back to the
keyless chain. `CROSSREF_MAILTO` is optional and disclosed to Crossref for academic
metadata queries. Jina rendering is separately opt-in and sends target URLs to Jina.

Never commit real keys. Environment placeholder expansion depends on your host,
not Sibyl. The old model-backed `report/full` profiles are no longer available.
See the [migration inventory](docs/MIGRATION.md) before reusing an old installation
or checkpoint directory.

Read the [Agent protocol](docs/agent-protocol.md), [Usage](docs/USAGE.md),
[Privacy](PRIVACY.md) and [release evidence](docs/LAUNCH_READINESS.md).
