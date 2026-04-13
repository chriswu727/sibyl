# Sibyl

**AI-powered deep research agent.** Ask any question — Sibyl searches the web, reads multiple sources, and generates a comprehensive research report with analysis and predictions.

Works as an **MCP server** (for Claude Code, Cursor, etc.) or as a **standalone CLI**. Bring your own LLM API keys — supports DeepSeek, OpenAI, Gemini, GLM, Anthropic, Ollama, and 100+ providers via LiteLLM.

## What It Does

```
You ask: "What's the outlook for Canadian housing market in 2026?"

Sibyl:
  1. Generates diverse search queries (market data, expert opinions, policy changes...)
  2. Searches the web, finds 10+ relevant sources
  3. Reads and extracts content from each source
  4. Synthesizes a structured report with:
     - Summary
     - Key findings (with source citations)
     - Analysis (conflicting viewpoints, trends)
     - Predictions (with confidence level)
```

## Quick Start

### MCP Server (for Claude Code)

```bash
pip install sibyl-research
claude mcp add sibyl -- sibyl-mcp

# Configure your LLM (pick one)
export DEEPSEEK_API_KEY=sk-...    # cheapest
export OPENAI_API_KEY=sk-...      # or OpenAI
export GEMINI_API_KEY=...         # or Gemini (free tier)
```

Then in Claude Code:

> "Research the impact of AI on software engineering jobs over the next 5 years"

### CLI

```bash
pip install sibyl-research

# Quick research
sibyl "Canadian immigration policy changes 2026"

# Deep research with predictions
sibyl "Bitcoin price prediction next 6 months" -d 3

# Specify model
sibyl "Impact of tariffs on US-China trade" --model deepseek/deepseek-chat
```

## Multi-Provider Support

Configure multiple LLM providers for different tasks:

```yaml
# sibyl.yaml
providers:
  - model: deepseek/deepseek-chat
    api_key: sk-xxx
    role: analysis    # Deep reasoning

  - model: gemini/gemini-2.5-flash
    api_key: xxx
    role: fast        # Quick tasks, free tier

  - model: openai/glm-4-flash
    api_key: xxx
    api_base: https://open.bigmodel.cn/api/paas/v4
    role: chinese     # Chinese content
```

```bash
sibyl "..." --config sibyl.yaml
```

## MCP Tools

| Tool | What it does |
|------|-------------|
| `research(query, depth)` | Full research cycle: search + scrape + analyze + report |
| `quick_search(query)` | Fast web search, returns raw results |
| `read_url(url)` | Extract clean text from any URL |
| `analyze(text, question)` | Analyze provided text with LLM |

## Research Depth

| Depth | Queries | Sources | Output |
|-------|---------|---------|--------|
| 1 (quick) | 2 | ~5 | Summary + findings |
| 2 (standard) | 4 | ~10 | + analysis |
| 3 (deep) | 6 | ~10 | + predictions + confidence |

## Requirements

- Python 3.10+
- At least one LLM API key (DeepSeek, OpenAI, Gemini, GLM, etc.)
- No other API keys needed (search uses DuckDuckGo, free)

## License

MIT
