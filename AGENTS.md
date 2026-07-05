# Sibyl — Agent Guide

This file helps AI agents (Claude Code, Cursor, etc.) use Sibyl effectively via MCP.

## Setup

```bash
pip install sibyl-research
# Retrieval-provider mode (recommended) needs NO key — you do the reasoning:
claude mcp add sibyl -- sibyl-mcp
# For the one-shot research() tool, add a provider key so sibyl's own model runs it:
claude mcp add sibyl -e DEEPSEEK_API_KEY=sk-... -- sibyl-mcp
```

## Two modes — who does the reasoning?

**Recommended: YOU (the host model) research; sibyl only retrieves.** This is
keyless and gives the best quality, because YOUR reasoning is applied to real
evidence — a mid-tier model doing the synthesis fabricates on hard questions,
whereas you can cross-reference sources and abstain when they don't answer.

```
# research a question yourself:
gather_sources("Serbian quarterfinalist 2018 Madrid Open men's singles")
gather_sources("2018 Madrid Open men's singles draw results")   # call again with sub-queries
→ read the returned [Source N] blocks, cross-reference, answer WITH citations.
→ if the sources don't contain it, gather more or say you don't know — never guess.
```

**One-shot: sibyl does everything with its own model** (needs a provider key):
```
research(query, depth=2)   # search→scrape→rank→synthesize→verify→report, by sibyl's LLM
```
Prefer `gather_sources` + your own synthesis for factual/hard questions; use
`research()` when you want a finished report in one call.

## Recommended Workflows

### Deep research on a topic
```
research(query, depth=3, language="auto")
→ save_report("both")
```

### Research with market data
```
research("NVIDIA outlook 2026", depth=2)
→ fetch_market_data("NVDA,AMD,INTC")
→ chart("NVDA,AMD,INTC")
→ save_report("pdf")
```

### Quick comparison
```
compare("React,Vue,Angular", query="for a startup in 2026")
```

### Strategic analysis
```
swot("Tesla")
```

### Trend tracking
```
trends("ChatGPT,Claude,Gemini", timeframe="today 12-m")
```

### Event timeline
```
timeline("OpenAI company history")
```

## Tool Selection Guide

| Goal | Use |
|------|-----|
| Full research report | `research(query, depth=1/2/3)` |
| Side-by-side comparison | `compare(items, query)` |
| SWOT analysis | `swot(subject)` |
| Google Trends data | `trends(keywords)` |
| Event timeline | `timeline(topic)` |
| Stock/ETF data | `fetch_market_data(symbols)` |
| Price chart | `chart(symbols, period)` |
| **Research a question yourself (keyless)** | **`gather_sources(query)` — retrieves full-text sources; you synthesize** |
| Quick web search | `quick_search(query)` |
| Read a specific page | `read_url(url)` |
| Analyze text | `analyze(text, question)` |
| Save as PDF/Markdown | `save_report(format, output_dir)` |

## Research Depth

| Depth | What happens |
|-------|-------------|
| 1 | 2-3 queries, basic synthesis. Fast (~30s). |
| 2 | Sub-question decomposition, per-question analysis, cross-referencing, review. (~90s) |
| 3 | + Knowledge gap filling, predictions with bull/bear/base case, confidence. (~120s) |

## Tips

- Use `depth=3` for any topic involving predictions, forecasts, or market analysis
- Use `depth=1` for factual questions or quick lookups
- Call `fetch_market_data()` and `chart()` after `research()` — they auto-attach to the report for PDF output
- Use `language="zh"` for Chinese output
- `compare()` and `swot()` automatically do a quick research before generating analysis
- `save_report("both")` generates PDF + Markdown; charts are embedded in PDF
- Multiple symbols in `chart()` generates both a line chart and a comparison bar chart
