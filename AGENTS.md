# Sibyl — Agent Guide

This file helps AI agents (Claude Code, Cursor, etc.) use Sibyl effectively via MCP.

## Setup

```bash
pip install sibyl-research
claude mcp add sibyl -e DEEPSEEK_API_KEY=sk-... -- sibyl-mcp
```

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
