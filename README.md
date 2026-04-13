# Sibyl

**AI-powered deep research agent.** Ask any question — Sibyl searches the web across multiple sources, reads dozens of pages, cross-references findings, and generates an executive-quality research report with analysis, predictions, and citations.

Not just another search summarizer. Sibyl is a **research analysis platform** — it does structured comparisons, SWOT analysis, Google Trends tracking, event timelines, and financial data visualization. All from a single question.

## What Makes Sibyl Different

| | Traditional Search | ChatGPT/Perplexity | GPT Researcher | **Sibyl** |
|---|---|---|---|---|
| Web search + summary | Yes | Yes | Yes | Yes |
| Multi-source (news, Reddit, Wikipedia) | No | Partial | Partial | **Yes (4 engines)** |
| Sub-question decomposition | No | No | Yes | **Yes** |
| Iterative gap-filling (search → analyze → identify gaps → search again) | No | No | Partial | **Yes** |
| Cross-source analysis (sentiment, consensus, disagreements) | No | No | No | **Yes** |
| Structured comparison tables | No | No | No | **Yes** |
| SWOT analysis | No | No | No | **Yes** |
| Google Trends data | No | No | No | **Yes** |
| Event timelines | No | No | No | **Yes** |
| Financial data + charts | No | No | No | **Yes** |
| MCP server (Claude Code, Cursor) | No | No | No | **Yes** |
| Multi-LLM (DeepSeek, Gemini, GLM, OpenAI) | No | No | Limited | **Yes (auto-detect)** |
| PDF reports with embedded charts | No | No | Basic | **Yes** |

## Quick Start

### MCP Server (for Claude Code / Cursor)

```bash
pip install sibyl-research
claude mcp add sibyl -e DEEPSEEK_API_KEY=sk-... -- sibyl-mcp
```

Then in Claude Code:

> "Research the impact of AI on software engineering jobs over the next 5 years"

> "Compare NVIDIA vs AMD vs Intel for AI workloads"

> "SWOT analysis of Tesla in 2026"

### CLI

```bash
pip install sibyl-research
export DEEPSEEK_API_KEY=sk-...   # or OPENAI_API_KEY, GEMINI_API_KEY, etc.

# Standard research
sibyl "Canadian housing market outlook 2026"

# Deep research with predictions + market data + PDF
sibyl "Will NVIDIA maintain AI chip dominance?" -d 3 --symbols NVDA,AMD,INTC --pdf

# Chinese output
sibyl "加拿大移民政策变化" -l zh --pdf -o reports/
```

## How It Works

```
You ask a question
  │
  ├─ Step 1: Decompose into 3-5 focused sub-questions
  ├─ Step 2: Generate 15-20 diverse search queries
  ├─ Step 3: Search across 4 engines (DuckDuckGo, Google News, Reddit, Wikipedia)
  ├─ Step 4: Scrape 15-20 sources (realistic browser headers, retry, Google Cache fallback)
  ├─ Step 5: Filter sources by relevance (LLM-scored)
  ├─ Step 6: Analyze each sub-question independently
  ├─ Step 7: Identify knowledge gaps → auto-search for missing info
  ├─ Step 8: Cross-reference sources (sentiment, consensus, disagreements)
  ├─ Step 9: Section-by-section synthesis (Summary, Findings, Analysis, Predictions)
  ├─ Step 10: Review and refine draft
  └─ Output: PDF/Markdown report with Table of Contents, citations, charts
```

## Research Tools (11 MCP tools)

### Core Research
| Tool | What it does |
|------|-------------|
| `research(query, depth, language)` | Full research cycle: search → scrape → analyze → report. Depth 1-3. |
| `quick_search(query)` | Fast web search, returns raw results |
| `read_url(url)` | Extract clean text from any URL |
| `analyze(text, question)` | Analyze provided text with LLM |

### Analysis Tools (unique to Sibyl)
| Tool | What it does |
|------|-------------|
| `compare(items)` | Structured side-by-side comparison table with metrics and recommendation |
| `swot(subject)` | Strengths / Weaknesses / Opportunities / Threats with evidence |
| `trends(keywords)` | Real Google Trends data: interest level, direction, rising searches |
| `timeline(topic)` | Chronological event table with dates and impact assessment |

### Financial Data
| Tool | What it does |
|------|-------------|
| `fetch_market_data(symbols)` | Real stock/ETF prices, trends, moving averages, 52-week range |
| `chart(symbols)` | Generate price trend charts (PNG) |

### Output
| Tool | What it does |
|------|-------------|
| `save_report(format)` | Save as PDF (with embedded charts) and/or Markdown |

## Research Depth

| Depth | What happens | LLM calls | Time |
|-------|-------------|-----------|------|
| 1 (quick) | 2-3 search queries, basic synthesis | ~3 | 20-30s |
| 2 (standard) | Sub-question decomposition, per-question analysis, cross-referencing, review | ~10 | 60-90s |
| 3 (deep) | + Knowledge gap filling, predictions with bull/bear/base case, confidence rating | ~13 | 90-120s |

## Multi-Provider Support

Sibyl works with any LLM. Auto-detects from environment variables:

| Provider | Env var | Model |
|----------|---------|-------|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| Gemini | `GEMINI_API_KEY` | `gemini/gemini-2.5-flash` |
| GLM (ZhipuAI) | `ZHIPUAI_API_KEY` | `glm-4-flash` |

Or configure multiple providers with roles:

```yaml
# sibyl.yaml
providers:
  - model: deepseek/deepseek-chat
    api_key: sk-xxx
    role: analysis

  - model: gemini/gemini-2.5-flash
    api_key: xxx
    role: fast

  - model: openai/glm-4-flash
    api_key: xxx
    api_base: https://open.bigmodel.cn/api/paas/v4
    role: chinese
```

## Example Reports

Reports generated by Sibyl on real topics:

- **Federal Reserve interest rate outlook 2026-2027** — 5 pages, 12 findings, 6 sources, analysis of "higher-for-longer" vs "steady easing" debate
- **Impact of Trump tariffs on trade 2026** — 5 pages, 10 findings, 4 sources, historical comparison to Smoot-Hawley, second-order effects on AI labor displacement
- **AI industry landscape 2026** — Market size ($538B), investment trends ($2.9T infrastructure), regulatory outlook, with NVDA/GOOGL/META stock charts

## Requirements

- Python 3.10+
- At least one LLM API key
- No other API keys needed (all search engines are free)

## License

MIT
