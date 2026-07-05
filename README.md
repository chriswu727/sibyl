<div align="center">

# Sibyl

**Keyless deep-research for your AI agent.**

Sibyl retrieves the web — searching across engines, scraping and cleaning pages, cross-referencing sources — and hands the evidence to *your* model to reason over. Or it runs the whole research cycle itself and returns a cited report.

[![tests](https://github.com/chriswu727/sibyl/actions/workflows/tests.yml/badge.svg)](https://github.com/chriswu727/sibyl/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sibyl-research?color=blue)](https://pypi.org/project/sibyl-research/)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![web search: keyless](https://img.shields.io/badge/web_search-keyless-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Two ways to use it

Sibyl runs as an **MCP server** and a **CLI / Python library**. The key choice is *who does the reasoning*.

### 1. Retrieval provider — your model is the brain (recommended, keyless)

`gather_sources(query)` searches, scrapes, dedupes, and returns the top **full-text sources** — without writing an answer. Your agent (Claude, or any MCP host) reads the evidence, cross-references it, and answers itself, citing sources and abstaining when they don't contain the answer. **No API key required.**

This is the highest-quality path, because a frontier model applied to real retrieved evidence beats a mid-tier model doing the synthesis — and it doesn't fabricate when the answer isn't found.

### 2. One-shot pipeline — Sibyl is the brain (needs a provider key)

`research(query, depth)` runs the full cycle — decompose → search → scrape → rank → synthesize → **verify each finding against its source** → report — using Sibyl's own configured LLM (DeepSeek by default). Use it when you want a finished report in a single call.

## Benchmarks

Measured on 30 questions from the official **SimpleQA** test set (human-verified gold, deliberately obscure long-tail facts), graded on the CORRECT / INCORRECT / NOT_ATTEMPTED rubric.

| Setup | Hard SimpleQA | Fabrication |
|---|---|---|
| **Host model over `gather_sources`** (keyless) | **~93%** | **~0** — abstains instead of guessing |
| Sibyl one-shot pipeline (DeepSeek-flash) | 17% | high |

The gap isolates one variable: **the model consuming the sources is the ceiling, not the sources.** The same keyless retrieval that yields 17% with a mid-tier synthesizer yields ~93% when a frontier host model reasons over it. Full method, per-question results, and honest caveats: [`docs/EVAL_HOST_CLAUDE.md`](docs/EVAL_HOST_CLAUDE.md).

> Reproduce the keyless path: `python scripts/gather.py "<query>"` and reason over the output yourself. Reproduce the one-shot path: `python scripts/eval.py --depth 2 --dataset evals/gold/simpleqa_real_30.jsonl`.

## Quick start

### MCP server (Claude Code, Cursor, …)

```bash
pip install sibyl-research

# Keyless — retrieval-provider mode, your host model reasons:
claude mcp add sibyl -- sibyl-mcp

# Or with a key, to also enable the one-shot research() tool:
claude mcp add sibyl -e DEEPSEEK_API_KEY=sk-... -- sibyl-mcp
```

Then, in your agent:

> "Research the Serbian quarterfinalist at the 2018 Madrid Open" — *uses `gather_sources`, you synthesize*

> "Compare NVIDIA vs AMD vs Intel for AI workloads" — *one-shot `research()` + `compare()`*

### CLI

```bash
pip install sibyl-research
export DEEPSEEK_API_KEY=sk-...   # or OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, …

sibyl "Canadian housing market outlook 2026"                       # standard
sibyl "Will NVIDIA keep AI-chip dominance?" -d 3 --symbols NVDA,AMD --pdf   # deep + charts
sibyl "加拿大移民政策变化" -l zh --pdf -o reports/                    # Chinese output
```

## Tools (12 MCP tools)

| Group | Tool | What it does |
|---|---|---|
| **Retrieval** | `gather_sources(query)` | Keyless search + scrape + dedup; returns full-text sources for *you* to reason over |
| | `quick_search(query)` | Raw search hits (title / url / snippet), no scraping |
| | `read_url(url)` | Clean full text of one page |
| **Research** | `research(query, depth)` | Full one-shot cycle with source-verified findings (depth 1–3) |
| | `analyze(text, question)` | Reason over text you provide |
| **Analysis** | `compare(items)` | Side-by-side comparison table with metrics + recommendation |
| | `swot(subject)` | Strengths / Weaknesses / Opportunities / Threats, evidence-backed |
| | `trends(keywords)` | Real Google Trends: interest, direction, rising queries |
| | `timeline(topic)` | Chronological event table with impact |
| **Finance** | `fetch_market_data(symbols)` | Real prices, moving averages, 52-week range |
| | `chart(symbols)` | Price trend charts (PNG) |
| **Output** | `save_report(format)` | PDF (with embedded charts) and/or Markdown |

## How the one-shot pipeline works

```
You ask a question
  ├─ 1. Decompose into 3–5 focused sub-questions
  ├─ 2. Generate diverse, perspective-guided search queries
  ├─ 3. Search 4 keyless engines (DuckDuckGo, Google News, Reddit, Wikipedia; Mojeek fails over)
  ├─ 4. Scrape sources (browser headers, retry, JS-render fallback for thin pages)
  ├─ 5. Dedupe + rank by relevance
  ├─ 6. Analyze each sub-question; identify knowledge gaps → search again
  ├─ 7. Cross-reference (sentiment, consensus, disagreements)
  ├─ 8. Section-by-section synthesis (Summary, Findings, Analysis, Predictions)
  ├─ 9. Verify every finding against its cited source — flag the unsupported
  └─ Output: PDF / Markdown report with ToC, citations, charts
```

Depth controls cost: **1 (quick)** ~20–30s · **2 (standard)** ~60–90s · **3 (deep)** adds gap-filling + bull/bear/base predictions.

## Multi-provider

Sibyl auto-detects a provider from the environment; `gather_sources` needs none.

| Provider | Env var | Default model |
|---|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-v4-flash` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| Gemini | `GEMINI_API_KEY` | `gemini/gemini-2.5-flash` |
| GLM (ZhipuAI) | `ZHIPUAI_API_KEY` | `glm-4-flash` |

Configure several providers with per-role routing in `sibyl.yaml` — e.g. cheap model for search/ranking, a stronger one for `synthesis` and `verify`:

```yaml
providers:
  - model: deepseek/deepseek-v4-flash
    api_key: sk-xxx
    role: search
  - model: anthropic/claude-sonnet-4-20250514
    api_key: sk-ant-xxx
    role: synthesis
```

## Requirements

- Python 3.10+
- **`gather_sources` and all web search are keyless** — no API keys to search the web
- One LLM key only for the one-shot `research()` / CLI paths

## License

MIT
