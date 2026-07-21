# Sibyl — Agent Guide

This file helps AI agents (Claude Code, Cursor, etc.) use Sibyl effectively via MCP.

## Setup

```bash
pip install sibyl-research
# Optional local cross-encoder ranking:
pip install 'sibyl-research[rerank]'
# Optional experimental one-shot reports and report export:
pip install 'sibyl-research[report]'
# Retrieval-provider mode (recommended) needs NO key — you do the reasoning:
claude mcp add sibyl -- sibyl-mcp
# For the experimental research() tool, add a provider key so sibyl's own model runs it:
claude mcp add sibyl -e DEEPSEEK_API_KEY=sk-... -- sibyl-mcp --profile report
```

## Core boundary — Sibyl retrieves, the host reasons

**Recommended: YOU (the host model) research; sibyl only retrieves.** This is
keyless and gives the best quality, because YOUR reasoning is applied to real
evidence — a mid-tier model doing the synthesis fabricates on hard questions,
whereas you can cross-reference sources and abstain when they don't answer.

```
# structured retrieval for an agent pipeline (Loop/Argus-style):
gather_bundle("Serbian quarterfinalist 2018 Madrid Open men's singles")
→ consume ranked sources[].evidence[] passages; cite each by citation_id.
→ inspect content_origin; treat search_snippet evidence as a lead, not full text.
→ compare content_cluster_id; matching clusters are not independent corroboration.
→ inspect status and diagnostics before synthesis; relevance defaults to lexical_v1.
→ optional: ranker="flashrank"; inspect ranking_method and ranking_warning for fallback.
→ query_term_coverage is a recall hint, not proof of factual sufficiency.
→ quality_score=null means source quality has not been computed.

# readable retrieval for a conversational host:
gather_sources("Serbian quarterfinalist 2018 Madrid Open men's singles")
gather_sources("2018 Madrid Open men's singles draw results")   # call again with sub-queries
→ read the returned [Source N] blocks, cross-reference, answer WITH citations.
→ if the sources don't contain it, gather more or say you don't know — never guess.

# bounded workflow for a dependent or multi-part question:
gather_evidence(question="In what year was the company that created CUDA founded?")
→ if next_action is decompose_query, call again with loop_id and one atomic query.
→ finish with the synthesis-ready supporting E-step IDs.
→ synthesize only after the loop status is ready; the host still does all reasoning.
```

**Experimental one-shot report: Sibyl uses a configured model** (needs a provider key):
```
research(query, depth=2)   # search→scrape→rank→synthesize→verify→report, by sibyl's LLM
```
Prefer `gather_bundle` for programmatic agents and `gather_sources` for readable
conversation context. In both cases, do your own synthesis for factual/hard
questions. `research()` is a secondary convenience for users who explicitly
want a model-backed report in one call.

The default `auto` MCP profile exposes only the five keyless retrieval tools
unless the report extra and an LLM credential are both available. Use
`sibyl-mcp --list-tools` to inspect the active surface. Thin-page Jina rendering
is opt-in through `render_thin_pages=true` because it discloses target URLs to a
third-party service.

## Recommended Workflows

### Experimental one-shot report
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
| Experimental one-shot report | `research(query, depth=1/2/3)` |
| Side-by-side comparison | `compare(items, query)` |
| SWOT analysis | `swot(subject)` |
| Google Trends data | `trends(keywords)` |
| Event timeline | `timeline(topic)` |
| Stock/ETF data | `fetch_market_data(symbols)` |
| Price chart | `chart(symbols, period)` |
| **Programmatic evidence retrieval (keyless)** | **`gather_bundle(query)` — structured sources, passages, provenance, and diagnostics** |
| **Readable evidence retrieval (keyless)** | **`gather_sources(query)` — renders full-text source blocks; you synthesize** |
| **Bounded multi-step retrieval (keyless)** | **`gather_evidence(question)` — host-planned atomic steps with a four-call ceiling** |
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
- After changing retrieval ranking, run `python scripts/eval_retrieval.py --ranker lexical`; with the optional extra installed, also run it with `--ranker flashrank`
