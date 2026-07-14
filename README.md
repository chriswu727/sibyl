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

`gather_bundle(query)` searches, scrapes, dedupes, and returns structured evidence — without writing an answer. Each bundle includes versioned source/passage IDs, content hashes, retrieval timestamps, and diagnostics. `gather_sources(query)` runs the same retrieval and renders it as readable `[Source N]` blocks for conversational use. Both accept `ranker="lexical"` (default), `ranker="flashrank"` (optional local cross-encoder), or `ranker="none"` (preserve retrieval order). Your agent reads the evidence, cross-references it, and answers itself, citing sources and abstaining when they don't contain the answer. **No API key required.**

This is the highest-quality path, because a frontier model applied to real retrieved evidence beats a mid-tier model doing the synthesis and can abstain when the retrieved evidence does not contain the answer.

### 2. One-shot pipeline — Sibyl is the brain (needs a provider key)

`research(query, depth)` runs the full cycle — decompose → search → scrape → rank → synthesize → **verify each finding against its source** → report — using Sibyl's own configured LLM (DeepSeek by default). Use it when you want a finished report in a single call.

If no usable evidence is retrieved, or the configured LLM backend fails, this path returns an explicit failure/insufficient-evidence result instead of synthesizing from model memory.

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

> "Research the Serbian quarterfinalist at the 2018 Madrid Open" — *uses `gather_bundle` or `gather_sources`; you synthesize*

> "Compare NVIDIA vs AMD vs Intel for AI workloads" — *one-shot `research()` + `compare()`*

### CLI

```bash
pip install sibyl-research
export DEEPSEEK_API_KEY=sk-...   # or OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, …

sibyl "Canadian housing market outlook 2026"                       # standard
sibyl "Will NVIDIA keep AI-chip dominance?" -d 3 --symbols NVDA,AMD --pdf   # deep + charts
sibyl "加拿大移民政策变化" -l zh --pdf -o reports/                    # Chinese output
```

## Tools (13 MCP tools)

| Group | Tool | What it does |
|---|---|---|
| **Retrieval** | `gather_bundle(query)` | Structured keyless evidence with stable bundle/source/passage IDs, hashes, timestamps, and diagnostics |
| | `gather_sources(query)` | The same retrieval rendered as full-text `[Source N]` blocks for conversational use |
| | `quick_search(query)` | Raw search hits (title / url / snippet), no scraping |
| | `read_url(url)` | Clean full text of one page |
| **Research** | `research(query, depth)` | Full one-shot cycle; claim verification runs at depth 2+ unless fast/disabled |
| | `analyze(text, question)` | Reason over text you provide |
| **Analysis** | `compare(items)` | Side-by-side comparison table with metrics + recommendation |
| | `swot(subject)` | Strengths / Weaknesses / Opportunities / Threats, evidence-backed |
| | `trends(keywords)` | Real Google Trends: interest, direction, rising queries |
| | `timeline(topic)` | Chronological event table with impact |
| **Finance** | `fetch_market_data(symbols)` | Real prices, moving averages, 52-week range |
| | `chart(symbols)` | Price trend charts (PNG) |
| **Output** | `save_report(format)` | PDF (with embedded charts) and/or Markdown |

`gather_bundle` currently returns SourceBundle schema `1.5`. Its `bundle_id` is derived from the trimmed query, bundle status, selected URLs, and evidence hashes. Each source contains up to three passages with source-text offsets and bundle-scoped `citation_id` values such as `sb_…/S1/P1`; the combined passage text stays within `chars_per_source`. `content_hash` values are SHA-256. `relevance_score` and passage `score` are 0–1 retrieval scores from the actual ranking backend, not probabilities or correctness judgments; they are `null` when ranking is disabled. Diagnostics distinguish `requested_ranking_method` from the actual `ranking_method` and expose `ranking_warning` when FlashRank falls back to `lexical_v1`.

Schema 1.5 also reports `substantive_sources`, `evidence_chars`, `evidence_sufficiency`, and machine-readable `sufficiency_reasons`. The deterministic sufficiency check marks evidence as insufficient when there is no substantive full text, less than 200 selected evidence characters, or under 25% lexical query-term coverage. Evidence with fewer than two substantive sources, fewer than two independent domains, or no usable lexical query terms is marked `limited`; limited evidence still returns bundle status `ok`, while insufficient evidence returns `insufficient_evidence` even when lead sources are included. These are retrieval-recall signals, not proof that the evidence is true. `quality_score` remains `null` until a separate source-quality evaluator computes it. Check `status` before synthesis.

Within one MCP server process, matching `gather_bundle` and `gather_sources` calls share in-flight work and reuse successful evidence for 30 seconds. Failed retrievals are never cached; cached bundles retain their original `retrieved_at` provenance timestamps.

### Offline ranker regression

Run the fixed, network-free ranker checks before changing retrieval scoring:

```bash
python scripts/eval_retrieval.py --ranker lexical
pip install 'sibyl-research[rerank]'
python scripts/eval_retrieval.py --ranker flashrank
```

The command reports per-case first-relevant rank plus aggregate Hit@1 and MRR, and exits non-zero below the checked-in regression floors. The small synthetic set covers multilingual text, identifiers, version numbers, and high-overlap distractors; it is a deterministic regression guard, not a claim about production search accuracy.

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

Source reranking defaults to the dependency-free local `lexical` backend, so ranking does not consume an extra LLM call. Install the optional cross-encoder with `pip install 'sibyl-research[rerank]'`, then pass `ranker="flashrank"` to `gather_bundle` / `gather_sources` or set `reranker: flashrank` for the one-shot pipeline. The model is loaded lazily and cached in-process. If FlashRank is unavailable or fails, Sibyl falls back to lexical ranking; SourceBundle diagnostics disclose that fallback. Use `ranker="none"` or `reranker: none` to preserve retrieval order. The one-shot pipeline also supports `reranker: llm` explicitly.

## Multi-provider

Sibyl auto-detects a provider from the environment; `gather_bundle` and `gather_sources` need none.

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
- **`gather_bundle`, `gather_sources`, and all web search are keyless** — no API keys to search the web
- URL fetching is restricted to public HTTP(S) destinations on ports 80/443; local, private, credential-bearing, and unsafe redirect targets are rejected, and decompressed response bodies are capped at 2 MiB
- One LLM key only for the one-shot `research()` / CLI paths

## License

MIT
