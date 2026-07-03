# Performance notes

Sibyl is I/O-bound: nearly all wall-clock time is spent on web search, page
scraping, and LLM calls. The pipeline is fully `async`, so the wins come from
(1) not serializing independent I/O and (2) not blocking the event loop with
synchronous CPU/network work.

## DeepSeek V4 + reasoning-mode control (v0.2.2)

DeepSeek's API now serves **`deepseek-v4-flash`** and **`deepseek-v4-pro`**
(the old `deepseek-chat` still works as a non-thinking alias to flash). Both V4
models are **thinking-first** — every call spends ~35-256 reasoning tokens
(scales with prompt complexity) before emitting content. Two consequences:

1. Short-output calls get starved. `_filter_sources` used `max_tokens=100`;
   with thinking on, reasoning ate the whole budget and returned empty → silent
   fallback. Any call under a few hundred tokens is at risk.
2. Reasoning adds latency (a complex prompt went 4.7s thinking vs 2.2-3.0s not).

Reasoning can be disabled per call via `extra_body={"thinking":{"type":"disabled"}}`
(litellm forwards it). So the split is:

- **Mechanical steps** (decompose, query generation, source filtering, gap-ID)
  run with `thinking=False` — they don't benefit from reasoning and need speed.
  `_llm_call(..., thinking=False)` injects the toggle only for deepseek models.
- **Analytical steps** (sub-question analysis, cross-analysis, synthesis,
  review) keep thinking on for quality.

### flash vs pro — flash wins for this workload

`Config.from_env` points every role at **v4-flash**. A head-to-head on sibyl's
long-form structured tasks (the deep-analysis section) found:

| | v4-flash | v4-pro |
|---|---|---|
| latency (avg of 3) | **30.1s** | 40.7s |
| latency spread | 28.6–31.8s (tight) | 29.1–46.6s (wide) |
| reasoning tokens | 484–606 | 396–**2500** |
| output at 2000-tok cap | completes all 4 sections | **truncates** mid-section |

Insight quality was comparable (both produced sophisticated framing, historical
parallels, second-order effects). But pro reasons more and writes longer, so at
sibyl's token budgets it truncates before finishing — and in the worst case
spends the entire budget on reasoning and returns empty (the `analysis=0` bug we
hit). Flash is faster, lower-variance, completes its output, and is cheaper. So
flash is used across the board. Keep pro only if you raise the analytical
`max_tokens` well above 2500 and can absorb the extra latency — otherwise it's
strictly worse here. Update model ids if DeepSeek renames the tier again.

`_llm_call` also retries once on empty content, since V4 thinking can
occasionally return nothing when reasoning exhausts the token budget.

### Thinking is disabled everywhere, not just mechanical steps

A follow-up head-to-head on the analytical generation (deep-analysis section):

| | thinking | no-thinking |
|---|---|---|
| latency (avg of 3) | 24.8s | **22.4s** |
| content | 6611 chars | **7812 chars** |
| concreteness | good | equal or better (gave a quantified share estimate) |

Reasoning gives no quality gain on sibyl's tasks — they're *generation*, not
puzzle-solving. So `_llm_call` defaults to `thinking=False` and no call site
overrides it. This shaved a few seconds per parallel analytical phase. Keep
`thinking=True` in reserve only if a genuinely reasoning-heavy call is ever
added.

### Cross-analysis: disable thinking + tolerant bullet parsing

The "Source Cross-Analysis" section was silently coming back empty
(`Consensus: 0` every run). The prompt output parses fine in isolation — the
real-run failure was thinking + 8 large sources overflowing the 1500-token
budget and truncating before the CONSENSUS section. Fix: `analyze_sources`
disables thinking (it's structured extraction), bumps the budget to 1800, and
the parser now accepts `-`, `*`, `•`, `·`, `—` and numbered bullets rather than
only `-`.

## Result

Depth-2 run (DeepSeek V4, same query): **183s → ~60s (~3x)**, with per-run
variance down from ±40% to a few seconds, and report quality held or improved
(longer, complete analysis; findings that no longer truncate).

## Research pipeline parallelization (v0.2.2)

Measured depth-2 run (DeepSeek V4): **183s → ~80s** across these changes.

- **Cross-analysis runs concurrently with synthesis.** The cross-analysis
  output is only attached to the report at the very end — it has no dependency
  on the synthesized sections — so `_do_research` `gather`s the two. Cross (~20s)
  hides fully under synthesis (the long pole). Note: pairing cross with
  *synthesis* beats pairing it with *sub-question analysis*, because synthesis
  is the longer of the two and absorbs cross entirely.
- **Analysis joins the first synthesis batch.** In `_synthesize`, the deep-
  analysis section depends only on the sources, not on summary/findings, so at
  depth 2+ it runs in the same `gather` as summary+findings instead of as a
  separate serial call. Predictions (depth 3) still run after, since they
  reference the finished summary.
- **Review/refine split into two parallel calls.** `_review_and_refine` used
  one giant call to regenerate summary + findings together; they're independent
  outputs, so they now refine in parallel (~38s → ~20s).
- **Semantic Scholar backs off cheaply.** It runs alongside the fast search
  engines and gates the search phase, so its 429 retry was cut from 2s+4s (3
  tries) to a single 1.2s retry (2 tries) with a 6s request timeout — it's a
  nice-to-have, so skip it fast rather than stall the whole search.

## What was optimized (v0.2.2)

### Search fan-out — the biggest win (~6x on the search phase)
`Researcher._do_research` used to `await search_web(...)` for each of the
15-20 generated queries **one at a time**. They're independent, so they now run
concurrently under an `asyncio.Semaphore(8)` (the cap keeps us from hammering a
single host and getting blocked).

Measured on 6 queries × 4 engines: **73s → 12s**, same result count.

### Semantic Scholar is opt-in, not per-query
The academic API is aggressively rate-limited (429 + backoff sleeps of 2-6s).
It used to run on *every* search query, so 20 queries meant 20 rate-limited
calls with retry sleeps — a large chunk of the old 73s. `search_web` now takes
`include_academic` and the orchestrator enables it on only the **first 2**
queries.

### One pooled HTTP client per run
Every search/scrape call created a fresh `httpx.AsyncClient` (new TCP + TLS
handshake each time). The same hosts (`lite.duckduckgo.com`, `news.google.com`,
`reddit.com`, Wikipedia) get hit once per query, so a single pooled client with
keep-alive (`max_keepalive_connections=10`) reuses connections across the whole
run. The client is created in `research()` and closed in a `finally`; all
search/scrape functions accept an optional `client=` and fall back to a
short-lived one when called standalone (e.g. MCP `quick_search`).

### HTML parsing: lxml + off the event loop
- `BeautifulSoup(html, "html.parser")` → `"lxml"` (2-3x faster parsing). `lxml`
  is now a declared dependency (the RSS `"xml"` mode already required it).
- Page extraction (`_extract_content`) is CPU-bound and was running inline in
  the async path, stalling other concurrent scrapes. It's now dispatched with
  `asyncio.to_thread`, so network I/O for other pages proceeds during a parse.
- Scrape concurrency raised from 5 → 8.

### Parallelized the remaining serial LLM/search steps
- Search-query **generation** for the main query + each sub-question ran in a
  serial loop; now a single `asyncio.gather`.
- The depth-3 **knowledge-gap** second search round ran its 3 queries serially;
  now concurrent.
- (Per-sub-question analysis and the summary/findings/analysis/predictions
  synthesis were already parallelized.)

### Financial data (`data.py`)
- `fetch_stock_data` is `async` but calls **blocking** yfinance
  (`ticker.history()`, `ticker.info` — each a synchronous network round trip
  that stalls the event loop). Now wrapped in `asyncio.to_thread`.
- `fetch_multiple` fetched symbols one at a time; now `asyncio.gather`.

## Guardrails / things to keep in mind
- The shared client is bound to the event loop that created it. It's created and
  closed inside `research()` (one `asyncio.run` per CLI invocation, one loop in
  the MCP server), so don't hoist it to module scope — that breaks across
  separate `asyncio.run` calls.
- Search concurrency is deliberately capped at 8. Raising it risks 429/403
  blocks from DuckDuckGo/Google News.
- If you add more search engines, thread the `client=` param through them too,
  and only add rate-limited APIs behind an opt-in flag like `include_academic`.
