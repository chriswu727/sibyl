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

### Sub-question analysis is depth-3 only

The per-sub-question analysis step (Step 5) generated 3 parallel LLM calls
(~8-14s) and fed the results to synthesis as "preliminary analysis" context. An
A/B (synthesis with vs without that context, identical sources) found the two
summaries equivalent in quality — the synthesis model does the analysis just as
well straight from the sources. So Step 5 is gated to depth 3, where it's still
needed as input to gap-finding. Depth-2 dropped ~54s → ~46s with no quality loss
(cross-analysis actually surfaced more consensus/disagreement points).

### What was NOT cut: the review step

The review/refine pass (~12s) was A/B'd the same way (draft vs reviewed). It
earns its keep: the reviewed summary leads with a definitive thesis that answers
the question up front (the draft buried it), tightens prose, and strengthens
causal reasoning — clearly better, not just shorter. Kept.

This is roughly the floor. The remaining depth-2 time is ~9s mechanical +
~23s synthesis‖cross + ~12s review; the last two are irreducible long-form
generation on flash and both earn their quality. Cutting further trades quality
for marginal speed.

### Density prompts + truncation fix

Profiling the real synthesis batch showed it is **decode-bound**: wall time is
gated by whichever section generates the most tokens (usually the deep
analysis), at ~85 tok/s. You cannot parallelize your way out of generating N
tokens — an A/B splitting analysis into two parallel halves gave only 1.2x
because each half still filled its own budget. The one real lever is generating
*fewer* tokens: an A/B found a "be information-dense, every sentence adds a
distinct fact, no filler" directive produced output that was both ~1.7x faster
*and* sharper (the verbose variant was padding). So all synthesis sections carry
that directive.

Profiling also surfaced a latent bug: the analysis/predictions sections were
**truncating at their token caps** (the old 2000 cap was below what a full
4-part analysis needs — even the pre-density version was cut off mid-section).
Caps are now sized to fit a complete section (analysis 2400, predictions 2200);
the density directive keeps the average well under that, so most runs finish
faster while verbose runs still complete instead of truncating.

Net: this stage is at the decode floor. Density buys tighter reports and a
faster average, not a step change — the remaining time is each section
generating its complete content, which is irreducible without thinning the
report.

### Fast mode (opt-in) — the last real speed lever

The review/refine pass (~5-10s) is the only remaining step that can be cut for
speed, and it A/B'd as genuinely valuable (leads with a definitive thesis,
tightens prose). So instead of dropping it, it's gated behind an opt-in
`Config.fast` flag: CLI `--fast`, MCP `research(..., fast=True)`, or `fast: true`
in YAML. Fast mode keeps search/scrape/synthesis/cross-analysis and skips only
review, for ~12-20% lower latency at the cost of a little polish. Default stays
full-quality.

### Prefix caching — a cost lever, not a speed one

Profiling showed `cache_hit=0`: the ~3670-token source context is reprocessed
by every synthesis call (they run in parallel and all miss DeepSeek's prefix
cache). A warm-up call (max_tokens=1 with the shared prefix) before the batch
makes the batch hit the cache (measured `cache_hit` 0 → 11264 tokens), which
cuts input-token cost ~10× on that portion. But it's **latency-neutral**
(15.2s → 15.4s — the warm call's prefill offsets the batch's prefill savings),
so it's deliberately NOT wired in: it would add complexity for a cost win only,
and decode (not prefill) dominates latency. Revisit if cost becomes the
priority over speed.

## Result

Depth-2 run (DeepSeek V4, averaged over trials): **183s → ~40-46s (~4x)**
(fast mode ~34s),
per-run variance down from ±40% to a few seconds, report quality improved
(denser prose, no truncated sections, working cross-analysis).
Depth-1 ~15s, depth-3 ~56s (down from ~80s).

## Research pipeline parallelization (v0.2.2)

Measured depth-2 run (DeepSeek V4): **183s → ~80s** across these changes.

- **Cross-analysis runs concurrently with synthesis.** The cross-analysis
  output is only attached to the report at the very end — it has no dependency
  on the synthesized sections — so `_do_research` `gather`s the two. Cross (~20s)
  hides fully under synthesis (the long pole). Note: pairing cross with
  *synthesis* beats pairing it with *sub-question analysis*, because synthesis
  is the longer of the two and absorbs cross entirely.
- **Analysis and predictions join the first synthesis batch.** In
  `_synthesize`, the deep-analysis (depth 2+) and predictions (depth 3) sections
  depend only on the sources, not on summary/findings — A/Bs confirmed feeding
  them the finished summary gave no quality gain. So they run in the same
  `gather` as summary+findings instead of as serial follow-up calls. This alone
  took depth-3 from ~80s to ~56s.
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
