# Research-Quality Build Plan (borrowed capabilities)

> **Status (2026-07-14):** The implementation batches in this document have shipped in the 0.3.0 codebase and the test suite has grown from the 60-test baseline to more than 200 tests. This file is retained as an architectural decision record, not an active checklist. Current release steps live in [`RELEASING.md`](RELEASING.md); remaining product work is cross-project MCP integration, production observability, and broader source-quality evaluation.

Reconciled from an 8-capability spec workflow. Executed batch by batch; eval-measured.

I have the full ground truth now: 60 tests green under `.venv/bin/python`, and I've read every shared function the 8 specs touch. Two things the specs got right that anchor the reconciliation: (1) there is a **real pre-existing citation-alignment bug** in `_synthesize` — `report.sources` is built from `search_results` filtered by URL (search order), while the `[Source N]` numbers in the prompt enumerate `good_pages[:12]` (page order), so the rendered Sources list is misnumbered against the prose citations; (2) two specs (1 and 7) both add a "verification" pass under different names/defaults, and two specs (3 and 7) both overload the name `reflect_rounds` with incompatible meanings. Both must be collapsed to one.

Here is the reconciled build plan.

---

# SIBYL — Reconciled Build Plan (8 specs → 1 ordered build)

Baseline verified: `.venv/bin/python -m unittest discover tests` → **60 passed**. Every batch below must keep that green; new tests are additive.

## 0. Cross-cutting conflict resolutions (decided once, applied everywhere)

| # | Conflict | Decision | Why |
|---|---|---|---|
| C1 | Spec 1 `verify_claims: bool = True` (standalone `verifier.py`, keyless lexical fallback, MARK-mode, non-destructive) **vs** Spec 7 `verify: bool = False` (`_verify_claims` in-researcher, rewrites findings, needs stronger model). | **One capability.** Adopt Spec 1's architecture (`verifier.py`, keyless-degradeable, offline-testable, default-ON mark-mode). Drop Spec 7's separate `_verify_claims`. Keep Spec 7's *role routing* so `verify_findings` calls `get_provider('verify')` (keyless-falls-back to `analysis`). Single flag `verify_claims: bool = True`; destructive drop stays `verify_drop_unsupported: bool = False`. | Spec 7's version is a strict subset that violates keyless-by-default (assumes a stronger model to be worth running). Spec 1 is keyless, offline-testable, and non-destructive — safe default-on. Role routing is orthogonal and kept. |
| C2 | Spec 3 `Config.reflect_rounds` = number of expensive reflect+refill cycles (default **0/off**) **vs** Spec 7 `EffortTier.reflect_rounds` = gate for the existing review pass (standard=**1/on**). Same name, incompatible defaults. | **Split cleanly.** (a) The existing review/refine pass stays gated on `depth>=2 and not fast` — **do NOT** rebind it to a tier field. (b) Remove `reflect_rounds` from `EffortTier` entirely. (c) Spec 3's reflect loop keeps its own `Config.reflect_rounds: int = 0` (opt-in). Tiers carry caps only. Deep tier does **not** auto-enable reflect. | Rebinding review to `tier.reflect_rounds` is "behavior-equivalent today" but couples two unrelated features and would silently turn Spec 3's expensive loop on for every standard run. Keeping caps-only tiers preserves keyless-fast defaults and predictable latency. |
| C3 | Spec 4 rewrites `_filter_sources` body (JSON 0-10 rerank) **and** Spec 7 changes its provider (`general`→`compaction`). | Compose: adopt Spec 4's JSON-scoring body, call `get_provider('compaction')` (keyless→`fast`→`general`). | Non-overlapping edits to the same function; merge is mechanical. |
| C4 | `_synthesize` gets `compacted` (Spec 5), `tier` (Spec 7), `supporting_snippet` population + source-order fix (Spec 6), and `build_source_context` (Spec 1). | **One rewrite** of `_synthesize` with signature `(query, search_results, pages, depth, sub_analyses=None, language="auto", tier=None, compacted=None)`. Context via shared `build_source_context`; token budgets from `tier`; sources built **in cited-page order** with `supporting_snippet`. | Four specs edit the same 60 lines; a single coherent rewrite is the only sane merge. Also fixes the real citation-alignment bug. |
| C5 | `[Source N]` mapping invariant (Spec 1's #1 risk). | Introduce explicit `cited_pages` in `_do_research` = `compacted if compacted else good_pages[:max_synth_sources]`. `build_source_context(cited_pages)` feeds synthesis; `verify_findings(report.key_findings, cited_pages, …)` verifies against the *same* list. `report.sources` built from `cited_pages` in order. | Guarantees synthesis prompt, verifier, and rendered Sources all number sources identically. Kills the pre-existing misnumbering bug. |
| C6 | Spec 6 `finding_confidence: List[str]` (populated "by the verification pass") **vs** Spec 1 `finding_verifications: List[FindingVerification]`. | **One field:** `finding_verifications: List[FindingVerification]`. Reporter derives the confidence marker and the `(unverified)` flag from it by index. Drop `finding_confidence` to avoid two parallel lists that desync. | Single source of truth; `FindingVerification` already carries `.supported` and `.confidence`. |
| C7 | `_canned_llm` test fixture matcher (`"relevance"→"1,2,3,4,5"`) breaks when `_filter_sources` emits JSON. | New rerank prompt **retains the word "relevance"** (`"Score the relevance of each source (0-10)…"`) AND the fixture's relevance branch is updated to return `{"scores":[{"id":i,"score":11-i}…]}`. Verify/reflect/compaction prompts fall through to the prose branch by default (parse-fail → safe fallback), so they need no fixture change until their dedicated tests. | Keeps all 60 green with a one-line fixture edit; matcher keyword stays valid. |

---

## 1. Single merged shared-file diffs (edit these files **once**)

### `sibyl/config.py`

Add module-level (after imports):

```python
@dataclass(frozen=True)
class EffortTier:
    name: str
    depth: int
    max_queries: int
    max_urls: int
    synthesis_max_tokens: int
    latency_target_s: int

TIERS = {
    "quick":    EffortTier("quick",    1, 3,  8,  1200, 30),
    "standard": EffortTier("standard", 2, 6,  20, 1600, 90),   # reproduces today exactly
    "deep":     EffortTier("deep",     3, 10, 30, 2000, 240),
}
_DEPTH_TO_TIER = {1: "quick", 2: "standard", 3: "deep"}
_ROLE_FALLBACKS = {
    "verify":    ["verify", "synthesis", "analysis", "general"],
    "synthesis": ["synthesis", "analysis", "general"],
    "analysis":  ["analysis", "general"],
    "compaction":["compaction", "fast", "general"],
    "fast":      ["fast", "general"],
    "general":   ["general"],
}
```

`Config` dataclass — **all new fields in one diff** (append at end so `from_yaml` stays positional-free):

```python
    # verification (C1)
    verify_claims: bool = True
    verify_drop_unsupported: bool = False
    # js render (Spec 2)
    js_render: bool = True
    js_render_threshold: int = 500
    # dedup + rerank (Spec 4)
    dedup: bool = True
    reranker: str = "llm"            # "llm" | "flashrank" | "none"
    rerank_top_n: int = 12
    # perspectives + compaction (Spec 5)
    perspectives: bool = True
    compact_sources: bool = False
    max_synth_sources: int = 12
    # enriched citations (Spec 6)
    rich_citations: bool = True
    # tiers (Spec 7)
    tier: str = "standard"
    # reflect loop (Spec 3)
    reflect_rounds: int = 0
```

`get_provider` — replace the exact-match-else-`providers[0]` body with the fallback chain (strict superset; preserves `test_role_lookup_and_fallback`):

```python
    def get_provider(self, role: str = "general") -> Provider:
        for r in _ROLE_FALLBACKS.get(role, [role]):
            for p in self.providers:
                if p.role == r:
                    return p
        return self.providers[0] if self.providers else Provider(model="deepseek/deepseek-v4-flash")

    def resolve_tier(self, depth: int = 0) -> EffortTier:
        if depth:
            return TIERS.get(_DEPTH_TO_TIER.get(depth, self.tier), TIERS["standard"])
        return TIERS.get(self.tier, TIERS["standard"])
```

`from_yaml` — read every new key with the dataclass default (`data.get("verify_claims", True)`, … `data.get("reflect_rounds", 0)`, `data.get("tier", "standard")`).

`from_env` — after the DeepSeek 3-provider block, append optional stronger providers (keyless default sets neither, so nothing changes):

```python
    for role, mkey in (("synthesis", "SIBYL_SYNTHESIS_MODEL"), ("verify", "SIBYL_VERIFY_MODEL")):
        m = os.environ.get(mkey)
        if m:
            if "/" not in m: m = f"deepseek/{m}"
            providers.append(Provider(model=m,
                api_key=os.environ.get(f"SIBYL_{role.upper()}_API_KEY", ""),
                api_base=os.environ.get(f"SIBYL_{role.upper()}_API_BASE", ""), role=role))
```

### `sibyl/researcher.py` — dataclass additions (one merged diff)

```python
@dataclass
class Source:
    url: str
    title: str
    snippet: str
    relevance: str = ""
    supporting_snippet: str = ""          # Spec 6

# in verifier.py, imported here:
# @dataclass FindingVerification: index:int; supported:bool; confidence:str; cited:List[int]; note:str=""; method:str="llm"

@dataclass
class ResearchReport:
    ...existing fields...
    finding_verifications: List["FindingVerification"] = field(default_factory=list)   # C6
```

Both additions have defaults → no call site or existing test breaks.

### `sibyl/scraper.py` — one shared helper (used by `_synthesize` AND `verifier.py`, C5)

```python
def build_source_context(pages, limit: int = 12, per_char: int = 4000) -> str:
    """The exact [Source i] block synthesis and the verifier both use — single
    source of truth for the [Source N] → pages[N-1] mapping."""
    parts = [f"[Source {i}: {p.title}]\nURL: {p.url}\n{p.text[:per_char]}\n"
             for i, p in enumerate(pages[:limit], 1)]
    return "\n---\n".join(parts)
```

---

## 2. Ordered build batches

**Build the eval harness FIRST** so every subsequent batch reports a measured delta against a fixed baseline — the harness only calls the public `research()` API, touches zero product code, and re-runs unchanged as capabilities land.

### Batch 0 — Eval harness (Spec 8) · measurement instrument, no product code
- **New:** `scripts/eval.py`, `evals/gold/simpleqa_frames_20.jsonl` (20 hand-verified Q/gold/aliases), `evals/cache/` (gitignored).
- `eval.py`: `load_dataset`, `_grade_call(provider, prompt, max_tokens=64)` (local copy of the `_llm_call` contract — thinking-disabled, 3-retry, **temperature=0**; deliberately NOT `Researcher._llm_call`, so `researcher.py` stays untouched), `extract_answer(provider, q, report)`, `judge(provider, q, gold, aliases, predicted)` returning the SimpleQA trichotomy **CORRECT / INCORRECT / NOT_ATTEMPTED**, `run_one` (writes raw `ResearchReport`+answer to `evals/cache/{id}_d{depth}.json`), `main` (`--depth 1`, `--limit`, `--concurrency 4`, `--dataset`, `--score-only` for network-free re-grade, `--write-badge`).
- Metrics: headline **CORRECT/total (%)**, plus attempted% and correct-given-attempted.
- **Run now → record baseline** (expected low: current pipeline is one 1600-token synthesis, no grounding pass).
- No test-count impact.

### Batch 1 — Shared scaffolding · no default behavior change
- Apply **all** of §1 (config fields/tiers/role-fallback/from_env; dataclass additions; `build_source_context`).
- `researcher.py`: add `_best_snippet(query, text, max_len=240)` staticmethod (Spec 6's pure-Python query-overlap sentence picker; empty/CJK query → `text[:max_len]` fallback).
- **Tests:** extend `test_config.py` (defaults for all new fields; `from_yaml` reads them; `get_provider('synthesis'/'verify'/'compaction')` fallback chain; `resolve_tier(1/2/3)`; `from_env` synthesis/verify providers appear only with env set). `test_snippet.py` for `_best_snippet`.
- Gate: 60 existing + new all green. **No pipeline behavior change** — get_provider is a superset, all new flags inert until wired.

### Batch 2 — Retrieval precision before synthesis (Spec 2 + Spec 4)
Shapes `good_pages`; all edits sit upstream of synthesis.
- **New module `sibyl/dedup.py`:** `canonical_url(url)` (drop scheme/`www.`/fragment/trailing-slash, strip `utm_*`+tracking params, sort remaining query, non-http→lowercased input), `dedup_pages(pages)` (first-occurrence position wins, longer `.text` wins on collision).
- **`scraper.py`:** `scrape_url` gains `js_render=True, js_render_threshold=500`; in the `200` branch, after `_extract_content`, if `js_render` and content-type is html-ish and `len(page.text) < threshold`, `await _try_jina` through a new shared gate and keep the longer text. `_try_jina` wrapped in lazy per-event-loop `asyncio.Semaphore(2)` + 3.0s keyless min-interval throttle (0s when `JINA_API_KEY` set), 7s cap. `scrape_urls` threads the two params.
- **`researcher._do_research`:** pass `js_render=cfg.js_render, js_render_threshold=cfg.js_render_threshold` to **both** `scrape_urls` calls; insert `if cfg.dedup: good_pages = dedup_pages(good_pages)` between the "Total usable sources" line and the 4b guard; apply tier caps here — `search_queries = search_queries[:tier.max_queries]` after dedup, and `scrape_count = min(len(unique_results), cfg.max_sources*2, tier.max_urls)`.
- **`researcher._filter_sources`:** rewrite to ranked JSON scoring, `provider = get_provider('compaction')`, keep top `rerank_top_n`, `reranker` dispatch (`llm`/`flashrank` lazy-import→fallback/`none`). Add `_parse_scores(text, n)`.
  - **Prompt (json_mode, max_tokens=400), retains "relevance" keyword (C7):**
    > Score the **relevance** of each source (0–10) for answering this research question. RESEARCH QUESTION: {query} SOURCES: {n. [title] — text[:200]} … 10 = directly answers; 5 = partial; 0 = off-topic/spam/error. Judge topical relevance only. Return json: `{"scores":[{"id":<n>,"score":<0-10>}, …]}`, every source once. Example: `{"scores":[{"id":1,"score":9},{"id":2,"score":3}]}`
- **`pyproject.toml`:** `[project.optional-dependencies] rerank = ["flashrank"]` (optional only).
- **`cli.py`:** `--js-render/--no-js-render`, `--js-render-threshold`, (rerank/dedup stay config/yaml).
- **Tests:** `test_dedup.py`, `test_rerank.py`, `test_js_render.py`; update `_canned_llm` relevance branch → JSON scores.

### Batch 3 — Synthesis rewrite + tiers + compaction + perspectives + citation-order fix (Spec 5 + Spec 7 + Spec 6-population)
The single `_synthesize` rewrite (C4/C5) plus its `_do_research` wiring.
- **`researcher._generate_search_queries`** (behind `cfg.perspectives`, default on): perspective-guided prompt — silently enumerate `max(3, min(num_queries+1, 5))` stakeholder perspectives, emit the **same count** of one-per-line queries anchored per perspective. **Retain the substring "search queries"** (matcher). Counts/parsing unchanged.
- **`researcher._decompose_question`** (behind `cfg.perspectives`): append one stakeholder line; **retain "Break this" + "sub-questions"**.
- **New `researcher._compact_sources(query, pages)`:** fan out one `get_provider('compaction')` `_llm_call` per page under `Semaphore(12)`, `max_tokens=400`, `page.text[:5000]`, returns 3–6 query-relevant bullets or sentinel `IRRELEVANT` (drops page). `asyncio.gather(return_exceptions=True)`; non-`WebPage` filtered out; output text sliced `[:1200]`.
  - **Prompt:** > Extract 3–6 bullet points from this source that bear on RESEARCH QUESTION: {query}. SOURCE [{title}]: {text}. Keep numbers/dates/names; omit boilerplate/ads. If nothing relevant, reply exactly `IRRELEVANT`.
- **`researcher._synthesize` — single merged rewrite:**
  - Signature `(query, search_results, pages, depth, sub_analyses=None, language="auto", tier=None, compacted=None)`.
  - `synth_pages = compacted if compacted else pages`; `context = build_source_context(synth_pages, limit=(cfg.max_synth_sources if compacted else 12))`. (Compacted bullets already dense → no per-char truncation needed; pass a large `per_char` on that path.)
  - Token budgets from tier: `t = (tier or TIERS['standard']).synthesis_max_tokens`; summary=`t`, findings=`int(t*1.25)`, analysis=`int(t*1.5)`, predictions=`int(t*1.375)`. **standard=1600 → 1600/2000/2400/2200 (byte-identical to today).**
  - Provider `get_provider('synthesis')` (keyless→analysis).
  - **Citation-order fix (kills the real bug):** build `sources` from `synth_pages[:limit]` **in enumerate order**, `supporting_snippet=_best_snippet(query, p.text)` when `cfg.rich_citations`, so `report.sources[N-1]` ⇔ `[Source N]`. Preserve the `search_results`-only fallback when `pages` empty.
  - Density directive, warm-cache guard, findings retry, `_parse_findings` — unchanged.
- **`researcher._review_and_refine`:** provider → `get_provider('synthesis')`.
- **`researcher._do_research`:** after `depth` resolved, `tier = cfg.resolve_tier(depth)`. After 4b filter, insert **Step 4c**: `compacted = None; if cfg.compact_sources: compacted = await self._compact_sources(query, good_pages[:cfg.max_synth_sources]); compacted = compacted if len(compacted) >= 3 else None`. Define `cited_pages = compacted if compacted else good_pages[:cfg.max_synth_sources]` (C5). Pass `tier=tier, compacted=compacted` into the `_synthesize` call inside the existing gather (keep it overlapping `_cross_analysis`, which still reads full-text `good_pages`).
- **`cli.py`:** `--effort/-e Choice(quick,standard,deep)` (maps to `depth = TIERS[effort].depth`, wins over `--depth`), `--compact` (sets `compact_sources`; bumps `max_synth_sources` 12→40 if left default), `--max-synth-sources`, `--no-perspectives`. **`mcp_server.research()`:** add `effort: str = ""` (optional, default-neutral).
- **Tests:** `test_tiers.py` (registry monotonic; `standard.synthesis_max_tokens==1600`; `resolve_tier`; query/scrape caps via spies; synthesis-budget lock: summary max_tokens==1600 at standard, ==2000 at deep), `test_role_routing.py` (fallback chain; from_env synthesis provider; synthesis uses strong provider when configured), `test_compaction.py` (compacted path builds ordered sources, `<3` falls back to full-text), `test_perspectives.py` (prompt retains matcher substrings, counts unchanged).

### Batch 4 — Verification (one pass) + citation rendering (Spec 1 + Spec 7-verify-role + Spec 6-render)
- **New `sibyl/verifier.py`** (mirrors `analyzer.py`): `@dataclass FindingVerification`; `async verify_findings(findings, pages, provider) -> List[FindingVerification]` — own `litellm.acompletion` (thinking-disabled for deepseek, `json_object`, 3 retries), `context = build_source_context(pages)` (C5 prefix-cache hit off synthesis), align verdicts by `index`, fill omissions via `_lexical_verdict` (ascii token-overlap, pure stdlib), whole-pass lexical fallback on JSON failure. `_extract_citations` = `re.findall(r'\[Source\s+(\d+)\]')`.
  - **Prompt (json_mode, max_tokens=2000):** > SOURCES: {context} FINDINGS TO VERIFY: {numbered}. For each finding check whether the cited [Source N] text supports its specific claim (exact numbers/dates/entities). A finding cites a [Source N] not shown, or unsupported → not supported. Judge only against source text. Return json `{"verdicts":[{"index":<1-based>,"supported":bool,"confidence":"high|medium|low","cited":[<n>],"note":"<=12 words"}]}`.
- **`researcher._do_research` — Step 9b, after `_review_and_refine`:**
  ```python
  if cfg.verify_claims and not cfg.fast and depth >= 2 and report.key_findings and cited_pages:
      from .verifier import verify_findings
      report.finding_verifications = await verify_findings(
          report.key_findings, cited_pages, cfg.get_provider('verify'))
      if cfg.verify_drop_unsupported:
          # drop only LLM-explicit unsupported, floor guard max(3, len//2), re-index verifications
  ```
  Verifies against `cited_pages` (C5) — **never** `report.sources` order.
- **`reporter.py`** (one change reading `finding_verifications` by index, no-op when empty): Key Findings loop appends `[confidence: {conf}]` / ` (unverified)`; rename `## Sources`→`## References` with `> "{supporting_snippet}"` blockquote (fallback `src.snippet`); `generate_pdf` renders the marker + `supporting_snippet` (preferred over `snippet`, plain `pdf.write` path). Index-safe: `v = fv[i-1] if i-1 < len(fv) else None`.
- **`cli.py`:** `--no-verify`, `--drop-unsupported`. **`mcp_server.research()`:** `verify: bool = True`; `_format_report` annotates ` (unverified)`.
- **Tests:** `test_verify.py` (parse verdicts, lexical fallback on non-JSON, no-citation flagged, out-of-range no-crash); extend `_canned_llm` to answer the `FINDINGS TO VERIFY` prompt for the pipeline integration test; `test_reporter.py` (`(unverified)`/`[confidence]` render; empty-`finding_verifications` byte-identical; length-mismatch no IndexError).
- **Re-run eval** — this is the batch that should move citation-grounding most.

### Batch 5 — Reflect loop (Spec 3) · opt-in
- **`researcher._reflect(query, report, pages)`** → `(follow_ups, sufficient)` via `_llm_call(get_provider('general'), …, json_mode=True)`; `_parse_reflection` (garbage → `([], True)` safe-stop).
  - **Prompt:** research-auditor, reads `report.summary[:1500]` + `report.key_findings[:15]`, returns json `{"sufficient":bool,"gaps":[…],"queries":[1-3 standalone search queries]}`.
- **`researcher._do_research`:** loop **between the synth/cross gather and the review block**, gated `depth>=2 and cfg.reflect_rounds>0 and not cfg.fast and good_pages and report.summary`. Per round: reflect → search follow-ups → dedupe into `seen_urls`/`unique_results` → scrape ≤6 new → append `good_pages` (recompute `cited_pages`; if compaction on, re-compact) → extend `search_queries` → re-run `_synthesize` (reassign `report`). Stop on `sufficient`, empty queries, no new sources. Bounded by `reflect_rounds` (clamp ≤2).
- **`cli.py`:** `--reflect-rounds` (default 0).
- **Tests:** `test_reflect.py` (disabled-by-default: `_synthesize` called once, `_reflect` not called; runs-when-enabled; stops on sufficient/no-queries/no-new-sources; bounded to rounds; json_mode used).

---

## 3. Default-behavior matrix (keyless + speed consistent)

| Capability | Default | Fast mode | Depth tiers | Rationale |
|---|---|---|---|---|
| **verify_claims** (mark) | **ON** | **skipped** (gated `not fast`) | depth≥2 only | Headline grounding win; keyless (same DeepSeek), lexical fallback offline; non-destructive → 60 tests green. |
| verify_drop_unsupported | OFF | n/a | n/a | Destructive; needs explicit opt-in + floor guard. |
| **js_render** (thin-only jina) | **ON** | applies (scrape-level) | all | Fires only when bs4 <500 chars (already near-dropped pages); keyless jina, bounded (Sem 2 / 3s / 7s). Pure additive. |
| **dedup** | **ON** | applies | all | Pure-Python, can only remove redundant URLs. |
| **reranker='llm'** | **ON** | applies | all | Replaces existing filter LLM call at same cost; strictly better. |
| **perspectives** | **ON** | applies | all | Prompt-only, zero added calls. |
| **rich_citations** | **ON** | applies | all | Pure-Python snippet, zero latency. |
| **tiers / role-routing** | ON, **behavior-neutral** | unchanged | quick/standard/deep = depth 1/2/3 | standard = byte-identical calls; caps are ceilings equal to today's values; get_provider is a keyless-identical superset. |
| reflect_rounds | **OFF (0)** | skipped | opt-in per run | Full extra search+scrape+synthesis cycle — real cost. |
| compact_sources | **OFF** | applies if on | opt-in (`--compact` bumps to 40) | N extra LLM calls; high-ROI but not free → opt-in. |
| flashrank / strong synthesis+verify providers | OFF | — | env/yaml opt-in | Optional dep / needs keys — violates keyless-by-default if on. |

**Fast mode** = fastest path: skips review, verify, and reflect; keeps free wins (dedup, rerank-in-place, perspectives, rich_citations) + bounded js_render. **Quick tier (depth 1)**: no decompose/verify/review/reflect, tightest caps. **Deep (depth 3)**: larger budgets + existing gap round; reflect still requires explicit `--reflect-rounds`.

---

## 4. Eval harness plan & the number that demonstrates 4–5/5

- **Instrument:** `scripts/eval.py` + committed 20-question gold set (mix of SimpleQA-style single-fact and FRAMES-style multi-hop), graded with the SimpleQA **CORRECT/INCORRECT/NOT_ATTEMPTED** trichotomy by a temp=0 LLM judge; raw reports cached for network-free `--score-only` re-grading. Calls the unchanged public `research()` so it re-runs verbatim after every batch.
- **Workflow:** run at Batch 0 for baseline, then after Batches 2/3/4/5, attributing each delta. README badge = `CORRECT/total` at depth 2.
- **Target demonstrating 4–5/5 research quality** (mid-tier keyless DeepSeek V4-flash, depth 2):
  - **Headline accuracy (CORRECT/total) ≥ 70%** on the mixed set — with **SimpleQA-style factoid ≥ 85%** and **FRAMES-style multi-hop ≥ 60%**.
  - **correct-given-attempted ≥ 85%** and **NOT_ATTEMPTED ≤ 10%** (calibration, not evasion — the verification pass should *flag*, not silently drop).
  - A **measured lift of ≥ +15pp** from the Batch-0 baseline, with the verification batch (4) driving citation-grounding and the retrieval/synthesis batches (2/3) driving multi-hop coverage.
- Peer keyless LDRs publish ~95% SimpleQA; the credibility win is simply publishing a **reproducible** number at all (baseline: zero). Hitting ≥70% headline / ≥85% factoid with a re-runnable harness moves "Evidence of research quality" and "Reproducibility" from 1–2/5 to 4–5/5.

---

## Execution notes
- **Shared-file merge order matters:** land `config.py` + dataclasses + `build_source_context` (Batch 1) before any pipeline batch so downstream edits import stable symbols.
- **`cited_pages` (C5) is the load-bearing invariant** — synthesis context, `report.sources` order, and `verify_findings` all key off it. Any future `_synthesize` change must preserve it or citations silently drift.
- **`_canned_llm` fixture** gets exactly two edits total (relevance→JSON scores in Batch 2; add verify-prompt branch in Batch 4); reflect/compaction stay on the prose fallback until their own tests.
- After every batch: `.venv/bin/python -m unittest discover tests` must stay green, then re-run `scripts/eval.py --score-only` (or a fresh live run) to record the delta.

Key files (all absolute): `/Users/yichen/projects/sibyl/sibyl/config.py`, `/Users/yichen/projects/sibyl/sibyl/researcher.py`, `/Users/yichen/projects/sibyl/sibyl/scraper.py`, `/Users/yichen/projects/sibyl/sibyl/reporter.py`, `/Users/yichen/projects/sibyl/sibyl/cli.py`, `/Users/yichen/projects/sibyl/sibyl/mcp_server.py`; new: `/Users/yichen/projects/sibyl/sibyl/verifier.py`, `/Users/yichen/projects/sibyl/sibyl/dedup.py`, `/Users/yichen/projects/sibyl/scripts/eval.py`, `/Users/yichen/projects/sibyl/evals/gold/simpleqa_frames_20.jsonl`.
