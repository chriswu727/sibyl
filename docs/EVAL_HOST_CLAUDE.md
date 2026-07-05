# Host-model-over-gather_sources eval (validates the MCP retrieval-provider design)

The question: in MCP mode, should sibyl let the HOST model (e.g. Claude) reason
over sibyl's retrieval, instead of sibyl synthesizing with its own mid-tier model?

Experiment: 30 hard SimpleQA questions (official human-verified gold). Each was
researched by a Claude agent that could ONLY call sibyl's keyless `gather_sources`
(search + scrape + dedup, no synthesis) and had to answer strictly from the
retrieved text — abstaining, never guessing from memory. Graded by independent
agents on the SimpleQA CORRECT/INCORRECT/NOT_ATTEMPTED rubric.

## Result

| | sibyl's own DeepSeek-flash pipeline | Claude over gather_sources |
|---|---|---|
| Correct | 5/30 = **17%** | 28/30 = **93%** |
| Fabricated-wrong | 13 | **1** |
| Abstained | 12 | 1 |

Swapping only the *reasoner* over the *same keyless retrieval* moved the score
from 17% to 93% and collapsed fabrication from 13 to 1. That isolates the
variable: **sibyl's ceiling was the model consuming the sources, not the sources.**
The MCP retrieval-provider design (host model is the brain, keyless) is validated.

Honest caveat: 4 of the 28 correct used a direct WebFetch/Crossref fallback when
gather_sources came up short, so a pure-gather run is ~24/30 ≈ 80% — still ~4.7×
the baseline.

## gather_sources optimizations (applied)

The agents' retrieval feedback surfaced, ranked by ROI:
1. **Truncation** — the right page is retrieved but cut before the fact. Fixed:
   scrape deep (30k) + `relevant_window` returns the query-densest slice, not the head.
2. **JS-loaded/collapsed sections** (tail History tables, infoboxes) — NOT captured
   by keyless scraping even when the page is found (e.g. a game wiki's version table).
   Remaining limit; follow-up = JS-render those pages or use the Wikipedia API.
3. **Google-News RSS title-only stubs** polluted results — filtered (snippet>120).
4. Thin encyclopedic reach on niche/foreign facts — follow-up.

## Reproduce

`scripts/gather.py "<query>"` (keyless) + your own reasoning; or connect
`sibyl-mcp` in a Claude client and use the `gather_sources` tool.
