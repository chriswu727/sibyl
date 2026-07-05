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

## gather_sources optimizations (applied + verified)

The agents' retrieval feedback drove four fixes, each verified on the cases it targeted:
1. **Truncation** (right page, cut before the fact) → scrape deep (30k) +
   `relevant_window` returns the query-densest slice, not the head. Fixed real17
   (Terraria patch, in a tail History table).
2. **Infobox / tail sections not captured** → `fetch_wikipedia_extract` pulls the
   full clean article text via the Wikipedia API. Fixed real02, real07, real18
   (facts in the infobox / deep sections).
3. **Google-News RSS title-only stubs** polluted results → filtered (snippet>120).
4. **"No sources found"** (engines rate-limited or brittle to short queries, even
   for entities with Wikipedia pages) → `wikipedia_lookup` opensearch fallback when
   web search is thin. Fixed real03, real12.

Re-measured **pure** gather_sources (WebFetch fallback forbidden): **26/30 = 87%**,
up from ~24/30, with all four prior fallback cases now gather-native. With fix #4
(real03 + real12 verified to now surface their answers) pure-gather projects to
~28/30 ≈ the fallback-allowed number — i.e. the tool is now largely self-sufficient.
Remaining: real20 (Seiko 1967-vs-1968, a source-backed date dispute, not a
fabrication) and genuinely non-encyclopedic facts.

## Reproduce

`scripts/gather.py "<query>"` (keyless) + your own reasoning; or connect
`sibyl-mcp` in a Claude client and use the `gather_sources` tool.
