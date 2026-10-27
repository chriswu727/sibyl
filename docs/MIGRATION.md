# Host-only migration and legacy boundary

Audited 2026-10-26 against host-only main after
[`e416529`](https://github.com/chriswu727/sibyl/commit/e416529).
This is a source/design audit, not a new live quality result or package release.
The [architecture](MCP_ARCHITECTURE.md) describes current execution; the
[functional goals](REFACTOR_GOALS.md) and [plan](QUALITY_BUILD_PLAN.md) define
the complete-product target and remaining work.

## What was removed

- Internal generative-model execution, LiteLLM/provider routing, hidden model
  judging and model configuration. Sibyl never adopts host subscription tokens.
- Autonomous research execution and its resume path. New jobs accept host mode
  only; saved autonomous jobs cannot continue, be assessed or be finalized anew.
- The `research`, `analyze`, `compare`, `swot`, `timeline`, finance and chart tools,
  and the old model-backed CLI/implicit question command.
- The old `report`, `finance` and `full` MCP profiles, and (2026-10-27, before the
  first stable release) the `keyless` compatibility profile with its text
  responses and its `gather_evidence`, `gather_sources`, `quick_search` and
  `read_url` tools. Sibyl serves one eight-tool set; `--profile keyless` exits
  with an error instead of silently changing the tool list. Hosts that need web
  search should use their own and read results with `continue_research(url=...)`;
  `gather_bundle` remains the keyless fallback.
- The short-lived evidence loop (`gather_evidence`, EvidenceLoop 1.0 types and its
  offline evaluation). Persistent jobs replace it.
- Contract-1 prose completion. Old contract-1 checkpoints get `legacy_contract`
  from `complete_research`; already completed ones stay readable and exportable.
  `save_report` now requires `job_id`; the `last_report` fallback and the
  `format="both"` option were removed.
- Model and finance dependencies/extras. The `report` extra is PDF rendering only;
  Markdown and JSON are core capabilities.
- The unreferenced `context.py` single-window helper. Active retrieval and direct
  reads use `passages.select_passages`; tail, body and multilingual selection
  regressions exercise that production path instead.

The retrieval CLI, source bundles, optional local reranker and optional
rendering services are not internal researcher models.
They remain explicit retrieval capabilities, with separate dependency/network
disclosures. The persistent host-led job is the primary agent workflow.

## What intentionally remains

| Surface | Reason and boundary |
|---|---|
| `ResearchMode` includes `autonomous` | Deserialize old records; creation/continuation guards reject execution |
| Notebook model/token/reflection counters | Preserve old checkpoints and budget history; no current model scheduler consumes them |
| Stored `depth` / `language` job fields | Preserve checkpoint shape; new MCP jobs do not expose model-effort or internal synthesis controls |
| Old report fields such as `predictions`, `cross_analysis`, `model_used`, `charts`, `market_data_summary` | Keep historical JSON load/export compatibility; no removed producer is restored |
| `FindingVerification` | Preserve recorded historical verdicts; rendering identifies them as historical, not fresh validation |
| Claim contract 1 checkpoints | Readable and exportable when already completed; no longer finalizable |
| Python retrieval exports (`sibyl.gather_bundle`, `sibyl.gather_sources`) | Library API for pipelines, not an MCP profile |
| Dated changelog, experiments, evaluation JSON and original-language evidence | Preserve provenance and failures; historical feature descriptions are not current product promises |

Do not delete old data fields just to eliminate a word from a scan. Removing them
requires an explicit storage/schema migration, not an incidental cleanup.
Historical source content and recorded results are not rewritten to make current
documentation look cleaner. Old model verdicts remain fallible recorded judgments.

## Upgrade safely

1. Install the current source in a fresh Python 3.11+ virtual environment, as in
   [installation](../llms-install.md). Package upgrades do not uninstall obsolete
   transitive dependencies from an old environment.
2. Use the default agent profile with a structured-content stdio host; use
   keyless only for the documented compatibility surface. Configure absolute
   executable and private output-root paths.
3. Remove old model fields from `SIBYL_CONFIG`. It accepts only `search_engine`,
   `extractor`, `jina_fallback` and `js_render` for compatibility reads/search.
   Unsupported fields fail without echoing their values. Model environment
   variables do not enable model execution.
4. Keep the old checkpoint directory private and stop its owner before backup.
   The default output root is now the per-user data directory, not `./reports`.
   Old stores are not moved or merged; set `SIBYL_OUTPUT_ROOT` to an old store's
   absolute path to keep using it. Stop older runtimes before newer ones use the
   same directory.
   Existing reports remain readable/exportable. To extend a historical autonomous
   investigation, explicitly create a new host job rather than converting it.
5. Install only needed extras. Optional Tavily/Jina calls have their own disclosure
   and billing boundaries; host subscription access does not pay for them.

## Documentation and publication boundary

English is canonical; `README.zh-CN.md` is the sole maintained Chinese document.
Current docs describe host-only source. Published 0.4.0b1 and 0.3.0 are older
architectures, and the unchanged development version string does not identify
this source. Include the commit when reporting results. No historical beta
exception authorizes a new release or reuse of an uploaded version.

Current instructions no longer require finance extras or internal-model audits.
The [stable gates](LAUNCH_READINESS.md#release-gates) still require reliable live
retrieval, real host research quality and the declared paired comparison. Passing
the removal/dependency guards and offline replay proves an architecture boundary,
not better answers than built-in search or zero hallucinations.
