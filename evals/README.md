# Evaluation records of the retired research server

Everything in this directory evaluated Sibyl's earlier research MCP server, not
the current skill and checker. The records are kept unchanged, including every
failed run. The code, scripts and protocols that produced them are in the
[last commit of the server](https://github.com/chriswu727/sibyl/tree/5e2537088992300b8e91b709a024c5da283760f3),
whose [`docs/LAUNCH_READINESS.md`](https://github.com/chriswu727/sibyl/blob/5e2537088992300b8e91b709a024c5da283760f3/docs/LAUNCH_READINESS.md)
states each gate's frozen protocol and outcome.

## Paired host evaluation

The release gate that decided the rewrite. One host model ran 20 tasks (six
factual, four with a false premise, ten open research questions) under three
conditions: built-in web search and fetch, the same plus Sibyl's server, and the
same plus a multi-agent deep-research skill. A blind judge, a larger model of the
same family, checked each claim–URL pair against the cited page's text. The
criteria, frozen
before the first run, asked Sibyl to beat built-in search by 5 points of citation
support without answering fewer factual or trap tasks correctly or scoring more
than 0.2 lower on any rubric dimension.

| Run | Citation support: built-in / Sibyl / deep-research | Outcome |
|---|---|---|
| [2026-10-27](results/paired-host-2026-10-27/manifest.json) | 69.1% / 78.6% / 71.6% | Failed on counterevidence (rubric 1.1 vs 1.4) |
| [2026-10-28](results/paired-host-rerun-2026-10-28/manifest.json) | 80.5% / 80.2% / 73.1% | Failed on citation support and one factual task |

Built-in search alone moved 11 points between runs, so single runs of 20 tasks
could not resolve the differences. Pooled, Sibyl's citation support was 79.4%
against 74.9% for built-in search, with equal factual and trap results, at about
twice the cost. The owner stopped further comparisons and narrowed Sibyl to the
part an agent cannot do for itself: checking its quotes deterministically.

## Other records

- `results/host-source-retrieval-*.json`, `results/live-retrieval-*.json`:
  retrieval gates over the cases in `host_source_cases.jsonl` and `gold/`.
- `results/host-*`, `results/body-evidence-*`, `results/source-discovery-*`,
  `results/query-identifiers-*`: development experiments on the server's search,
  selection and host workflow.
- `results/runtime-soak-*`: resource-recovery soak of the server runtime.
