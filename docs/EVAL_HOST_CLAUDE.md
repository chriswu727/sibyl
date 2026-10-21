# Historical host-model experiment

This exploratory 30-question SimpleQA note predates the current persistent
research workflow. It records a historical comparison, not a release gate or
a current-model product benchmark. The [current evaluation plan](QUALITY_BUILD_PLAN.md)
requires matched models, source access, and budgets.

## Recorded outcomes

| Historical configuration | Recorded outcome | Interpretation |
|---|---:|---|
| Configured DeepSeek-flash pipeline | 5/30 correct | Different model and workflow from the host condition |
| Claude over pure `gather_sources`, concurrent load | 26/30 correct, 2 incorrect, 2 abstentions | Observed result recorded in this experiment |
| Pure retrieval with two abstentions replaced by successful later lookups | 28/30, approximately 93% | Post-hoc projection, not a fresh complete run |
| Host workflow with WebFetch fallback | 28/30 correct | Different source access; not pure Sibyl retrieval |

The questions were a fixed seed-42 sample of the SimpleQA test set. Host agents
were instructed to answer from retrieved text and abstain when evidence was
missing. Other model agents graded answers against the reference answers.
Agent grading and a small sample introduce uncertainty; the experiment does
not isolate the effect of Sibyl from the effect of the model or source access.

## Retrieval observations

Concurrent requests coincided with search throttling and inconsistent source
availability. Two abstained questions produced their expected answer text during
later sequential retrieval. That establishes recoverability for those lookups,
not a measured 93% sequential-workflow success rate. Timing and public source
state also changed between attempts.

The two incorrect responses illustrate distinct risks:

- An exact completion day was inferred from nearby source text that supplied
  only a month/year. Source proximity did not establish support for that day.
- A product date conflicted with the benchmark reference. Such a disagreement
  needs explicit source/date reconciliation; a source-backed answer is not
  automatically correct, and the benchmark is not automatically wrong.

The original note classified no answers as memory-only fabrications. That
classification is not a guarantee of hallucination resistance: unsupported
inferences can occur even when a report cites a retrieved source.

Historical retrieval changes included query-relevant text windows, fuller
Wikipedia extraction, filtering thin news stubs, and fallback discovery. These
are implementation observations, not evidence that the current system should
prefer encyclopedias over original papers or official sources.

## What this experiment supports

It motivated separating evidence retrieval from the model that reasons over it.
It does not prove that host reasoning always outperforms autonomous research,
that a particular model remains preferable, or that the new Deep Research
workflow outperforms ordinary search.

For a new comparison, retain full traces and outputs, hold the host/model,
retrieval access, concurrency, and budget constant, and predeclare report-level
metrics. Repeat across representative research questions, including counterevidence
and missing information. Human-check a sample of claim-to-passage support.
Model/subscription-backed runs require explicit operator approval.

The [original detailed note and per-question table](https://github.com/chriswu727/sibyl/blob/d343267/docs/EVAL_HOST_CLAUDE.md)
remain available in Git history. Historical claims there are superseded by the
interpretation above. Formal retained retrieval artifacts and release gates are
listed in [Launch readiness](LAUNCH_READINESS.md).
