# Evidence linked research results

Status: source preview, unreleased. New MCP jobs use `claim_contract_version=2`.
The host authors atomic claims through `assess_research`; Sibyl checks references
and reasoning dependencies, preserves unknowns, and renders optional reports from
the same ledger. It does not independently establish semantic support or truth.

## Record facts and reasoning

Read exact passages with `get_research(job_id, passage_id=...)`, then submit the
complete claim set for one question. Assessment replaces that question's claims;
use stable IDs such as `Q1.C1`. Up to twelve material claims are allowed per question.
Claim IDs can be referenced across questions in the same job.

```json
{
  "job_id": "the saved job ID",
  "revision": 7,
  "question_id": "Q1",
  "status": "supported",
  "conclusion": "This option fits the specified concurrency need under explicit conditions.",
  "counterevidence_note": "Read the documented limitations; actual workload testing is still needed.",
  "claims": [
    {
      "claim_id": "Q1.C1",
      "kind": "fact",
      "status": "supported",
      "text": "The inspected documentation states the relevant capability and conditions.",
      "supporting_passage_ids": ["the exact passage citation_id"],
      "quotes": [
        {
          "passage_id": "the exact passage citation_id",
          "text": "Exact text copied from that passage."
        }
      ]
    },
    {
      "claim_id": "Q1.C2",
      "kind": "recommendation",
      "status": "supported",
      "text": "Consider this option when those documented conditions fit the workload.",
      "basis_claim_ids": ["Q1.C1"],
      "rationale": "The documented capability meets the user's stated concurrency need.",
      "assumptions": ["The actual workload has not been benchmarked."],
      "criteria": [
        {
          "name": "Concurrency needs",
          "preference": "The capability requested by the user",
          "origin": "user"
        }
      ]
    }
  ]
}
```

This is an argument example, not factual evidence about a real product. Replace
the prose, revision and IDs with inspected evidence and the current checkpoint.
The question-level conclusion is a host review note; report content comes from
the individual claims, not from additional untracked report prose.
Raw question-review notes remain in the JSON audit ledger; new human-readable
report narratives render claims rather than those free-form review notes.

| Field | Contract |
|---|---|
| `claim_id` | Question-prefixed ID, for example `Q0.C1` or `Q3.C2` |
| `kind` | `fact`, `inference`, or `recommendation` |
| `status` | Host-declared `supported`, `disputed`, or `unresolved` |
| `text` | One material assertion, up to 1,000 characters |
| `supporting_passage_ids`, `challenging_passage_ids` | Stable exact-passage IDs, up to twelve each |
| `basis_claim_ids` | Up to twelve existing claims supplying reasoning premises |
| `rationale` | Up to 1,000 characters; required for reasoning and unresolved claims |
| `assumptions` | Up to four explicit conditions, each at most 300 characters |
| `criteria` | Up to six recommendation criteria: name, preference and origin |
| `quotes` | Up to twelve `{passage_id, text}` excerpts, each at most 500 characters, from passages this claim cites |

Supported facts require passage references. Facts cannot substitute another
claim for original evidence.

### Verbatim quotes

Supported and disputed facts must quote exact text from at least one supporting
passage; disputed facts also quote a challenging passage. Each quote must come
from a passage the claim cites. Sibyl searches the stored passage text for the
quote, ignoring case, fullwidth ASCII forms, composed versus combining accents,
typographic quote/dash variants, whitespace runs, leading/trailing ellipses and
one surrounding pair of quotation marks. Superscripts, Roman numerals and other
compatibility characters are not folded, so `10²` never matches `102`. A match
must cover whole source characters. A quote shorter than eight characters after
folding, or one not found, rejects the whole assessment with
`field="claims.quotes"` and the claim ID. Stored quotes are replaced by the
matched source text with whitespace runs collapsed to single spaces, so a stored
claim can be resubmitted unchanged. JSON report claims keep that text unescaped
and add `start_char`/`end_char` offsets into the extracted source; the stored
passage itself is never modified. Markdown reports place each quote mid-line and
backslash-escape inline Markdown (emphasis, code, links, images, HTML and
entities), so quoted source text cannot add links, images or structure. Bare
URLs inside a quote may still be auto-linked by some Markdown renderers. Markdown, PDF and text reports print each
quote beside its claim.

This catches a citation to the wrong passage or an invented excerpt. It does not
establish entailment, source truth or freedom from hallucination: the host must still quote text that carries every asserted
qualification, split compound claims or mark them unresolved. The rule applies
to new assessments; previously stored claims are not rewritten or re-validated. Supported/disputed inferences and recommendations
need a reasoning basis and rationale. A recommendation also requires criteria;
`origin` is `user` or `host_assumption`, declared by the host. Criteria and origin
are not independently verified against the conversation.

A disputed claim needs a supporting basis and challenging passages. A supported
reasoning claim requires supported premises throughout its dependency chain;
an unresolved premise cannot be upgraded into a supported recommendation.
Changed premises in an existing ledger make dependent reviews stale. One passage
cannot both support and challenge the same claim, but can play different roles
for different claims. New unsupported precision must not be added to fill a field.
An unknown price, for example, is a fact with `status="unresolved"`, text stating
that it is unknown, and a rationale explaining the missing evidence.

Support/challenge links, including transitive reasoning premises, feed the existing
question review and pinned source window. Unknown references, cycles, duplicate
IDs and removal of a premise still referenced by another question fail atomically.
The twelve-source review window remains bounded; narrow evidence if it overflows.

## Follow up and reassess

The brief exposes claims under `research_plan.questions[].review.claims`, alongside
gaps, question conclusions, suggested queries and current revision. It omits raw
passage text; read that text on demand. No report is required to consume the ledger.

New retrieval makes the affected question's assessment stale. Recommendations or
inferences depending on that question also need review. Changing a premise's
content or status makes its dependent assessments stale, even without another
search. Reassess the question with its complete updated claim set and the current
job revision. The host decides whether the new evidence actually changes support.

Unresolved/disputed claims and unresolved premises prevent a complete readiness
verdict. Host-declared support is not a machine-verification result. Existing
counterevidence and substantive-source gates still apply; adding claim records
does not waive missing investigation.

## Optional reports

For contract-2 jobs, finalize using the current revision and optional claim IDs:

```text
complete_research(job_id="...", revision=the_current_revision,
                  claim_ids=["Q1.C2"], limitations=["A material scope limit."])
save_report(job_id="...", format="md")
```

Omitting `claim_ids` selects all claims, with a maximum of 32 selected claims.
The report includes their transitive premises and all unresolved/disputed claims,
even if those claims were not selected. Summary/findings and analysis are rendered
from the host's claim text, criteria, reasoning, assumptions and current citations.
For new jobs omit `summary`, `key_findings` and `analysis`; submit that content as
claims before finalizing instead. Report rendering introduces no model call.

When gaps remain, `allow_incomplete=true` produces a partial report with those gaps
retained. An explicit unresolved finding can be exported without any sources;
it receives no invented citation. Finalization makes the report immutable.
Its status does not establish factual correctness.

JSON report schema 1.1 adds `claims` with `question_id` and `review_current`.
Markdown/PDF/plain text retain the reasoning and qualifications in the generated
analysis under one Evidence-linked conclusions section, without repeating each
statement in three human-facing sections. JSON keeps the compatible summary,
findings and analysis fields and the full claim ledger. PDF font/render failures
must not silently omit conclusions or criteria; a failed format has no confirmed
artifact path. `verification_scope="citation_targets_only"` remains unchanged: reference
checks do not prove entailment, source accuracy or that the host read carefully.
Stale dependency warnings are preserved when exporting a partial report.

## Compatibility and verification

Old checkpoints without the new contract version load as contract 1 and retain
their existing prose completion interface through the keyless profile; the agent
profile's `complete_research` takes no prose fields and returns `legacy_contract`. They are not silently upgraded or
assigned new verification results. Direct Python `ResearchJobs.start` retains
contract 1 by default for embedding compatibility; pass `claim_contract_version=2`
to use the same contract as new MCP jobs. Both MCP profiles create contract-2 jobs.
No tool or profile was added, and no old checkpoint is deleted.

The [recorded SQLite workflow](examples/sqlite-wal/README.md) runs through real
MCP stdio with claim-linked facts and a conditional recommendation, report export,
revision protection and process restart. Its sources are fixed excerpts, not a
live source-discovery or semantic-accuracy benchmark. Controlled tests also cover
unknown/disputed outputs, cross-question dependency changes and input limits.
Real-host usability and a paired comparison with native search remain separate
quality work; see the [implementation plan](QUALITY_BUILD_PLAN.md).

The [anonymous source-discovery smoke record](../evals/results/source-discovery-claims-2026-10-26.json)
retains the observed original-source results and a PostgreSQL version mismatch.
Its retrieval-ready statuses are not semantic answer judgments or release evidence.
