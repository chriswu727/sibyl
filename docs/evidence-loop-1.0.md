# Evidence loop 1.0

`gather_evidence` is Sibyl's bounded workflow for questions that require more
than one focused retrieval. It uses ordinary MCP tool calls and does not use MCP
sampling or a hidden Sibyl model. The calling host proposes atomic queries and
performs the final synthesis.

## Start

Call the tool with an original question:

```json
{"question": "In what year was the company that created CUDA founded?", "max_steps": 3}
```

An atomic question is retrieved immediately. A high-confidence dependent fact
chain starts with no retrieval and returns `next_action: "decompose_query"`.

## Continue

Use the returned `loop_id` and one new atomic query:

```json
{"loop_id": "el_<id>", "query": "Which company created CUDA?"}
```

Each response contains compact summaries for every historical step and one
full `current_step.bundle` for the new retrieval. This avoids returning the same
web passages repeatedly. Follow the bundle diagnostics and the loop's
`next_action`; a repeated or still-dependent query does not consume a retrieval
step.

## Finish

When the retrieved evidence covers the original question, identify the steps
that support the synthesis:

```json
{
  "loop_id": "el_<id>",
  "finish": true,
  "supporting_step_ids": ["E1", "E2"]
}
```

Sibyl accepts only step IDs whose SourceBundle has
`recommended_action: "synthesize"`. The host may synthesize only after the loop
returns `status: "ready"` and `next_action: "synthesize"`.

## Bounds and lifecycle

- `max_steps` defaults to 3 and must be between 1 and 4.
- Loops expire ten minutes after their most recent call.
- One process retains at most 64 loops in memory and never writes them to disk.
- A server restart invalidates every loop ID.
- Search-provider usage is charged per actual retrieval step. A SourceBundle
  can itself make one bounded exclusion refinement, so provider request count
  can exceed the number of loop steps.
- `budget_exhausted` and `failed` are not synthesis-ready states.

The host must retain the full bundle returned for each current step, including
its citation IDs. Historical summaries are an audit index, not a replacement
for the evidence passages.
