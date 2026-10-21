# Example: does SQLite WAL eliminate write contention?

![Diagram of the two-round offline MCP replay](../../../assets/research-replay.svg)

This example demonstrates a complete host-assisted investigation, not a model
benchmark. Read the [exported report](report.md) and [recorded source excerpts](sources.json).
The original pages were read through Sibyl on 2026-10-21. Excerpts are selected
and whitespace-normalized; they are not full-page snapshots. Both pages are from
SQLite's documentation, so two URLs do not imply two independent publishers.

## Investigation

1. Start one persistent question with a three-attempt budget.
2. Read the official [WAL documentation](https://sqlite.org/wal.html). It describes
   reader/writer concurrency and notes that busy errors can still occur.
3. Record an unresolved review: a reader/writer benefit does not answer whether
   writers can proceed concurrently. Add a focused follow-up question.
4. Read the official [isolation documentation](https://sqlite.org/isolation.html)
   as counterevidence to the stronger "no write contention" claim. It explicitly
   states that writers take turns. The round records the earlier gaps and revision.
5. Reassess both passages, inspect the current citation numbering, and finalize
   the narrower conclusion: WAL does not eliminate writer serialization.
6. Export Markdown/JSON, optionally PDF. Restart the MCP process and inspect the
   same immutable report and evidence hashes.

The reasoning, query choice, and report prose in this replay are hand-authored.
Sibyl enforces workflow structure, evidence references, revisions, budgets, and
export boundaries; it does not independently prove semantic support. Direct
reads still carry limited-coverage diagnostics after host completion.

## Reproduce offline

From an installed source checkout, run:

```bash
python scripts/check_research_workflow.py
```

The check uses a real MCP stdio client/server pair and a test-only fetch adapter
serving the committed excerpts. Unexpected HTTP requests fail. It calls no model
and requires neither a provider key nor network access. Temporary checkpoints
and exports are removed on exit. To retain outputs in a **new** directory:

```bash
python scripts/check_research_workflow.py --output /tmp/sibyl-wal-example
```

Install the report extra and append `--pdf` to verify PDF export as well.
The script checks stale-revision rejection, gap tracking, mandatory reassessment,
completed-report immutability, restart recovery, exact JSON export, and passage
hashes. CI runs the replay against both source and clean installed-wheel profiles.

Report timestamps in a replay are replay times, not fresh source-access dates.
The committed report is one actual replay output; timestamps, job IDs, and export
filenames vary between runs. No latency, accuracy, independent corroboration,
or production-reliability claim follows from this example.
