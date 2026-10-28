# Ledger format and checks

A ledger is one UTF-8 JSON file of at most 1 MiB, written by the agent. Unknown
fields are rejected so that a misspelled field is not silently ignored.
`sibyl template` prints a complete example.

## Ledger

| Field | Required | Meaning |
|---|---|---|
| `ledger` | no | `"sibyl-ledger/1"`, the format version. |
| `question` | yes | The research question. |
| `claims` | yes | 1–200 claims, below. |
| `counterevidence` | no | `searched_for` (required) and `found`: the search for evidence against the main conclusion. A check without one adds a note. |
| `gaps` | no | Up to 50 questions the research could not answer. |

## Claims

| Field | Meaning |
|---|---|
| `id` | Unique; letters, digits, `_`, `.` and `-`. Used by `based_on` and in check results; reports omit it. |
| `text` | The claim as it appears in the answer. |
| `kind` | `fact` (default) or `inference`. |
| `status` | `supported` (default), `disputed` or `unresolved`. |
| `evidence` | Facts only: up to 20 items of `url`, `quote` and `role` (`supports`, the default, or `challenges`). |
| `based_on`, `rationale` | Inferences only: the claim ids the conclusion rests on and why it follows. |
| `note` | Required when `unresolved`: what was searched and what is missing. |

A supported fact needs a supporting quote. A disputed fact needs a supporting
and a challenging quote. An inference has no quotes, cannot be disputed (record
the disputed fact it rests on), and its `based_on` may not form a cycle. An
unresolved claim needs only its note; any quotes it lists are still checked.

## Checks

`sibyl check` reads each distinct cited URL once, over a client that accepts only
public HTTP(S) destinations, re-validates every redirect and caps responses at
2 MiB. HTML is reduced to its main text; text PDFs are read up to 40 pages or
200,000 characters with `[pdf]` installed on POSIX.

A quote is found when its normalized text occurs in the page's normalized text.
Normalization applies Unicode NFKC, case folding, straight quotation marks and
hyphens for their typographic forms, removal of soft hyphens and zero-width
characters, and collapses all whitespace. `...`, `…` or `[...]` splits a quote
into parts that must occur in order.

| Issue | Severity | Raised when |
|---|---|---|
| `quote_not_found` | blocking | A quote does not occur in the text read from its page. |
| `source_unreadable` | blocking | A cited page could not be read or had no text. |
| `number_not_quoted` | blocking | A number in a fact (thousands separators ignored) is in none of its supporting quotes. |
| `basis_has_issues` | blocking | An inference rests, directly or through other inferences, on a claim with a blocking issue. |
| `single_source` | warning | All of a fact's supporting quotes come from one host name (ignoring `www.`). |
| `name_not_in_sources` | warning | A capitalized name in a fact appears in no supporting quote, page title or the 1,000 characters around a found quote. |
| `possible_conflict` | warning | Another stated fact shares most of its subject words but states different numbers. |
| `short_quote` | warning | A quote is shorter than 20 normalized characters. |

The check passes when no claim has a blocking issue. Warnings are listed in the
report's check notes.

## Check result

`NAME.check.json` (format `sibyl-check/1`) holds `ledger_sha256` (the exact ledger
bytes it checked), `checked_at`, `sibyl_version`, `passed`, every source's
`title`, `source_type`, `text_sha256`, `chars`, `error` and extraction
`warnings`, every claim's `blocking` and `warnings` with `details`
(`unquoted_numbers`, `unmatched_names`, `conflicts_with`), every quote's `found`
and `offset` in the normalized page text, and ledger-level `notes`.
