# MCP LLM-backend investigation (sampling vs retrieval-provider)

Verdict: do NOT build on MCP sampling; reshape sibyl-as-MCP into a retrieval/context provider so the host model (Claude) is the brain.

# SIBYL LLM-backend architecture — final investigation report

## 1. VERDICT: NO (do not build sibyl's reasoning path on MCP sampling)

MCP sampling is **not** a viable engine for a research agent that makes many LLM calls per run. Three independent, each-sufficient blockers — all three verified against primary sources:

- **The Claude hosts you care about don't implement the client side.** Claude Code is a confirmed ❌ in the apify/mcp-client-capabilities matrix, corroborated by claude-code issue [#1785](https://github.com/anthropics/claude-code/issues/1785), open since 2025-06-08, assigned but with no ship commitment. Claude.ai is a confirmed ❌ in the same matrix. **Claude Desktop is an inference, not a verified fact** — Desktop is *not a row in the apify matrix at all*, so its ❌ is extrapolated from Claude.ai plus secondary "not yet supported" write-ups (and there exists at least one mid-2025 post titled "MCP Sampling with Claude Desktop" muddying the picture). The inference is low-risk and almost certainly correct, but treat it as inference. Net: inside the exact hosts where you wanted "keyless frontier Claude," a `sampling/createMessage` request is not serviced, and you fall back to your own DeepSeek key every time. The headline benefit does not exist today. Evidence: apify/mcp-client-capabilities (community matrix), github.com/anthropics/claude-code/issues/1785.

- **The feature is deprecated in the upcoming draft spec — not yet in the live governing spec.** The current *stable* protocol is **2025-11-25, which does not deprecate sampling.** The deprecation lands in the **draft/RC revision `2026-07-28` (SEP-2577)**, which as of today (2026-07-03) is unreleased. That draft says verbatim: *"New implementations SHOULD NOT adopt it; existing implementations SHOULD migrate to integrating directly with LLM provider APIs"* — with a ≥12-month grace before removal-eligibility. So the direction of travel is unambiguous, but state it precisely: sampling is **on a deprecation track in the next revision**, not deprecated in the spec that governs clients right now. Relatedly, the Python SDK's `@deprecated` decorator on `create_message` is **on the `main` branch / upcoming release**, not confirmed in any pinned or released version — verify against your actual installed `mcp` version before citing it. Evidence: modelcontextprotocol spec draft 2026-07-28 / SEP-2577.

- **Even where it works, the economics, consent flow, and rate limits are wrong for an autonomous agent.** Sampling runs on the *user's own subscription/quota* (keyless ≠ free), and the spec mandates human-in-the-loop with **no spec-level auto-approve**. Plain sampling is up to two approvals (prompt + response) per call — and that's the floor: the spec's own sequence diagram shows **tool-enabled sampling adds more gates** (approve request, approve tool calls, approve continuation, approve final response, per loop iteration). Sibyl's 5-stage pipeline, fanned out, means dozens of popups per run. Two further operational blockers the original draft understated: (a) the spec lists **"Clients SHOULD implement rate limiting"** as Security Consideration #4 — for an agent firing dozens of calls per run, client-imposed rate limiting is an independent failure mode; (b) the reason clients gate every call and refuse auto-approve is the documented **quota-drain / sampling-abuse attack vector** (Unit42, practical-devsecops) — a malicious or runaway server draining the user's paid quota. That abuse risk *reinforces* the approval-spam blocker rather than being a footnote. And `modelPreferences` hints are advisory — you cannot even guarantee Claude does the work.

The one thing sampling had going for it — "let the host's frontier model reason, keyless" — is precisely what the next spec revision is steering everyone away from.

## 2. CLIENT SUPPORT TRUTH: thinly implemented, and not in the Claude hosts

Only **one** support claim rests on a first-party primary source. Everything else rests on a single community matrix that **contradicts itself on re-read** (when fetched directly, its summary placed OpenAI Codex under ❌ — the opposite of what a column-parse suggested). So the roster below is split by evidence quality, not asserted flat.

| Client | Sampling support | Evidence quality |
|---|---|---|
| **VS Code (GitHub Copilot)** | ✅ | First-party primary doc (code.visualstudio.com) |
| Claude Code | ❌ | Verified — apify matrix + open issue #1785 |
| Claude.ai | ❌ | Verified — apify matrix |
| Claude Desktop | ❌ (inferred) | **Not in the matrix; inference only** |
| Cursor, Cline, Windsurf, Zed, Gemini CLI, Goose | ❌ | Community matrix, unverified |
| OpenAI Codex, JetBrains AI Assistant, Postman, Glama, Le Chat, AmpCode | listed ✅, **unverified** | Single community matrix that contradicts itself on re-read (Codex read as ❌ on direct fetch) — do not rely on |

The load-bearing facts: **the only mainstream host where sampling is confirmed to light up via a primary source is VS Code / Copilot** — a client where the reasoning model is *not* Claude-guaranteed and where sibyl's benefit is marginal. The Claude hosts are ❌ (two verified, Desktop inferred). Any aggregator blog claiming "Claude Desktop supports sampling" is citing stale, contradicted info. Evidence: apify/mcp-client-capabilities; code.visualstudio.com/docs (MCP sampling).

## 3. Not viable → skip the full sampling implementation

Because the verdict is NO, this is not a sampling-first backend. But a small, capability-gated hedge is cheap, so here is the scoped version.

Keep **litellm + user key as the real engine in BOTH CLI and MCP modes.** That is the spec's own migration recommendation and it works everywhere today. Do **not** hard-cut DeepSeek.

If you want the VS-Code-only upside, add an `LLMBackend` abstraction with two implementations behind one interface (`generate(messages, system, max_tokens, model_prefs) -> str`):

- `LiteLLMBackend` — current DeepSeek path, the **default**.
- `SamplingBackend` — calls `ctx.session.create_message(...)`, gated strictly by runtime detection via `ctx.session.check_client_capability(ClientCapabilities(sampling=SamplingCapability()))`. Only select it when that returns True. Wrap every call in `try/except (McpError, NoBackChannelError)` **and** `asyncio.wait_for(timeout)` because it blocks on human approval; on any failure/denial (`code: -1`), fall through to `LiteLLMBackend`. Convert prompts via `SamplingMessage(role='user', content=TextContent(type='text', text=...))`, pass your prompt through `system_prompt=`, set `include_context='none'`, hints `['claude']`, `intelligencePriority≈0.9` — treating the model as best-effort.

**Version note (important):** the pseudocode above targets the **legacy sampling shape** — the initialize-time `check_client_capability` handshake and `create_message` of **2025-06-18 / 2025-11-25**. That is the **correct thing to code against today** on the current released SDK. But be aware the draft `2026-07-28` redesign is **stateless**: it removes the initialize handshake and moves to the **MRTR / `InputRequiredResult`** pattern with capabilities in per-request `_meta.io.modelcontextprotocol/clientCapabilities`. So the seam you build now is deliberately the pre-stateless one; see §4 on why you should *not* chase MRTR yet.

Approval-spam mitigation (only partially solvable): collapse the pipeline to **as few sampling round-trips as possible** — batch query-gen + rerank into one call, synthesis + cross-analysis into one, rather than per-source fan-out. But be honest that even 2–3 approvals per run is friction, that client-side rate limiting can independently throttle you, and that this path realistically only benefits VS Code users. Given all that, rate `SamplingBackend` as **optional / nice-to-have, not worth blocking a release on.** Ship the abstraction as a seam if you want it; don't ship it as the story.

## 4. The honest alternative — better at your real goal

Your real goal is *"let the host AI (Claude) do the reasoning, keyless for the user."* Sampling was a bad way to get there. There is a native way that works in **every** client today, needs zero deprecated/draft features, and adds zero approval popups.

**In MCP mode, demote sibyl from a self-contained agent to a retrieval / structured-context provider, and let the host model be the brain.** The host that called your tool *is already Claude* — so instead of borrowing Claude's tokens through a side channel, hand Claude the raw material and let it reason in its own turn.

Concretely, expose thin tools that fan out searches and return **structured intermediate artifacts**: `search_web`, `fetch_source`, `rank_sources`, `extract_claims`, `build_comparison` — each returning ranked sources, per-source snippets, candidate claims, comparison tables. Claude then does query planning, synthesis, cross-analysis, and claim verification natively. This is keyless, free of DeepSeek for the MCP user, works in Claude Desktop / Code / Cursor / everything, and gives *better* reasoning than DeepSeek-flash ever did.

Keep **one heavyweight `research` tool** that still runs the full DeepSeek pipeline end-to-end with sibyl's own key, for hosts/users who want a one-shot answer and accept that sibyl pays. Expose both surfaces; let the host choose.

Weighing it: the retrieval-provider model is strictly superior to sampling for the stated goal — it achieves "host does the reasoning, keyless" *without* the deprecation track, the client-support gap, the approval spam, or the rate-limit/quota-drain exposure. The one cost is that it is a genuine redesign of the MCP surface (thin tools + artifacts) rather than a drop-in backend swap, and the "full autonomous research in one call" experience only survives via the heavyweight keyed tool. That is the right trade. Do **not** invest in the MRTR / `InputRequiredResult` successor now — it is even more round-trip-heavy and still needs client adoption Claude does not have; make it a watch-item.

**Bottom line:** drop the sampling plan. Keep litellm+key as the CLI/API engine and as an optional heavyweight MCP tool; reshape sibyl-as-MCP into a retrieval/context provider where the host is the brain. Optionally add a capability-gated `SamplingBackend` seam for VS Code, but don't sell it as the architecture.

## 5. What you can and cannot test from a dev box

**Can test without any Claude client:**
- That `check_client_capability(sampling)` correctly returns False and cleanly falls back to litellm — use the **MCP Inspector** or a client that declares no sampling; confirm no hang/crash.
- The full sampling code path *end-to-end* against a **supporting client**. **VS Code / Copilot is the only proving ground backed by a first-party doc**; MCPJam, `fast-agent`, and `mcp-agent` are *suggested but unverified* — confirm each actually declares the sampling capability before relying on it. This proves your `create_message` wiring, prompt→`SamplingMessage` conversion, error handling, and timeout logic, and lets you **measure how many approval prompts a real run generates** and whether client rate limiting kicks in.
- The retrieval-provider design fully: structured tools + artifacts are just normal MCP tools; testable in Inspector and in any client including real Claude Code.
- Whether your **pinned** `mcp` SDK version already emits `@deprecated` warnings — check the installed version rather than assuming (the decorator is on `main`, not confirmed for releases).

**Cannot test from a dev box:**
- That sampling routes to **Claude specifically** inside Claude Desktop / Code — those clients don't implement the client side (Code verified, Desktop inferred), so there is nothing to hit. You cannot validate the original architecture's premise anywhere, because the premise is false.
- Real quota accounting / rate-limit behavior against a Claude Max subscription — no such path exists to measure.
- Whether/when #1785 ships — that is a monitoring task, not a test. Gate any future work on the apify matrix flipping Claude to ✅.

The decisive fact — "keyless Claude reasoning via sampling inside Claude hosts" — is not something a dev box can make true; it is blocked on Anthropic shipping a feature that is simultaneously unimplemented in their clients and on a deprecation track in the next spec revision. Build for the world that exists.