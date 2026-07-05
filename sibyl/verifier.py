"""Claim verification — re-check each finding against its cited source text.

Borrowed from Claude's CitationAgent: after synthesis, verify that each finding's
specific claim is actually supported by the [Source N] it cites. Unsupported
findings (including confident claims about fictional/nonexistent entities that no
real source backs) get flagged. Keyless: one cheap DeepSeek JSON call, with a
pure-stdlib lexical fallback so it degrades offline and never crashes the run.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import List

import litellm

from .config import Provider
from .context import build_source_context
from .scraper import WebPage

litellm.suppress_debug_info = True

_CITE_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


@dataclass
class FindingVerification:
    index: int              # 1-based finding index
    supported: bool
    confidence: str = "low"  # high | medium | low
    cited: List[int] = field(default_factory=list)
    note: str = ""


def _extract_citations(finding: str) -> List[int]:
    return [int(m) for m in _CITE_RE.findall(finding or "")]


def _tokens(s: str):
    return set(re.findall(r"[a-z0-9]{4,}", (s or "").lower()))


def _lexical_verdict(index: int, finding: str, pages: List[WebPage]) -> FindingVerification:
    """Fallback grounding check: does the cited source text share enough
    distinctive tokens with the finding? Used to fill LLM omissions or when the
    JSON parse fails entirely."""
    cited = _extract_citations(finding)
    if not cited:
        return FindingVerification(index, False, "low", [], "no citation")
    f_tok = _tokens(finding)
    if not f_tok:
        return FindingVerification(index, True, "low", cited, "")
    best = 0.0
    for n in cited:
        if 1 <= n <= len(pages):
            s_tok = _tokens(pages[n - 1].text)
            if f_tok:
                best = max(best, len(f_tok & s_tok) / len(f_tok))
    supported = best >= 0.35
    return FindingVerification(index, supported, "medium" if supported else "low",
                               cited, "" if supported else "low source overlap")


async def verify_findings(findings: List[str], pages: List[WebPage],
                          provider: Provider) -> List[FindingVerification]:
    """Return a FindingVerification per finding (same order/length)."""
    if not findings:
        return []
    pages = pages or []
    # limit=len(pages): the verifier must see EVERY cited source, or a finding
    # citing [Source 15] gets falsely flagged when synthesis numbered 1..N>12.
    context = build_source_context(pages, limit=len(pages) or 1)
    numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(findings, 1))
    prompt = f"""SOURCES:
{context}

FINDINGS TO VERIFY:
{numbered}

For each finding, check whether its cited [Source N] text substantiates the
finding's claim. The source counts as support if it states OR clearly implies the
claim — exact wording may differ, and a close paraphrase or a figure rounded/
reformatted from the source still counts. Mark NOT supported only when the finding
cites a [Source N] not shown above, cites nothing, or asserts something the cited
source does not contain or contradicts (including confident claims about fictional
or nonexistent entities). Judge against the source text, not your own knowledge.

Return json: {{"verdicts": [{{"index": <1-based>, "supported": <bool>, "confidence": "high|medium|low", "cited": [<n>...], "note": "<=12 words"}}]}}, one per finding."""

    kwargs = {
        "model": provider.model,
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if provider.api_key:
        kwargs["api_key"] = provider.api_key
    if provider.api_base:
        kwargs["api_base"] = provider.api_base
    if "deepseek" in provider.model:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    by_index = {}
    for attempt in range(3):
        try:
            resp = await litellm.acompletion(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                break
            data = json.loads(text)
            verdicts = data.get("verdicts") if isinstance(data, dict) else data
            if isinstance(verdicts, list):
                for v in verdicts:
                    if not isinstance(v, dict) or "index" not in v:
                        continue
                    try:
                        idx = int(v["index"])
                    except (ValueError, TypeError):
                        continue
                    cited = v.get("cited")
                    cited = [int(c) for c in cited if str(c).strip().isdigit()] if isinstance(cited, list) else []
                    conf = str(v.get("confidence", "low")).lower()
                    if conf not in ("high", "medium", "low"):
                        conf = "low"
                    by_index[idx] = FindingVerification(
                        idx, bool(v.get("supported", False)), conf, cited,
                        str(v.get("note", ""))[:80])
            break
        except json.JSONDecodeError:
            break  # non-JSON → lexical fallback below
        except Exception:
            if attempt == 2:
                break
            await asyncio.sleep(1 * (attempt + 1))

    # Fill any finding the LLM omitted (or the whole set on parse failure) with the
    # lexical verdict, so the list is always aligned 1:1 with findings.
    out = []
    for i, f in enumerate(findings, 1):
        out.append(by_index.get(i) or _lexical_verdict(i, f, pages))
    return out
