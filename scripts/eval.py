#!/usr/bin/env python
"""Reproducible research-quality eval for sibyl.

Runs sibyl end-to-end on a small curated SimpleQA/FRAMES-style set and grades
each answer with a temperature-0 LLM judge on the SimpleQA trichotomy
(CORRECT / INCORRECT / NOT_ATTEMPTED). Touches zero product code — it only
calls the public `research()` API, so it re-runs verbatim as capabilities land.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/eval.py --depth 2                # full live run + grade
    python scripts/eval.py --depth 2 --limit 5      # quick smoke
    python scripts/eval.py --score-only --depth 2   # re-grade cached reports, no network
    python scripts/eval.py --write-badge            # update the README accuracy line

Results (raw report + extracted answer) are cached under evals/cache/ so
--score-only re-grades without re-running research.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import litellm

litellm.suppress_debug_info = True

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "evals" / "gold" / "simpleqa_frames_20.jsonl"
CACHE = ROOT / "evals" / "cache"


def _provider():
    """Grader provider from env — independent of the product config."""
    from sibyl.config import Config
    cfg = Config.from_env()
    return cfg.get_provider("general")


async def _grade_call(provider, prompt: str, max_tokens: int = 64) -> str:
    """A local copy of the _llm_call contract (temp=0, thinking off, retry) so
    the eval never imports Researcher internals."""
    kwargs = {
        "model": provider.model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    if provider.api_key:
        kwargs["api_key"] = provider.api_key
    if provider.api_base:
        kwargs["api_base"] = provider.api_base
    if "deepseek" in provider.model:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    for attempt in range(3):
        try:
            r = await litellm.acompletion(**kwargs)
            return (r.choices[0].message.content or "").strip()
        except Exception:
            if attempt == 2:
                return ""
            await asyncio.sleep(1 * (attempt + 1))
    return ""


def load_dataset(path: Path, limit: int = 0):
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return items[:limit] if limit else items


async def extract_answer(provider, question: str, report_text: str) -> str:
    prompt = (
        f"From the research report below, extract the single concise factual answer to the "
        f"question. If the report does not contain an answer, reply exactly NO_ANSWER.\n\n"
        f"QUESTION: {question}\n\nREPORT:\n{report_text[:6000]}\n\n"
        f"Answer with just the fact (a name, number, date, or short phrase) or NO_ANSWER."
    )
    return await _grade_call(provider, prompt, max_tokens=48)


async def judge(provider, question: str, gold: str, aliases, predicted: str) -> str:
    if not predicted or predicted.strip().upper() == "NO_ANSWER":
        return "NOT_ATTEMPTED"
    gold_str = gold + ((" (also accept: " + ", ".join(aliases) + ")") if aliases else "")
    prompt = (
        f"You are grading a predicted answer against the gold answer for a factual question, "
        f"using the SimpleQA rubric.\n\nQUESTION: {question}\nGOLD ANSWER: {gold_str}\n"
        f"PREDICTED ANSWER: {predicted}\n\n"
        f"Grade CORRECT if the predicted answer contains the gold answer (or an accepted alias) "
        f"and does not contradict it; minor formatting/extra words are fine. Grade INCORRECT if it "
        f"gives a different or contradictory answer. Grade NOT_ATTEMPTED only if it declines/says it "
        f"doesn't know. Reply with exactly one token: CORRECT, INCORRECT, or NOT_ATTEMPTED."
    )
    out = (await _grade_call(provider, prompt, max_tokens=8)).upper()
    for tag in ("NOT_ATTEMPTED", "INCORRECT", "CORRECT"):
        if tag in out:
            return tag
    return "INCORRECT"


async def run_one(item, depth: int, provider):
    """Run sibyl (or load cache), extract the answer, cache the raw result."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{item['id']}_d{depth}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return data
    from sibyl.config import Config
    from sibyl.researcher import Researcher
    cfg = Config.from_env()
    r = Researcher(cfg)
    rep = await r.research(item["question"], depth=depth)
    report_text = rep.summary + "\n\n" + "\n".join(rep.key_findings)
    answer = await extract_answer(provider, item["question"], report_text)
    data = {"id": item["id"], "question": item["question"], "answer": answer,
            "report_text": report_text[:8000]}
    cache_file.write_text(json.dumps(data, ensure_ascii=False))
    return data


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dataset", default=str(GOLD))
    ap.add_argument("--score-only", action="store_true", help="re-grade cached reports, no research")
    ap.add_argument("--write-badge", action="store_true")
    args = ap.parse_args()

    provider = _provider()
    items = load_dataset(Path(args.dataset), args.limit)
    sem = asyncio.Semaphore(args.concurrency)

    async def _do(item):
        async with sem:
            if args.score_only:
                cache_file = CACHE / f"{item['id']}_d{args.depth}.json"
                if not cache_file.exists():
                    return item, None, "NO_CACHE"
                data = json.loads(cache_file.read_text())
            else:
                data = await run_one(item, args.depth, provider)
            grade = await judge(provider, item["question"], item["gold"],
                                item.get("aliases", []), data.get("answer", ""))
            return item, data, grade

    results = await asyncio.gather(*[_do(it) for it in items])

    counts = {"CORRECT": 0, "INCORRECT": 0, "NOT_ATTEMPTED": 0, "NO_CACHE": 0}
    print(f"\n{'id':7} {'type':8} {'grade':13} answer")
    print("-" * 70)
    for item, data, grade in results:
        counts[grade] = counts.get(grade, 0) + 1
        ans = (data.get("answer", "") if data else "")[:34]
        print(f"{item['id']:7} {item.get('type',''):8} {grade:13} {ans}")

    total = len(results)
    correct = counts["CORRECT"]
    attempted = correct + counts["INCORRECT"]
    acc = 100 * correct / total if total else 0
    cga = 100 * correct / attempted if attempted else 0
    print("-" * 70)
    print(f"CORRECT {correct}/{total} = {acc:.1f}%  |  attempted {attempted}/{total}  |  "
          f"correct-given-attempted {cga:.1f}%  |  not_attempted {counts['NOT_ATTEMPTED']}")
    by_type = {}
    for item, _, grade in results:
        t = item.get("type", "?")
        by_type.setdefault(t, [0, 0])
        by_type[t][1] += 1
        if grade == "CORRECT":
            by_type[t][0] += 1
    for t, (c, n) in sorted(by_type.items()):
        print(f"  {t}: {c}/{n} = {100*c/n:.0f}%")

    if args.write_badge:
        badge = f"Research-quality eval: **{acc:.0f}%** correct ({correct}/{total}) on a 20-question SimpleQA/FRAMES set, depth {args.depth}."
        print("\nbadge:", badge)


if __name__ == "__main__":
    asyncio.run(main())
