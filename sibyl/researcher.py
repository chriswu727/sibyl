"""Core research engine — multi-step deep research with iterative refinement."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import litellm

from .config import Config, Provider
from .search import SearchResult, search_web
from .scraper import WebPage, scrape_urls

litellm.suppress_debug_info = True


@dataclass
class Source:
    url: str
    title: str
    snippet: str
    relevance: str = ""


@dataclass
class ResearchReport:
    query: str
    summary: str
    key_findings: List[str]
    sources: List[Source]
    analysis: str = ""
    predictions: str = ""
    cross_analysis: str = ""  # sentiment + consensus + disagreements
    confidence: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    model_used: str = ""
    search_queries: List[str] = field(default_factory=list)
    charts: List[str] = field(default_factory=list)
    market_data_summary: str = ""
    sub_questions: List[str] = field(default_factory=list)


class Researcher:
    """Multi-step research engine with sub-question decomposition and iterative deepening."""

    def __init__(self, config: Config):
        self.config = config

    async def research(
        self,
        query: str,
        depth: int = 0,
        language: str = "",
        on_progress: Optional[callable] = None,
    ) -> ResearchReport:
        # One pooled client for the whole run — reuses TCP/TLS connections
        # across the many search + scrape calls to the same hosts.
        self._http = httpx.AsyncClient(
            follow_redirects=True,
            timeout=12.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        try:
            return await self._do_research(query, depth, language, on_progress)
        finally:
            await self._http.aclose()

    async def _do_research(
        self,
        query: str,
        depth: int = 0,
        language: str = "",
        on_progress: Optional[callable] = None,
    ) -> ResearchReport:
        depth = depth or self.config.max_depth

        def progress(msg: str):
            if on_progress:
                on_progress(msg)

        # Step 1: Decompose into sub-questions (depth 2+)
        sub_questions = []
        if depth >= 2:
            progress("Decomposing research question into sub-questions...")
            sub_questions = await self._decompose_question(query, depth)
            progress(f"Sub-questions: {sub_questions}")

        # Step 2: Generate search queries (main query + sub-questions) in parallel
        progress("Generating search queries...")
        gen_tasks = [self._generate_search_queries(query, depth)]
        gen_tasks += [self._generate_search_queries(sq, 1) for sq in sub_questions]
        gen_results = await asyncio.gather(*gen_tasks, return_exceptions=True)
        all_queries = []
        for lst in gen_results:
            if isinstance(lst, list):
                all_queries.extend(lst)
        # Deduplicate
        seen_q = set()
        search_queries = []
        for q in all_queries:
            if q.lower() not in seen_q:
                seen_q.add(q.lower())
                search_queries.append(q)
        progress(f"Total search queries: {len(search_queries)}")

        # Step 3: Search the web — all queries concurrently (bounded)
        progress("Searching the web...")
        search_sem = asyncio.Semaphore(8)

        async def _run_search(q: str, academic: bool) -> List[SearchResult]:
            async with search_sem:
                return await search_web(
                    q, self.config.search_engine, max_results=5,
                    client=self._http, include_academic=academic,
                )

        # Academic API is heavily rate-limited — enable it on the first 2 queries only
        search_tasks = [_run_search(q, i < 2) for i, q in enumerate(search_queries)]
        results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)
        all_results: List[SearchResult] = []
        for res in results_lists:
            if isinstance(res, list):
                all_results.extend(res)
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        progress(f"Found {len(unique_results)} unique sources")

        # Step 4: Scrape top sources — try more URLs to compensate for failures
        scrape_count = min(len(unique_results), self.config.max_sources * 2)  # Try 2x to get enough
        urls_to_scrape = [r.url for r in unique_results[:scrape_count]]
        progress(f"Reading {len(urls_to_scrape)} sources...")
        pages = await scrape_urls(urls_to_scrape, max_chars=6000, client=self._http, extractor=self.config.extractor, jina_fallback=self.config.jina_fallback)
        good_pages = [p for p in pages if p.text and len(p.text) > 100 and not p.error]

        # Supplement with search snippets for failed pages
        scraped_urls = {p.url for p in good_pages}
        for r in unique_results:
            if r.url not in scraped_urls and r.snippet and len(r.snippet) > 50:
                good_pages.append(WebPage(url=r.url, title=r.title, text=r.snippet))
                if len(good_pages) >= self.config.max_sources + 5:
                    break

        progress(f"Total usable sources: {len(good_pages)} ({sum(1 for p in good_pages if len(p.text) > 200)} full, rest snippets)")

        # Step 4b: Source relevance filtering
        if len(good_pages) > 4:
            progress("Evaluating source relevance...")
            good_pages = await self._filter_sources(query, good_pages)
            progress(f"Kept {len(good_pages)} most relevant sources")

        # Step 5: Per-sub-question analysis — PARALLEL (depth 3 only).
        # A/B testing showed feeding these pre-digested analyses to synthesis
        # gives no quality gain over synthesizing straight from the sources, so
        # at depth 2 we skip the ~14s step. Depth 3 still needs them as input to
        # gap-finding below.
        sub_analyses = []
        if depth >= 3 and sub_questions and good_pages:
            progress(f"Analyzing {len(sub_questions)} sub-questions in parallel...")
            tasks = [self._analyze_sub_question(sq, good_pages) for sq in sub_questions]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sq, result in zip(sub_questions, results):
                if isinstance(result, str):
                    sub_analyses.append((sq, result))
                else:
                    sub_analyses.append((sq, f"(Analysis failed: {result})"))

        # Step 6: Identify knowledge gaps and do a second search round (depth 3)
        if depth >= 3 and good_pages:
            progress("Identifying knowledge gaps...")
            gaps = await self._identify_gaps(query, sub_analyses, good_pages)
            if gaps:
                progress(f"Found gaps, searching for: {gaps}")
                gap_lists = await asyncio.gather(*[
                    search_web(gap_query, self.config.search_engine, max_results=3, client=self._http)
                    for gap_query in gaps[:3]
                ], return_exceptions=True)
                for results in gap_lists:
                    if not isinstance(results, list):
                        continue
                    for r in results:
                        if r.url not in seen_urls:
                            seen_urls.add(r.url)
                            unique_results.append(r)
                # Scrape new sources
                scraped_page_urls = {p.url for p in pages}
                new_urls = [r.url for r in unique_results if r.url not in scraped_page_urls][:5]
                if new_urls:
                    progress(f"Reading {len(new_urls)} additional sources...")
                    new_pages = await scrape_urls(new_urls, max_chars=4000, client=self._http, extractor=self.config.extractor, jina_fallback=self.config.jina_fallback)
                    good_pages.extend([p for p in new_pages if p.text and not p.error])
                    progress(f"Total sources: {len(good_pages)}")
                search_queries.extend(gaps[:3])

        # Steps 7 & 8: Cross-source analysis runs concurrently with synthesis —
        # the cross-analysis output is only attached to the report at the end,
        # so it has no dependency on the synthesized sections.
        async def _cross_analysis():
            if not (depth >= 2 and good_pages):
                return "", None
            from .analyzer import analyze_sources, format_cross_analysis
            provider = self.config.get_provider("analysis")
            cross = await analyze_sources(good_pages, query, provider)
            return format_cross_analysis(cross), cross

        progress("Synthesizing report + cross-referencing sources in parallel...")
        lang = language or self.config.language
        report, (cross_analysis_text, cross) = await asyncio.gather(
            self._synthesize(query, unique_results, good_pages, depth, sub_analyses, lang),
            _cross_analysis(),
        )
        if cross:
            progress(f"Sentiment: {cross.overall_sentiment} | Consensus: {len(cross.consensus_points)} | Disagreements: {len(cross.disagreement_points)}")

        # Step 9: Review and refine (depth 2+; skipped in fast mode).
        # An A/B showed review meaningfully improves the report, so it's on by
        # default — fast mode trades that polish for ~20% lower latency.
        if depth >= 2 and report.summary and not self.config.fast:
            progress("Reviewing and refining report...")
            report = await self._review_and_refine(report, good_pages, lang)

        report.search_queries = search_queries
        report.sub_questions = sub_questions
        report.cross_analysis = cross_analysis_text

        progress("Research complete!")
        return report

    async def _decompose_question(self, query: str, depth: int) -> List[str]:
        """Break a complex question into 3-5 focused sub-questions."""
        provider = self.config.get_provider("general")
        num = {2: 3, 3: 5}.get(depth, 3)

        prompt = f"""Break this research question into {num} specific, focused sub-questions that together would provide a comprehensive answer.

Research question: {query}

Each sub-question should target a different aspect (e.g., causes, effects, data, opinions, future outlook).

Return ONLY the sub-questions, one per line, no numbering."""

        text = await self._llm_call(provider, prompt, max_tokens=500)
        questions = [q.strip().lstrip("0123456789.-) ") for q in text.strip().splitlines() if q.strip() and len(q.strip()) > 10]
        return questions[:num]

    async def _generate_search_queries(self, query: str, depth: int) -> List[str]:
        """Generate diverse search queries for a question."""
        provider = self.config.get_provider("general")
        num_queries = {1: 2, 2: 3, 3: 4}.get(depth, 2)

        prompt = f"""Generate {num_queries} diverse search queries to research this topic.
Cover different angles: data, expert opinions, recent news, contrarian views.

Topic: {query}

Return ONLY the queries, one per line."""

        text = await self._llm_call(provider, prompt, max_tokens=400)
        queries = [q.strip().lstrip("0123456789.-) ") for q in text.strip().splitlines() if q.strip() and len(q.strip()) > 10]
        if query not in queries:
            queries.insert(0, query)
        return queries[:num_queries + 1]

    async def _filter_sources(self, query: str, pages: List[WebPage]) -> List[WebPage]:
        """Have LLM rank sources by relevance, keep top ones."""
        provider = self.config.get_provider("general")

        source_list = "\n".join(
            f"{i}. [{page.title}] — {page.text[:150]}"
            for i, page in enumerate(pages, 1)
        )

        prompt = f"""Rate the relevance of each source for researching: {query}

Sources:
{source_list}

Return ONLY the numbers of the RELEVANT sources (score 7+/10), comma-separated.
Example: 1,3,4,7"""

        text = await self._llm_call(provider, prompt, max_tokens=200)

        try:
            indices = [int(x.strip()) - 1 for x in text.split(",") if x.strip().isdigit()]
            filtered = [pages[i] for i in indices if 0 <= i < len(pages)]
            return filtered if len(filtered) >= 3 else pages[:12]
        except Exception:
            return pages[:12]

    async def _review_and_refine(self, report, pages: List[WebPage], language: str = "auto"):
        """Review the draft and refine summary + findings — two independent
        refinements run in parallel."""
        provider = self.config.get_provider("analysis")

        draft = f"""Summary: {report.summary}

Key Findings:
{chr(10).join(f'- {f}' for f in report.key_findings)}

Analysis: {report.analysis}

Predictions: {report.predictions}"""

        lang_instruction = "回复必须使用中文。" if language == "zh" else ""

        summary_prompt = f"""You are a senior editor improving the SUMMARY of a research report.

RESEARCH QUESTION: {report.query}

DRAFT:
{draft}

Rewrite the summary to be stronger:
1. More specific data points and numbers (add if missing)
2. Stronger source citations
3. Explain WHY, not just what
4. Clearer structure and flow
5. A definitive conclusion that directly answers the research question

{lang_instruction}

Output ONLY the improved summary — 3-4 detailed paragraphs, no headers."""

        findings_prompt = f"""You are a senior editor improving the KEY FINDINGS of a research report.

RESEARCH QUESTION: {report.query}

DRAFT:
{draft}

Rewrite the key findings to be stronger: each must lead with a specific claim
backed by a number/percentage/date, cite [Source N] inline, and explain why it matters.

{lang_instruction}

Return json: an object with key "findings" whose value is a JSON array of strings, one improved finding per string. Example: {{"findings": ["Improved finding with data [Source 2]", "..."]}}."""

        new_summary, new_findings_text = await asyncio.gather(
            self._llm_call(provider, summary_prompt, max_tokens=2000),
            self._llm_call(provider, findings_prompt, max_tokens=2000, json_mode=True),
        )

        if new_summary and len(new_summary) > len(report.summary) * 0.5:
            report.summary = new_summary.strip()
        if new_findings_text:
            new_findings = self._parse_findings(new_findings_text)
            if len(new_findings) >= len(report.key_findings) * 0.5:
                report.key_findings = new_findings

        return report

    async def _analyze_sub_question(self, question: str, pages: List[WebPage]) -> str:
        """Analyze a single sub-question against the scraped sources."""
        provider = self.config.get_provider("analysis")

        # Find most relevant pages for this sub-question
        context_parts = []
        for i, page in enumerate(pages[:10], 1):
            context_parts.append(f"[Source {i}]: {page.text[:1500]}")
        context = "\n---\n".join(context_parts)

        prompt = f"""Based on the sources below, provide a concise answer to this question.
Cite sources by number. Be specific with data and facts.

Question: {question}

Sources:
{context}

Answer in 2-3 paragraphs."""

        return await self._llm_call(provider, prompt, max_tokens=800)

    async def _identify_gaps(self, query: str, sub_analyses: list, pages: List[WebPage]) -> List[str]:
        """Identify knowledge gaps after first round of analysis."""
        provider = self.config.get_provider("general")

        analyses_text = ""
        for sq, analysis in sub_analyses:
            analyses_text += f"\nQ: {sq}\nA: {analysis[:300]}\n"

        prompt = f"""I'm researching: {query}

I've analyzed these sub-questions so far:
{analyses_text}

What important aspects are MISSING from this analysis? What data or perspectives haven't been covered?

List 2-3 specific search queries that would fill these gaps.
Return ONLY the queries, one per line."""

        text = await self._llm_call(provider, prompt, max_tokens=300)
        return [q.strip().lstrip("0123456789.-) ") for q in text.strip().splitlines() if q.strip() and len(q.strip()) > 10][:3]

    async def _synthesize(
        self,
        query: str,
        search_results: List[SearchResult],
        pages: List[WebPage],
        depth: int,
        sub_analyses: list = None,
        language: str = "auto",
    ) -> ResearchReport:
        """Synthesize via section-by-section generation for maximum depth."""
        provider = self.config.get_provider("analysis")

        # Build source context (more content per page)
        context_parts = []
        if pages:
            for i, page in enumerate(pages[:12], 1):
                context_parts.append(f"[Source {i}: {page.title}]\nURL: {page.url}\n{page.text[:4000]}\n")
        else:
            for i, sr in enumerate(search_results[:10], 1):
                context_parts.append(f"[Source {i}: {sr.title}]\nURL: {sr.url}\n{sr.snippet}\n")
        context = "\n---\n".join(context_parts)
        if not context:
            context = "(No sources retrieved. Use your knowledge.)"

        sub_context = ""
        if sub_analyses:
            sub_parts = [f"Sub-question: {sq}\nAnalysis: {a}" for sq, a in sub_analyses]
            sub_context = "\n\nPRELIMINARY ANALYSIS:\n" + "\n".join(sub_parts)

        lang_inst = ""
        if language == "zh":
            lang_inst = "\n\nIMPORTANT: Write ENTIRELY in Chinese (中文). Only keep source URLs and proper nouns in English."
        elif language not in ("auto", "en", ""):
            lang_inst = f"\n\nIMPORTANT: Write ENTIRELY in {language}."

        base_prompt = f"RESEARCH QUESTION: {query}\n\nSOURCES:\n{context}{sub_context}{lang_inst}"

        # Density directive — A/B showed a length-bounded "every sentence adds a
        # distinct fact, no filler" instruction produces output that is both
        # faster to generate (fewer tokens) AND sharper (the verbose variant was
        # padding). Generation is decode-bound, so fewer tokens = real latency.
        DENSITY = ("\n\nBe information-dense: every sentence must add a distinct "
                   "fact or insight. No filler, no restating the question, no "
                   "throat-clearing preamble. Tight analytical prose.")

        # ── Sections 1 & 2: Summary + Findings (PARALLEL) ──
        summary_prompt = f"""{base_prompt}

You are a senior research analyst. Write a comprehensive SUMMARY that fully answers the research question.

Requirements:
- 4-5 substantial paragraphs
- Every paragraph must cite sources as [Source N]
- Include specific data: numbers, percentages, dollar amounts, dates
- Cover: current state, key drivers, outlook, implications
- Write with authority for decision-makers{DENSITY}"""

        findings_prompt = f"""{base_prompt}

Based on all sources, list 10-15 KEY FINDINGS. Each finding must:
- Lead with a specific, bold claim backed by data
- Include at least one number, percentage, or date
- Cite the source inline as [Source N]
- Explain the significance (WHY does this matter?)

Return json: an object with a single key "findings" whose value is a JSON array of strings, one finding per string with its [Source N] citation inline. Example: {{"findings": ["Revenue grew 12% to $4.1B in Q2 [Source 3], signaling accelerating demand", "..."]}}. Be specific, not generic.{DENSITY}"""

        # Analysis depends only on the sources (not on summary/findings), so it
        # joins this first parallel batch at depth 2+ instead of running after.
        analysis_prompt = f"""{base_prompt}

Write a DEEP ANALYSIS section covering:
1. Conflicting viewpoints — what do different sources/experts disagree on? Present both sides with evidence.
2. Underlying trends — what structural forces are driving this topic?
3. Second-order effects — what consequences might people be overlooking?
4. Historical context — how does the current situation compare to past patterns?

Cite sources as [Source N]. Be analytical, not just descriptive.{DENSITY}"""

        # Predictions (depth 3) also depends only on the sources — an A/B showed
        # feeding it the finished summary gave no quality gain — so it joins the
        # same parallel batch instead of running as a serial second round.
        predictions_prompt = f"""{base_prompt}

Write a PREDICTIONS section:
1. Most likely scenario (60%+ probability) — be specific with numbers and timeframes
2. Bull case — what could go better than expected?
3. Bear case — what could go wrong?
4. Key uncertainties — the 3 things that could change everything
5. Confidence rating (low/medium/high) with detailed justification

Be concrete — give specific numbers, dates, and thresholds where possible.{DENSITY}"""

        batch = [
            self._llm_call(provider, summary_prompt, max_tokens=1600),
            self._llm_call(provider, findings_prompt, max_tokens=2000, json_mode=True),
        ]
        slot = {}
        if depth >= 2:
            slot["analysis"] = len(batch)
            # Sized to fit a complete 4-part analysis even on a verbose run —
            # the density directive shortens the average, but length varies, and
            # a tight cap truncates mid-section (the old 2000 cap did too).
            batch.append(self._llm_call(provider, analysis_prompt, max_tokens=2400))
        if depth >= 3:
            slot["predictions"] = len(batch)
            batch.append(self._llm_call(provider, predictions_prompt, max_tokens=2200))

        # These calls share the whole source context as a prefix — warm the
        # cache once so the parallel batch hits it instead of each re-billing it.
        if len(batch) >= 2 and "deepseek" in provider.model and len(context) > 2000:
            await self._warm_cache(provider, base_prompt)
        batch_results = await asyncio.gather(*batch)
        summary, findings_text = batch_results[0], batch_results[1]
        analysis = batch_results[slot["analysis"]] if "analysis" in slot else ""
        predictions = batch_results[slot["predictions"]] if "predictions" in slot else ""

        # Quality checks (sequential since they depend on results)
        if len(summary) < 800:
            summary = await self._llm_call(provider, summary_prompt + "\n\nIMPORTANT: Write AT LEAST 4 substantial paragraphs with specific data points.", max_tokens=2500)

        findings = self._parse_findings(findings_text)

        if len(findings) < 5:
            findings_text = await self._llm_call(provider, f"""{base_prompt}

Return json: an object with key "findings" whose value is a JSON array of exactly 10 strings. Each string is one key finding with a specific number and an inline [Source N] citation. Example: {{"findings": ["Finding one with a datum [Source 1]", "..."]}}. Be detailed.""", max_tokens=2500, json_mode=True)
            findings = self._parse_findings(findings_text)

        confidence = ""
        for line in (predictions or "").splitlines():
            if "confidence" in line.lower():
                confidence = line.strip()
                break

        page_urls = {p.url for p in pages} if pages else set()
        sources = [
            Source(url=r.url, title=r.title, snippet=r.snippet)
            for r in search_results if r.url in page_urls
        ] if pages else [
            Source(url=r.url, title=r.title, snippet=r.snippet)
            for r in search_results[:10]
        ]
        # Include all scraped sources
        if pages and not sources:
            sources = [Source(url=p.url, title=p.title, snippet=p.text[:100]) for p in pages]

        return ResearchReport(
            query=query,
            summary=summary.strip(),
            key_findings=findings,
            sources=sources,
            analysis=analysis.strip(),
            predictions=predictions.strip(),
            confidence=confidence,
            model_used=provider.model,
        )

    async def _warm_cache(self, provider: Provider, prefix: str) -> None:
        """Fire a 1-token request to populate DeepSeek's prefix cache.

        The synthesis calls run in parallel and share a large source-context
        prefix; without warming they all cache-miss and re-bill that prefix.
        One cheap warm-up first makes them cache-hit (~10x cheaper input tokens),
        and its prefill is offset by the batch skipping prefill, so it's roughly
        latency-neutral. Best-effort: failures are swallowed.
        """
        kwargs = {
            "model": provider.model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": prefix}],
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if provider.api_key:
            kwargs["api_key"] = provider.api_key
        if provider.api_base:
            kwargs["api_base"] = provider.api_base
        try:
            await litellm.acompletion(**kwargs)
        except Exception:
            pass

    async def _llm_call(
        self,
        provider: Provider,
        prompt: str,
        max_tokens: int = 1500,
        thinking: bool = False,
        json_mode: bool = False,
    ) -> str:
        """Call an LLM provider via LiteLLM with retry.

        Thinking is disabled by default: a head-to-head found DeepSeek V4's
        reasoning pass gives no quality gain on sibyl's generation tasks (the
        non-thinking output was comparable or more concrete) while adding latency
        and, on short-output calls, starving the content budget. Pass
        ``thinking=True`` only if a future call genuinely needs step-by-step
        reasoning. The toggle is a no-op for non-DeepSeek providers.

        ``json_mode=True`` requests a JSON object response (DeepSeek's json_object
        mode) for machine-parsed calls — the prompt must contain the word "json"
        and an example. Keep ``max_tokens`` generous: a truncated JSON string
        fails to parse and triggers the empty-content retry.
        """
        kwargs = {
            "model": provider.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if provider.api_key:
            kwargs["api_key"] = provider.api_key
        if provider.api_base:
            kwargs["api_base"] = provider.api_base
        if not thinking and "deepseek" in provider.model:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(3):
            try:
                response = await litellm.acompletion(**kwargs)
                content = (response.choices[0].message.content or "").strip()
                # V4 thinking can occasionally spend the whole budget on
                # reasoning and return empty content — retry rather than emit an
                # empty section.
                if not content and attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return content
            except Exception as e:
                if attempt == 2:
                    return f"(LLM call failed after 3 attempts: {str(e)[:100]})"
                await asyncio.sleep(1 * (attempt + 1))

    @staticmethod
    def _parse_findings(text: str) -> List[str]:
        """Parse findings from a JSON `{"findings": [...]}` object, falling back
        to the legacy numbered/bulleted-line split so nothing regresses if the
        model ignores JSON mode or the JSON is truncated."""
        import re
        # Strip a leading list enumerator ("1. ", "2) ", "- ", "* ") so the
        # renderer's own numbering doesn't produce "1. 1. ...".
        def _clean(s: str) -> str:
            return re.sub(r'^\s*(?:\d+[.)]\s*|[-*•]\s*)', '', s).strip()

        stripped = (text or "").strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                data = json.loads(stripped)
                items = data.get("findings") if isinstance(data, dict) else data
                if isinstance(items, list):
                    out = [_clean(str(x)) for x in items if str(x).strip()]
                    out = [x for x in out if x]
                    if out:
                        return out
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
        return [_clean(f) for f in stripped.splitlines()
                if f.strip() and len(f.strip()) > 20 and (f.strip()[0] in "-*0123456789")]

    @staticmethod
    def _extract_section(text: str, start_header: str, end_header: str) -> str:
        """Extract content between two markdown headers."""
        import re
        lines = text.splitlines()
        start_idx = None
        end_idx = len(lines)

        for i, line in enumerate(lines):
            clean = re.sub(r'[#*_`]', '', line).strip()
            if clean.lower().startswith(start_header.lower()) and start_idx is None:
                start_idx = i + 1
            elif end_header and clean.lower().startswith(end_header.lower()) and start_idx is not None:
                end_idx = i
                break

        if start_idx is None:
            if not end_header:
                return text
            return ""
        return "\n".join(lines[start_idx:end_idx])
