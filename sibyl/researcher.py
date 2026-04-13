"""Core research engine — multi-step deep research with iterative refinement."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

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

        # Step 2: Generate search queries (for main query + sub-questions)
        progress("Generating search queries...")
        all_queries = await self._generate_search_queries(query, depth)
        for sq in sub_questions:
            extra = await self._generate_search_queries(sq, 1)
            all_queries.extend(extra)
        # Deduplicate
        seen_q = set()
        search_queries = []
        for q in all_queries:
            if q.lower() not in seen_q:
                seen_q.add(q.lower())
                search_queries.append(q)
        progress(f"Total search queries: {len(search_queries)}")

        # Step 3: Search the web
        progress("Searching the web...")
        all_results: List[SearchResult] = []
        for sq in search_queries:
            results = await search_web(sq, self.config.search_engine, max_results=5)
            all_results.extend(results)
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
        pages = await scrape_urls(urls_to_scrape, max_chars=6000)
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

        # Step 5: Per-sub-question analysis — PARALLEL (depth 2+)
        sub_analyses = []
        if depth >= 2 and sub_questions and good_pages:
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
                for gap_query in gaps[:3]:
                    results = await search_web(gap_query, self.config.search_engine, max_results=3)
                    for r in results:
                        if r.url not in seen_urls:
                            seen_urls.add(r.url)
                            unique_results.append(r)
                # Scrape new sources
                new_urls = [r.url for r in unique_results if r.url not in {p.url for p in pages}][:5]
                if new_urls:
                    progress(f"Reading {len(new_urls)} additional sources...")
                    new_pages = await scrape_urls(new_urls, max_chars=4000)
                    good_pages.extend([p for p in new_pages if p.text and not p.error])
                    progress(f"Total sources: {len(good_pages)}")
                search_queries.extend(gaps[:3])

        # Step 7: Cross-source analysis (depth 2+)
        cross_analysis_text = ""
        if depth >= 2 and good_pages:
            progress("Cross-referencing sources (sentiment + consensus + disagreements)...")
            from .analyzer import analyze_sources, format_cross_analysis
            provider = self.config.get_provider("analysis")
            cross = await analyze_sources(good_pages, query, provider)
            cross_analysis_text = format_cross_analysis(cross)
            progress(f"Sentiment: {cross.overall_sentiment} | Consensus: {len(cross.consensus_points)} | Disagreements: {len(cross.disagreement_points)}")

        # Step 8: Final synthesis
        progress("Synthesizing final report...")
        lang = language or self.config.language
        report = await self._synthesize(query, unique_results, good_pages, depth, sub_analyses, lang)

        # Step 9: Review and refine (depth 2+)
        if depth >= 2 and report.summary:
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

        text = await self._llm_call(provider, prompt, max_tokens=100)

        try:
            indices = [int(x.strip()) - 1 for x in text.split(",") if x.strip().isdigit()]
            filtered = [pages[i] for i in indices if 0 <= i < len(pages)]
            return filtered if len(filtered) >= 3 else pages[:12]
        except Exception:
            return pages[:12]

    async def _review_and_refine(self, report, pages: List[WebPage], language: str = "auto"):
        """Review the draft report and refine it for quality."""
        provider = self.config.get_provider("analysis")

        draft = f"""Summary: {report.summary}

Key Findings:
{chr(10).join(f'- {f}' for f in report.key_findings)}

Analysis: {report.analysis}

Predictions: {report.predictions}"""

        lang_instruction = ""
        if language == "zh":
            lang_instruction = "回复必须使用中文。"

        prompt = f"""You are a senior editor reviewing a research report draft.

RESEARCH QUESTION: {report.query}

DRAFT:
{draft}

Review this draft and provide an IMPROVED version with:
1. More specific data points and numbers (add if missing)
2. Stronger source citations
3. More nuanced analysis (not just listing facts but explaining WHY)
4. Clearer structure and better flow
5. A definitive conclusion that directly answers the research question

{lang_instruction}

Output the improved version in the same format:

## Summary
[improved summary — 3-4 detailed paragraphs]

## Key Findings
[improved findings — more specific, better cited]"""

        text = await self._llm_call(provider, prompt, max_tokens=3000)

        # Parse improved version
        new_summary = (self._extract_section(text, "Summary", "Key Findings")
                      or self._extract_section(text, "摘要", "关键发现")
                      or self._extract_section(text, "摘要", "要点"))
        new_findings_text = (self._extract_section(text, "Key Findings", "Analysis")
                            or self._extract_section(text, "关键发现", "分析")
                            or self._extract_section(text, "要点", "分析")
                            or self._extract_section(text, "Key Findings", ""))

        if new_summary and len(new_summary) > len(report.summary) * 0.5:
            report.summary = new_summary.strip()
        if new_findings_text:
            new_findings = [f.strip().lstrip("- ").lstrip("* ") for f in new_findings_text.splitlines()
                           if f.strip() and (f.strip()[0] in "-*0123456789")]
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

        # ── Sections 1 & 2: Summary + Findings (PARALLEL) ──
        summary_prompt = f"""{base_prompt}

You are a senior research analyst. Write a comprehensive SUMMARY that fully answers the research question.

Requirements:
- 4-5 detailed paragraphs
- Every paragraph must cite sources as [Source N]
- Include specific data: numbers, percentages, dollar amounts, dates
- Cover: current state, key drivers, outlook, implications
- Write with authority for decision-makers"""

        findings_prompt = f"""{base_prompt}

Based on all sources, list 10-15 KEY FINDINGS. Each finding must:
- Lead with a specific, bold claim backed by data
- Include at least one number, percentage, or date
- Cite the source [Source N]
- Explain the significance (WHY does this matter?)

Format each as a numbered item. Be specific, not generic."""

        summary, findings_text = await asyncio.gather(
            self._llm_call(provider, summary_prompt, max_tokens=2000),
            self._llm_call(provider, findings_prompt, max_tokens=2500),
        )

        # Quality checks (sequential since they depend on results)
        if len(summary) < 800:
            summary = await self._llm_call(provider, summary_prompt + "\n\nIMPORTANT: Write AT LEAST 4 substantial paragraphs with specific data points.", max_tokens=2500)

        findings = [f.strip().lstrip("- ").lstrip("* ") for f in findings_text.splitlines()
                     if f.strip() and len(f.strip()) > 20 and (f.strip()[0] in "-*0123456789")]

        if len(findings) < 5:
            findings_text = await self._llm_call(provider, f"""{base_prompt}

List exactly 10 KEY FINDINGS as numbered items. Each must include a specific number and cite [Source N]. Be detailed.""", max_tokens=2500)
            findings = [f.strip().lstrip("- ").lstrip("* ") for f in findings_text.splitlines()
                         if f.strip() and len(f.strip()) > 20 and (f.strip()[0] in "-*0123456789")]

        # ── Sections 3 & 4: Analysis + Predictions (PARALLEL for depth 3) ──
        analysis = ""
        predictions = ""

        analysis_prompt = f"""{base_prompt}

Write a DEEP ANALYSIS section. Cover:
1. Conflicting viewpoints — what do different sources/experts disagree on? Present both sides with evidence.
2. Underlying trends — what structural forces are driving this topic?
3. Second-order effects — what consequences might people be overlooking?
4. Historical context — how does the current situation compare to past patterns?

Cite sources as [Source N]. Be analytical, not just descriptive."""

        predictions_prompt = f"""{base_prompt}

SUMMARY SO FAR: {summary[:500]}

Write a PREDICTIONS section:
1. Most likely scenario (60%+ probability) — be specific with numbers and timeframes
2. Bull case — what could go better than expected?
3. Bear case — what could go wrong?
4. Key uncertainties — the 3 things that could change everything
5. Confidence rating (low/medium/high) with detailed justification

Be concrete — give specific numbers, dates, and thresholds where possible."""

        if depth >= 3:
            # Parallel: analysis + predictions
            analysis, predictions = await asyncio.gather(
                self._llm_call(provider, analysis_prompt, max_tokens=2000),
                self._llm_call(provider, predictions_prompt, max_tokens=2000),
            )
        elif depth >= 2:
            analysis = await self._llm_call(provider, analysis_prompt, max_tokens=2000)

        confidence = ""
        for line in (predictions or "").splitlines():
            if "confidence" in line.lower():
                confidence = line.strip()
                break

        sources = [
            Source(url=r.url, title=r.title, snippet=r.snippet)
            for r in search_results if pages and r.url in {p.url for p in pages}
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

    async def _llm_call(self, provider: Provider, prompt: str, max_tokens: int = 1500) -> str:
        """Call an LLM provider via LiteLLM with retry."""
        kwargs = {
            "model": provider.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if provider.api_key:
            kwargs["api_key"] = provider.api_key
        if provider.api_base:
            kwargs["api_base"] = provider.api_base

        for attempt in range(3):
            try:
                response = await litellm.acompletion(**kwargs)
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == 2:
                    return f"(LLM call failed after 3 attempts: {str(e)[:100]})"
                await asyncio.sleep(1 * (attempt + 1))

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
