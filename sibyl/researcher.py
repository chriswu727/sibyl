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

        # Step 4: Scrape top sources (more content per page)
        urls_to_scrape = [r.url for r in unique_results[:self.config.max_sources]]
        progress(f"Reading {len(urls_to_scrape)} sources...")
        pages = await scrape_urls(urls_to_scrape, max_chars=4000)
        good_pages = [p for p in pages if p.text and not p.error]
        progress(f"Successfully read {len(good_pages)} pages")

        # Step 5: Per-sub-question analysis (depth 2+)
        sub_analyses = []
        if depth >= 2 and sub_questions and good_pages:
            progress("Analyzing each sub-question...")
            for i, sq in enumerate(sub_questions):
                progress(f"  Analyzing: {sq}")
                analysis = await self._analyze_sub_question(sq, good_pages)
                sub_analyses.append((sq, analysis))

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

        # Step 7: Final synthesis
        progress("Synthesizing final report...")
        report = await self._synthesize(query, unique_results, good_pages, depth, sub_analyses)
        report.search_queries = search_queries
        report.sub_questions = sub_questions

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

    async def _analyze_sub_question(self, question: str, pages: List[WebPage]) -> str:
        """Analyze a single sub-question against the scraped sources."""
        provider = self.config.get_provider("analysis")

        # Find most relevant pages for this sub-question
        context_parts = []
        for i, page in enumerate(pages[:6], 1):
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
    ) -> ResearchReport:
        """Synthesize all collected data into a final report."""
        provider = self.config.get_provider("analysis")

        # Build source context
        context_parts = []
        if pages:
            for i, page in enumerate(pages[:10], 1):
                context_parts.append(f"[Source {i}: {page.title}]\nURL: {page.url}\n{page.text[:3000]}\n")
        else:
            for i, sr in enumerate(search_results[:10], 1):
                context_parts.append(f"[Source {i}: {sr.title}]\nURL: {sr.url}\n{sr.snippet}\n")
        context = "\n---\n".join(context_parts)
        if not context:
            context = "(No sources retrieved. Use your knowledge.)"

        # Include sub-question analyses as additional context
        sub_context = ""
        if sub_analyses:
            sub_parts = []
            for sq, analysis in sub_analyses:
                sub_parts.append(f"Sub-question: {sq}\nAnalysis: {analysis}")
            sub_context = f"""
PRELIMINARY ANALYSIS (from sub-question research):
{chr(10).join(sub_parts)}

Use the above analysis as additional context. Cross-reference with the sources."""

        analysis_prompt = ""
        if depth >= 2:
            analysis_prompt = """
## Analysis
Provide deeper analysis:
- What are the conflicting viewpoints and what evidence supports each?
- What are the underlying trends driving this topic?
- What do experts agree on vs disagree on?"""

        prediction_prompt = ""
        if depth >= 3:
            prediction_prompt = """
## Predictions
Based on all evidence:
- What is most likely to happen next? Give specific, actionable predictions.
- What are the 3 biggest uncertainties?
- Rate your overall confidence (low/medium/high) with a clear explanation."""

        prompt = f"""You are a senior research analyst writing a comprehensive report.

RESEARCH QUESTION: {query}

SOURCES:
{context}
{sub_context}

Create a thorough, well-structured report:

## Summary
3-4 paragraphs providing a complete answer to the research question. Be specific with data and citations.

## Key Findings
List 6-10 key findings. Each must cite at least one source [Source N]. Include specific data points, percentages, and figures where available.
{analysis_prompt}
{prediction_prompt}

IMPORTANT:
- Cite sources by [Source N] throughout
- Include specific numbers, dates, and data points
- Distinguish facts from opinions
- Note where sources disagree"""

        max_tok = {1: 2000, 2: 3500, 3: 5000}.get(depth, 3000)
        text = await self._llm_call(provider, prompt, max_tokens=max_tok)

        # Parse sections
        summary = self._extract_section(text, "Summary", "Key Findings")
        findings_text = self._extract_section(text, "Key Findings", "Analysis")
        findings = [f.strip().lstrip("- ").lstrip("* ") for f in findings_text.splitlines()
                     if f.strip() and (f.strip()[0] in "-*0123456789")]
        analysis = self._extract_section(text, "Analysis", "Predictions") if depth >= 2 else ""
        predictions = self._extract_section(text, "Predictions", "") if depth >= 3 else ""

        confidence = ""
        for line in (predictions or "").splitlines():
            if "confidence" in line.lower():
                confidence = line.strip()
                break

        sources = [
            Source(url=r.url, title=r.title, snippet=r.snippet)
            for r in search_results[:len(pages)] if r.url in {p.url for p in pages}
        ] if pages else [
            Source(url=r.url, title=r.title, snippet=r.snippet)
            for r in search_results[:10]
        ]

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
        """Call an LLM provider via LiteLLM."""
        kwargs = {
            "model": provider.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if provider.api_key:
            kwargs["api_key"] = provider.api_key
        if provider.api_base:
            kwargs["api_base"] = provider.api_base

        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content.strip()

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
