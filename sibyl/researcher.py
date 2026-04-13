"""Core research engine — search, scrape, analyze, synthesize."""
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
    relevance: str = ""  # LLM-assessed relevance


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
    charts: List[str] = field(default_factory=list)  # paths to chart PNGs
    market_data_summary: str = ""  # formatted market data text


class Researcher:
    """Multi-step research engine with LLM analysis."""

    def __init__(self, config: Config):
        self.config = config

    async def research(
        self,
        query: str,
        depth: int = 0,
        on_progress: Optional[callable] = None,
    ) -> ResearchReport:
        """Run a complete research cycle.

        Args:
            query: The research question
            depth: Override config depth (1=quick, 2=standard, 3=deep)
            on_progress: Callback for progress updates
        """
        depth = depth or self.config.max_depth

        def progress(msg: str):
            if on_progress:
                on_progress(msg)

        # Step 1: Generate search queries
        progress("Generating search queries...")
        search_queries = await self._generate_search_queries(query, depth)
        progress(f"Will search: {search_queries}")

        # Step 2: Search the web
        progress("Searching the web...")
        all_results: List[SearchResult] = []
        for sq in search_queries:
            results = await search_web(sq, self.config.search_engine, max_results=5)
            all_results.extend(results)
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        progress(f"Found {len(unique_results)} unique sources")

        # Step 3: Scrape top sources
        urls_to_scrape = [r.url for r in unique_results[:self.config.max_sources]]
        progress(f"Reading {len(urls_to_scrape)} sources...")
        pages = await scrape_urls(urls_to_scrape)
        good_pages = [p for p in pages if p.text and not p.error]
        progress(f"Successfully read {len(good_pages)} pages")

        # Step 4: Analyze and synthesize
        progress("Analyzing sources...")
        report = await self._synthesize(query, unique_results, good_pages, depth)
        report.search_queries = search_queries

        progress("Research complete!")
        return report

    async def _generate_search_queries(self, query: str, depth: int) -> List[str]:
        """Use LLM to generate diverse search queries."""
        provider = self.config.get_provider("general")
        num_queries = {1: 2, 2: 4, 3: 6}.get(depth, 3)

        prompt = f"""Generate {num_queries} diverse search queries to research this topic thoroughly.
The queries should cover different angles, perspectives, and data sources.

Topic: {query}

Return ONLY the queries, one per line, no numbering or bullets."""

        text = await self._llm_call(provider, prompt)
        queries = [q.strip() for q in text.strip().splitlines() if q.strip()]
        # Always include the original query
        if query not in queries:
            queries.insert(0, query)
        return queries[:num_queries + 1]

    async def _synthesize(
        self,
        query: str,
        search_results: List[SearchResult],
        pages: List[WebPage],
        depth: int,
    ) -> ResearchReport:
        """Synthesize scraped content into a research report."""
        provider = self.config.get_provider("analysis")

        # Build context from scraped pages (or search snippets as fallback)
        context_parts = []
        if pages:
            for i, page in enumerate(pages[:8], 1):
                context_parts.append(f"[Source {i}: {page.title}]\nURL: {page.url}\n{page.text[:2000]}\n")
        else:
            # Fallback: use search snippets if scraping failed
            for i, sr in enumerate(search_results[:10], 1):
                context_parts.append(f"[Source {i}: {sr.title}]\nURL: {sr.url}\n{sr.snippet}\n")
        context = "\n---\n".join(context_parts)
        if not context:
            context = "(No sources could be retrieved. Answer based on your knowledge.)"

        analysis_prompt = ""
        if depth >= 2:
            analysis_prompt = """
## Analysis
Provide deeper analysis:
- What are the conflicting viewpoints?
- What data or evidence supports each side?
- What are the underlying trends?"""

        prediction_prompt = ""
        if depth >= 3:
            prediction_prompt = """
## Predictions
Based on the evidence:
- What is likely to happen next?
- What are the key uncertainties?
- Rate your confidence (low/medium/high) and explain why."""

        prompt = f"""You are a research analyst. Based on the following sources, create a comprehensive research report.

RESEARCH QUESTION: {query}

SOURCES:
{context}

Create a structured report with:

## Summary
A concise 2-3 paragraph summary answering the research question.

## Key Findings
List 5-8 key findings, each as a bullet point with the source cited.
{analysis_prompt}
{prediction_prompt}

Be specific, cite sources by number [Source N], and distinguish facts from opinions."""

        max_tok = {1: 1500, 2: 2500, 3: 4000}.get(depth, 2500)
        text = await self._llm_call(provider, prompt, max_tokens=max_tok)

        # Parse the response into structured sections
        summary = self._extract_section(text, "Summary", "Key Findings")
        findings_text = self._extract_section(text, "Key Findings", "Analysis")
        findings = [f.strip().lstrip("- ").lstrip("* ") for f in findings_text.splitlines() if f.strip().startswith(("-", "*", "1", "2", "3", "4", "5", "6", "7", "8", "9"))]
        analysis = self._extract_section(text, "Analysis", "Predictions") if depth >= 2 else ""
        predictions = self._extract_section(text, "Predictions", "") if depth >= 3 else ""

        # Extract confidence if present
        confidence = ""
        if "confidence" in predictions.lower():
            for line in predictions.splitlines():
                if "confidence" in line.lower():
                    confidence = line.strip()
                    break

        sources = [
            Source(url=r.url, title=r.title, snippet=r.snippet)
            for r in search_results[:len(pages)]
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
            # Strip markdown: ##, **, *, etc.
            clean = re.sub(r'[#*_`]', '', line).strip()
            if clean.lower().startswith(start_header.lower()) and start_idx is None:
                start_idx = i + 1
            elif end_header and clean.lower().startswith(end_header.lower()) and start_idx is not None:
                end_idx = i
                break

        if start_idx is None:
            # Fallback: return everything if we can't find sections
            if not end_header:
                return text
            return ""
        return "\n".join(lines[start_idx:end_idx])
