"""Web search — DuckDuckGo (free, no API key needed)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
from urllib.parse import unquote, urlparse, parse_qs

import httpx


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def _extract_real_url(ddg_url: str) -> str:
    """Extract the real URL from DuckDuckGo's redirect URL."""
    if "uddg=" in ddg_url:
        parsed = urlparse(ddg_url)
        params = parse_qs(parsed.query)
        if "uddg" in params:
            return unquote(params["uddg"][0])
    return ddg_url


async def search_duckduckgo(query: str, max_results: int = 10) -> List[SearchResult]:
    """Search DuckDuckGo via its lite HTML interface (no API key needed)."""
    results = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        resp = await client.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; Sibyl/1.0)"},
        )
        if resp.status_code != 200:
            return results

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        # DuckDuckGo lite: links have class "result-link", snippets are in next table row
        links = soup.select("a.result-link")

        # Snippets are in <td> elements with class "result-snippet"
        snippet_tds = soup.select("td.result-snippet")

        for i, link in enumerate(links[:max_results]):
            raw_href = link.get("href", "")
            url = _extract_real_url(raw_href)
            title = link.get_text(strip=True)
            snippet = snippet_tds[i].get_text(strip=True) if i < len(snippet_tds) else ""

            if url and title and not url.startswith("//duckduckgo"):
                results.append(SearchResult(title=title, url=url, snippet=snippet))

    return results


async def search_web(query: str, engine: str = "duckduckgo", max_results: int = 10) -> List[SearchResult]:
    """Search the web using the configured engine."""
    if engine == "duckduckgo":
        return await search_duckduckgo(query, max_results)
    return await search_duckduckgo(query, max_results)
