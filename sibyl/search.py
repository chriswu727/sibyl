"""Web search — multiple engines (all free, no API keys needed)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
from urllib.parse import unquote, urlparse, parse_qs, quote_plus

import httpx
from bs4 import BeautifulSoup


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "web"  # web, news, reddit, wikipedia


_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Sibyl/1.0)"}


# ── DuckDuckGo ────────────────────────────────────────────────────

def _extract_ddg_url(ddg_url: str) -> str:
    if "uddg=" in ddg_url:
        parsed = urlparse(ddg_url)
        params = parse_qs(parsed.query)
        if "uddg" in params:
            return unquote(params["uddg"][0])
    return ddg_url


async def search_duckduckgo(query: str, max_results: int = 10) -> List[SearchResult]:
    results = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        resp = await client.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers=_HEADERS,
        )
        if resp.status_code != 200:
            return results

        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.select("a.result-link")
        snippets = soup.select("td.result-snippet")

        for i, link in enumerate(links[:max_results]):
            raw_href = link.get("href", "")
            url = _extract_ddg_url(raw_href)
            title = link.get_text(strip=True)
            snippet = snippets[i].get_text(strip=True) if i < len(snippets) else ""
            if url and title and not url.startswith("//duckduckgo"):
                results.append(SearchResult(title=title, url=url, snippet=snippet, source="web"))
    return results


# ── Google News (via RSS) ─────────────────────────────────────────

async def search_google_news(query: str, max_results: int = 8) -> List[SearchResult]:
    """Search Google News via RSS feed (free, no API key)."""
    results = []
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            resp = await client.get(
                f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en",
                headers=_HEADERS,
            )
            if resp.status_code != 200:
                return results

            soup = BeautifulSoup(resp.text, "xml")
            items = soup.find_all("item")

            for item in items[:max_results]:
                title = item.find("title")
                link = item.find("link")
                desc = item.find("description")
                if title and link:
                    results.append(SearchResult(
                        title=title.get_text(strip=True),
                        url=link.get_text(strip=True),
                        snippet=BeautifulSoup(desc.get_text(), "html.parser").get_text(strip=True)[:200] if desc else "",
                        source="news",
                    ))
    except Exception:
        pass
    return results


# ── Reddit (via JSON API) ────────────────────────────────────────

async def search_reddit(query: str, max_results: int = 5) -> List[SearchResult]:
    """Search Reddit via its public JSON API (no API key needed)."""
    results = []
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            resp = await client.get(
                "https://www.reddit.com/search.json",
                params={"q": query, "sort": "relevance", "limit": max_results},
                headers={**_HEADERS, "User-Agent": "Sibyl/1.0 research agent"},
            )
            if resp.status_code != 200:
                return results

            data = resp.json()
            for post in data.get("data", {}).get("children", [])[:max_results]:
                d = post.get("data", {})
                title = d.get("title", "")
                url = f"https://reddit.com{d.get('permalink', '')}"
                snippet = d.get("selftext", "")[:200]
                subreddit = d.get("subreddit", "")
                score = d.get("score", 0)
                if title:
                    results.append(SearchResult(
                        title=f"[r/{subreddit}] {title} ({score} upvotes)",
                        url=url,
                        snippet=snippet,
                        source="reddit",
                    ))
    except Exception:
        pass
    return results


# ── Wikipedia ─────────────────────────────────────────────────────

async def search_wikipedia(query: str, max_results: int = 3) -> List[SearchResult]:
    """Search Wikipedia via its free API."""
    results = []
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": max_results,
                },
                headers=_HEADERS,
            )
            if resp.status_code != 200:
                return results

            data = resp.json()
            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                snippet = BeautifulSoup(item.get("snippet", ""), "html.parser").get_text(strip=True)
                url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
                results.append(SearchResult(
                    title=f"[Wikipedia] {title}",
                    url=url,
                    snippet=snippet,
                    source="wikipedia",
                ))
    except Exception:
        pass
    return results


# ── Unified search ────────────────────────────────────────────────

async def search_web(
    query: str,
    engine: str = "all",
    max_results: int = 10,
) -> List[SearchResult]:
    """Search across multiple sources."""
    if engine == "duckduckgo":
        return await search_duckduckgo(query, max_results)

    # "all" — search all sources in parallel
    import asyncio
    tasks = [
        search_duckduckgo(query, max_results),
        search_google_news(query, min(max_results, 5)),
        search_reddit(query, min(max_results, 3)),
        search_wikipedia(query, 2),
    ]
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = []
    for res in results_lists:
        if isinstance(res, list):
            all_results.extend(res)

    return all_results
