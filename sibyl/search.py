"""Web search — multiple engines (all free, no API keys needed)."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import List, Optional
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


async def search_duckduckgo(
    query: str, max_results: int = 10, client: Optional[httpx.AsyncClient] = None,
) -> List[SearchResult]:
    results = []
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    try:
        resp = await client.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers=_HEADERS,
            timeout=10.0,
        )
        if resp.status_code != 200:
            return results

        soup = BeautifulSoup(resp.text, "lxml")
        links = soup.select("a.result-link")
        snippets = soup.select("td.result-snippet")

        for i, link in enumerate(links[:max_results]):
            raw_href = link.get("href", "")
            url = _extract_ddg_url(raw_href)
            title = link.get_text(strip=True)
            snippet = snippets[i].get_text(strip=True) if i < len(snippets) else ""
            if url and title and not url.startswith("//duckduckgo"):
                results.append(SearchResult(title=title, url=url, snippet=snippet, source="web"))
    finally:
        if own_client:
            await client.aclose()
    return results


# ── Mojeek (independent keyless index — failover when DDG is blocked) ──

async def search_mojeek(
    query: str, max_results: int = 10, client: Optional[httpx.AsyncClient] = None,
) -> List[SearchResult]:
    """Scrape Mojeek — an independent search index (its own crawler), keyless and
    scraper-tolerant. Independence matters: it does not fail in correlation with
    DuckDuckGo, so it's a real failover rather than a second view of the same
    provider."""
    results = []
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    try:
        resp = await client.get(
            "https://www.mojeek.com/search",
            params={"q": query},
            headers=_HEADERS,
            timeout=10.0,
        )
        if resp.status_code != 200:
            return results

        soup = BeautifulSoup(resp.text, "lxml")
        for li in soup.select("ul.results-standard li")[:max_results]:
            h2 = li.find("h2")
            a = h2.find("a") if h2 else li.select_one("a.ob")
            if not a:
                continue
            url = a.get("href", "")
            title = a.get_text(strip=True)
            snippet_el = li.select_one("p.s")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            if url.startswith("http") and title:
                results.append(SearchResult(title=title, url=url, snippet=snippet, source="web"))
    except Exception:
        pass
    finally:
        if own_client:
            await client.aclose()
    return results


async def _search_general_web(
    query: str, max_results: int = 10, client: Optional[httpx.AsyncClient] = None,
) -> List[SearchResult]:
    """General-web search with failover: DuckDuckGo first (its lite HTML endpoint
    is increasingly CAPTCHA/rate-limit gated in 2026); if it returns nothing — or
    raises (a timeout/connection reset is a common block symptom) — fall over to
    Mojeek's independent index. Keeps the keyless value while surviving a DDG
    block."""
    try:
        results = await search_duckduckgo(query, max_results, client=client)
    except Exception:
        results = []
    if not results:
        results = await search_mojeek(query, max_results, client=client)
    return results


# ── Google News (via RSS) ─────────────────────────────────────────

async def search_google_news(
    query: str, max_results: int = 8, client: Optional[httpx.AsyncClient] = None,
) -> List[SearchResult]:
    """Search Google News via RSS feed (free, no API key)."""
    results = []
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    try:
        resp = await client.get(
            f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en",
            headers=_HEADERS,
            timeout=10.0,
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
                    snippet=BeautifulSoup(desc.get_text(), "lxml").get_text(strip=True)[:200] if desc else "",
                    source="news",
                ))
    except Exception:
        pass
    finally:
        if own_client:
            await client.aclose()
    return results


# ── Reddit (via JSON API) ────────────────────────────────────────

async def search_reddit(
    query: str, max_results: int = 5, client: Optional[httpx.AsyncClient] = None,
) -> List[SearchResult]:
    """Search Reddit via its public JSON API (no API key needed)."""
    results = []
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    try:
        resp = await client.get(
            "https://www.reddit.com/search.json",
            params={"q": query, "sort": "relevance", "limit": max_results},
            headers={**_HEADERS, "User-Agent": "Sibyl/1.0 research agent"},
            timeout=10.0,
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
    finally:
        if own_client:
            await client.aclose()
    return results


# ── Wikipedia ─────────────────────────────────────────────────────

async def search_wikipedia(
    query: str, max_results: int = 3, client: Optional[httpx.AsyncClient] = None,
) -> List[SearchResult]:
    """Search Wikipedia via its free API."""
    results = []
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    try:
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
            timeout=10.0,
        )
        if resp.status_code != 200:
            return results

        data = resp.json()
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = BeautifulSoup(item.get("snippet", ""), "lxml").get_text(strip=True)
            url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
            results.append(SearchResult(
                title=f"[Wikipedia] {title}",
                url=url,
                snippet=snippet,
                source="wikipedia",
            ))
    except Exception:
        pass
    finally:
        if own_client:
            await client.aclose()
    return results


_WIKI_URL_RE = re.compile(r"https?://([a-z]{2,3})\.(?:m\.)?wikipedia\.org/wiki/(.+)", re.I)


async def fetch_wikipedia_extract(
    url: str, client: Optional[httpx.AsyncClient] = None,
) -> Optional[str]:
    """Full clean plain-text of a Wikipedia article via the API — bypasses HTML
    scraping, which truncates long articles before infoboxes / tail sections. The
    API returns the complete article text (all sections). Returns None for a
    non-Wikipedia URL or on any error, so the caller keeps its scraped version."""
    m = _WIKI_URL_RE.match(url or "")
    if not m:
        return None
    lang, title = m.group(1).lower(), unquote(m.group(2).split("#")[0]).replace("_", " ")
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=12.0)
    try:
        resp = await client.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={
                "action": "query", "prop": "extracts", "explaintext": "1",
                "redirects": "1", "titles": title, "format": "json",
            },
            headers=_HEADERS, timeout=12.0,
        )
        if resp.status_code != 200:
            return None
        for page in resp.json().get("query", {}).get("pages", {}).values():
            extract = page.get("extract")
            if extract and len(extract) > 200:
                return extract
    except Exception:
        return None
    finally:
        if own_client:
            await client.aclose()
    return None


# ── Semantic Scholar (academic papers) ────────────────────────────

async def search_semantic_scholar(
    query: str, max_results: int = 5, client: Optional[httpx.AsyncClient] = None,
) -> List[SearchResult]:
    """Search academic papers via Semantic Scholar API (free, no key needed)."""
    results = []
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    try:
        resp = None
        # Academic is a nice-to-have that runs alongside the fast engines and
        # gates the search phase, so back off cheaply rather than the old
        # 2s+4s and skip if still rate-limited.
        for attempt in range(2):
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": max_results,
                    "fields": "title,abstract,year,citationCount,url",
                },
                headers=_HEADERS,
                timeout=6.0,
            )
            if resp.status_code == 429 and attempt == 0:
                await asyncio.sleep(1.2)
                continue
            break
        if resp is None or resp.status_code != 200:
            return results

        data = resp.json()
        for paper in data.get("data", []):
            title = paper.get("title", "")
            url = paper.get("url", "")
            abstract = paper.get("abstract", "") or ""
            year = paper.get("year", "")
            citations = paper.get("citationCount", 0)
            if title and url:
                results.append(SearchResult(
                    title=f"[Paper, {year}] {title} ({citations} citations)",
                    url=url,
                    snippet=abstract[:200],
                    source="academic",
                ))
    except Exception:
        pass
    finally:
        if own_client:
            await client.aclose()
    return results


# ── Unified search ────────────────────────────────────────────────

async def search_web(
    query: str,
    engine: str = "all",
    max_results: int = 10,
    client: Optional[httpx.AsyncClient] = None,
    include_academic: bool = False,
) -> List[SearchResult]:
    """Search across multiple sources over a shared, pooled client.

    ``include_academic`` is opt-in: Semantic Scholar is aggressively
    rate-limited, so callers that fan out many queries should enable it on
    only a couple of them rather than every call.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)

    try:
        if engine == "duckduckgo":
            return await _search_general_web(query, max_results, client=client)

        # "all" — search every engine concurrently. General web uses a
        # DDG→Mojeek failover so a DuckDuckGo block doesn't wipe out web results.
        tasks = [
            _search_general_web(query, max_results, client=client),
            search_google_news(query, min(max_results, 5), client=client),
            search_reddit(query, min(max_results, 3), client=client),
            search_wikipedia(query, 2, client=client),
        ]
        if include_academic:
            tasks.append(search_semantic_scholar(query, min(max_results, 3), client=client))

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for res in results_lists:
            if isinstance(res, list):
                all_results.extend(res)
        return all_results
    finally:
        if own_client:
            await client.aclose()
