"""Web search across keyless and optional production providers."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
import weakref
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import quote, unquote, urlparse, parse_qs, quote_plus

import httpx
from bs4 import BeautifulSoup


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "web"  # web, news, reddit, wikipedia
    provider: str = ""


_HEADERS = {
    "User-Agent": "Sibyl/0.4 (+https://github.com/chriswu727/sibyl)"
}
_SEARCH_BATCH_TIMEOUT_SECONDS = 8.0
_SEARCH_PROVIDERS = {"keyless", "tavily"}
_logger = logging.getLogger(__name__)


class _ProviderGate:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.last_started: Optional[float] = None

    async def wait(self, min_interval: float) -> None:
        async with self.lock:
            now = time.monotonic()
            if self.last_started is not None:
                delay = min_interval - (now - self.last_started)
                if delay > 0:
                    await asyncio.sleep(delay)
                    now = time.monotonic()
            self.last_started = now


_PROVIDER_GATES = weakref.WeakKeyDictionary()
_PROVIDER_GATES_LOCK = threading.Lock()


def _get_provider_gate(provider: str) -> _ProviderGate:
    loop = asyncio.get_running_loop()
    with _PROVIDER_GATES_LOCK:
        gates = _PROVIDER_GATES.setdefault(loop, {})
        return gates.setdefault(provider, _ProviderGate())


async def _paced_get(provider, min_interval, client, url, **kwargs):
    await _get_provider_gate(provider).wait(min_interval)
    return await client.get(url, **kwargs)


async def _serialized_get(provider, min_interval, client, url, **kwargs):
    gate = _get_provider_gate(provider)
    async with gate.lock:
        now = time.monotonic()
        if gate.last_started is not None:
            delay = min_interval - (now - gate.last_started)
            if delay > 0:
                await asyncio.sleep(delay)
                now = time.monotonic()
        gate.last_started = now
        return await client.get(url, **kwargs)


def _configured_search_provider() -> str:
    provider = os.environ.get("SIBYL_SEARCH_PROVIDER", "keyless").strip().lower()
    if provider not in _SEARCH_PROVIDERS:
        choices = ", ".join(sorted(_SEARCH_PROVIDERS))
        raise ValueError(f"SIBYL_SEARCH_PROVIDER must be one of: {choices}.")
    if provider == "tavily" and not os.environ.get("TAVILY_API_KEY", "").strip():
        raise ValueError(
            "SIBYL_SEARCH_PROVIDER=tavily requires TAVILY_API_KEY."
        )
    return provider


async def search_tavily(
    query: str,
    max_results: int = 10,
    client: Optional[httpx.AsyncClient] = None,
    api_key: str = "",
) -> List[SearchResult]:
    key = api_key.strip() or os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        raise ValueError("Tavily search requires TAVILY_API_KEY.")

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    try:
        await _get_provider_gate("tavily").wait(0.1)
        response = await client.post(
            "https://api.tavily.com/search",
            headers={
                **_HEADERS,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": max(1, min(20, max_results)),
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
            timeout=10.0,
        )
        response.raise_for_status()
        results = []
        for item in response.json().get("results", []):
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
            if title and url.startswith("http"):
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=content,
                        source="web",
                        provider="tavily",
                    )
                )
        return results[:max_results]
    finally:
        if own_client:
            await client.aclose()


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
        resp = await _paced_get(
            "duckduckgo",
            1.0,
            client,
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=_HEADERS,
            timeout=10.0,
        )
        if resp.status_code != 200:
            return results

        soup = BeautifulSoup(resp.text, "lxml")
        links = soup.select("a.result__a, a.result-link")
        snippets = soup.select(".result__snippet, td.result-snippet")

        for i, link in enumerate(links[:max_results]):
            raw_href = link.get("href", "")
            url = _extract_ddg_url(raw_href)
            title = link.get_text(strip=True)
            snippet = snippets[i].get_text(strip=True) if i < len(snippets) else ""
            if url and title and not url.startswith("//duckduckgo"):
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="web",
                        provider="duckduckgo",
                    )
                )
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
        resp = await _paced_get(
            "mojeek",
            1.0,
            client,
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
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="web",
                        provider="mojeek",
                    )
                )
    except Exception:
        pass
    finally:
        if own_client:
            await client.aclose()
    return results


# ── Yahoo (third keyless general-web path) ───────────────────────

def _extract_yahoo_url(yahoo_url: str) -> str:
    marker = "/RU="
    if marker not in yahoo_url:
        return yahoo_url
    encoded = yahoo_url.split(marker, 1)[1].split("/RK=", 1)[0]
    return unquote(encoded)


async def search_yahoo(
    query: str, max_results: int = 10, client: Optional[httpx.AsyncClient] = None,
) -> List[SearchResult]:
    results = []
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    try:
        resp = await _paced_get(
            "yahoo",
            1.0,
            client,
            "https://search.yahoo.com/search",
            params={"q": query},
            headers=_HEADERS,
            timeout=10.0,
        )
        if resp.status_code != 200:
            return results

        soup = BeautifulSoup(resp.text, "lxml")
        for item in soup.select(".algo")[:max_results]:
            link = item.select_one(".compTitle a[href]")
            if link is None:
                continue
            url = _extract_yahoo_url(link.get("href", ""))
            heading = item.select_one(".compTitle h3")
            title = (heading or link).get_text(" ", strip=True)
            snippet_element = item.select_one(".compText")
            snippet = (
                snippet_element.get_text(" ", strip=True)
                if snippet_element is not None
                else ""
            )
            if url.startswith("http") and title:
                results.append(SearchResult(title, url, snippet, "web", "yahoo"))
    except Exception:
        pass
    finally:
        if own_client:
            await client.aclose()
    return results


async def _search_general_web(
    query: str, max_results: int = 10, client: Optional[httpx.AsyncClient] = None,
) -> List[SearchResult]:
    """Search DuckDuckGo HTML, then fail over to Mojeek and Yahoo."""
    async def search_with_failover():
        try:
            results = await search_duckduckgo(query, max_results, client=client)
        except Exception:
            results = []
        if not results:
            results = await search_mojeek(query, max_results, client=client)
        if not results:
            results = await search_yahoo(query, max_results, client=client)
        return results

    try:
        return await asyncio.wait_for(search_with_failover(), timeout=9.0)
    except asyncio.TimeoutError:
        return []


async def _search_configured_general_web(
    query: str,
    max_results: int,
    provider: str,
    client: Optional[httpx.AsyncClient],
) -> List[SearchResult]:
    if provider == "keyless":
        return await _search_general_web(query, max_results, client=client)

    try:
        results = await search_tavily(query, max_results, client=client)
    except Exception as exc:
        _logger.warning(
            "Tavily search failed (%s); falling back to keyless search.",
            type(exc).__name__,
        )
        return await _search_general_web(query, max_results, client=client)
    if results:
        return results
    _logger.warning("Tavily returned no results; falling back to keyless search.")
    return await _search_general_web(query, max_results, client=client)


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
        resp = await _paced_get(
            "google_news",
            0.25,
            client,
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
                    provider="google_news",
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
        resp = await _paced_get(
            "reddit",
            0.5,
            client,
            "https://www.reddit.com/search.json",
            params={"q": query, "sort": "relevance", "limit": max_results},
            headers=_HEADERS,
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
                    provider="reddit",
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
        resp = await _paced_get(
            "wikipedia",
            0.1,
            client,
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
                provider="wikipedia",
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
        resp = await _paced_get(
            "wikipedia",
            0.1,
            client,
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


async def wikipedia_lookup(
    query: str, client: Optional[httpx.AsyncClient] = None, max_pages: int = 2,
) -> List["WebPage"]:
    """Find the best-matching Wikipedia article(s) for a query via the opensearch
    API and return their full-text extracts as WebPages. A robust encyclopedic
    fallback when general web search is thin or rate-limited — opensearch matches
    titles even from a partial/entity query."""
    from .scraper import WebPage
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=12.0)
    try:
        resp = await _paced_get(
            "wikipedia",
            0.1,
            client,
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": max_pages,
                    "namespace": "0", "format": "json"},
            headers=_HEADERS, timeout=12.0,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        titles = data[1] if len(data) > 1 else []
        urls = data[3] if len(data) > 3 else []
        extracts = await asyncio.gather(
            *[fetch_wikipedia_extract(u, client) for u in urls], return_exceptions=True)
        pages = []
        for title, url, ex in zip(titles, urls, extracts):
            if isinstance(ex, str) and ex:
                pages.append(
                    WebPage(
                        url=url,
                        title=f"[Wikipedia] {title}",
                        text=ex,
                        content_origin="wikipedia_api",
                    )
                )
        return pages
    except Exception:
        return []
    finally:
        if own_client:
            await client.aclose()


# ── Crossref (academic metadata) ──────────────────────────────

def _crossref_date_text(item) -> str:
    for field, label in (
        ("published-online", "Published online"),
        ("published-print", "Published in print"),
        ("published", "Published"),
        ("issued", "Issued"),
    ):
        date_parts = (item.get(field) or {}).get("date-parts") or []
        if not date_parts or not date_parts[0]:
            continue
        parts = date_parts[0]
        try:
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 else 0
            day = int(parts[2]) if len(parts) > 2 else 0
        except (TypeError, ValueError):
            continue
        if not 1 <= year <= 9999:
            continue
        if not month:
            return f"{label}: {year}."
        if not 1 <= month <= 12:
            continue
        month_name = (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        )[month - 1]
        if not day:
            return f"{label}: {month_name} {year} ({year:04d}-{month:02d})."
        if not 1 <= day <= 31:
            continue
        suffix = (
            "th"
            if 10 < day % 100 < 14
            else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        )
        return (
            f"{label}: {month_name} {day}, {year}; "
            f"{day}{suffix} {month_name}, {year}; "
            f"{year:04d}-{month:02d}-{day:02d}."
        )
    return ""


def _crossref_title_matches(query: str, title: str) -> bool:
    query_text = " ".join(re.findall(r"[^\W_]+", query.casefold()))
    title_text = " ".join(re.findall(r"[^\W_]+", title.casefold()))
    if not query_text or not title_text:
        return False
    if title_text in query_text:
        return True
    query_terms = set(query_text.split())
    title_terms = set(title_text.split())
    if not title_terms:
        return False
    overlap = len(query_terms & title_terms)
    precision = overlap / len(title_terms)
    recall = overlap / len(query_terms)
    return precision >= 0.9 and recall >= 0.9


async def search_crossref(
    query: str,
    max_results: int = 3,
    client: Optional[httpx.AsyncClient] = None,
) -> List[SearchResult]:
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    try:
        params = {
            "query.bibliographic": query,
            "rows": max(1, min(5, max_results)),
        }
        mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        if mailto:
            params["mailto"] = mailto
        response = await _serialized_get(
            "crossref",
            0.25,
            client,
            "https://api.crossref.org/v1/works",
            params=params,
            headers=_HEADERS,
            timeout=8.0,
        )
        if response.status_code != 200:
            return []

        results = []
        for item in response.json().get("message", {}).get("items", []):
            titles = item.get("title") or []
            title = str(titles[0]).strip() if titles else ""
            doi = str(item.get("DOI") or "").strip()
            if not title or not doi or not _crossref_title_matches(query, title):
                continue
            containers = item.get("container-title") or []
            container = str(containers[0]).strip() if containers else ""
            authors = []
            for author in (item.get("author") or [])[:5]:
                name = " ".join(
                    part for part in (
                        str(author.get("given") or "").strip(),
                        str(author.get("family") or "").strip(),
                    ) if part
                )
                if name:
                    authors.append(name)
            details = [title, f"DOI: {doi}.", _crossref_date_text(item)]
            if container:
                details.append(f"Published by {container}.")
            if authors:
                details.append(f"Authors: {', '.join(authors)}.")
            results.append(
                SearchResult(
                    title=f"[Crossref] {title}",
                    url=f"https://api.crossref.org/v1/works/{quote(doi, safe='/')}",
                    snippet=" ".join(part for part in details if part),
                    source="academic",
                    provider="crossref",
                )
            )
        return results[:max_results]
    except Exception:
        return []
    finally:
        if own_client:
            await client.aclose()


# ── Semantic Scholar (academic papers) ───────────────────────────

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
            resp = await _paced_get(
                "semantic_scholar",
                1.0,
                client,
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
                    provider="semantic_scholar",
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

        search_provider = _configured_search_provider()

        # "all" — search every engine concurrently. General web uses an
        # independent fallback chain so one blocked endpoint does not erase web results.
        coroutines = [
            _search_configured_general_web(
                query, max_results, search_provider, client
            ),
            search_google_news(query, min(max_results, 5), client=client),
            search_reddit(query, min(max_results, 3), client=client),
            search_wikipedia(query, 2, client=client),
        ]
        if include_academic:
            coroutines.append(
                search_semantic_scholar(query, min(max_results, 3), client=client)
            )
            coroutines.append(
                search_crossref(query, min(max_results, 3), client=client)
            )

        tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
        done, pending = await asyncio.wait(
            tasks, timeout=_SEARCH_BATCH_TIMEOUT_SECONDS
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        results_lists = []
        for task in tasks:
            if task not in done or task.cancelled():
                results_lists.append([])
                continue
            try:
                results_lists.append(task.result())
            except Exception:
                results_lists.append([])

        all_results = []
        for res in results_lists:
            if isinstance(res, list):
                all_results.extend(res)
        return all_results
    finally:
        if own_client:
            await client.aclose()
