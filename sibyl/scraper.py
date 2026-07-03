"""Web scraper — extract clean text content from URLs with anti-block techniques."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup


@dataclass
class WebPage:
    url: str
    title: str
    text: str
    error: Optional[str] = None


# Realistic browser User-Agents (rotated)
_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
]


def _get_headers() -> dict:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def _extract_content(html: str, url: str, max_chars: int) -> WebPage:
    """Parse HTML and extract clean text. CPU-bound — run off the event loop."""
    soup = BeautifulSoup(html, "lxml")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                      "noscript", "iframe", "form", "button"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Try multiple content extraction strategies
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", {"role": "main"})
        or soup.find("div", class_=lambda c: c and any(
            x in (c if isinstance(c, str) else " ".join(c))
            for x in ["content", "article", "post", "entry", "body"]
        ))
        or soup.find("body")
    )

    text = main.get_text(separator="\n", strip=True) if main else ""

    # Clean up
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) > 5:  # Skip very short lines (likely UI elements)
            lines.append(line)
    text = "\n".join(lines)[:max_chars]

    return WebPage(url=url, title=title, text=text)


async def scrape_url(
    url: str,
    max_chars: int = 6000,
    client: Optional[httpx.AsyncClient] = None,
) -> WebPage:
    """Fetch a URL with retry and anti-block techniques.

    Reuses a shared ``client`` when provided so connections/TLS are pooled
    across a batch; otherwise creates a short-lived one.
    """
    # Skip non-HTTP URLs
    if not url.startswith("http"):
        return WebPage(url=url, title="", text="", error="Invalid URL")

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=12.0)

    try:
        for attempt in range(2):
            try:
                # Rotate User-Agent per request (shared client → set on request)
                resp = await client.get(url, headers=_get_headers(), timeout=8.0)

                if resp.status_code == 200:
                    return await asyncio.to_thread(_extract_content, resp.text, url, max_chars)

                # Retry on 403/429 with different User-Agent
                if resp.status_code in (403, 429) and attempt == 0:
                    continue

                # Try Google Cache as fallback on 403
                if resp.status_code == 403 and attempt == 1:
                    cache_page = await _try_google_cache(url, max_chars, client)
                    if cache_page and cache_page.text:
                        return cache_page

                return WebPage(url=url, title="", text="", error=f"HTTP {resp.status_code}")

            except httpx.TimeoutException:
                # A slow site won't get faster on a second try with the same
                # timeout — fail fast so it doesn't stall the whole scrape batch.
                return WebPage(url=url, title="", text="", error="timeout")
            except Exception as e:
                if attempt == 0:
                    continue
                return WebPage(url=url, title="", text="", error=str(e)[:200])

        return WebPage(url=url, title="", text="", error="All attempts failed")
    finally:
        if own_client:
            await client.aclose()


async def _try_google_cache(
    url: str,
    max_chars: int,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[WebPage]:
    """Try fetching from Google's web cache."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    try:
        cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}"
        resp = await client.get(cache_url, headers=_get_headers(), timeout=10.0)
        if resp.status_code == 200:
            page = await asyncio.to_thread(_extract_content, resp.text, url, max_chars)
            if page.text and len(page.text) > 100:
                return page
    except Exception:
        pass
    finally:
        if own_client:
            await client.aclose()
    return None


async def scrape_urls(
    urls: List[str],
    max_chars: int = 6000,
    concurrency: int = 8,
    client: Optional[httpx.AsyncClient] = None,
) -> List[WebPage]:
    """Scrape multiple URLs concurrently over a single pooled client."""
    semaphore = asyncio.Semaphore(concurrency)

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=12.0,
            limits=httpx.Limits(max_connections=concurrency * 2,
                                max_keepalive_connections=concurrency),
        )

    async def _limited_scrape(url: str) -> WebPage:
        async with semaphore:
            return await scrape_url(url, max_chars, client=client)

    try:
        return await asyncio.gather(*[_limited_scrape(url) for url in urls])
    finally:
        if own_client:
            await client.aclose()
