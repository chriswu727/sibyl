"""Web scraper — extract clean text content from URLs with anti-block techniques."""
from __future__ import annotations

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
    """Parse HTML and extract clean text."""
    soup = BeautifulSoup(html, "html.parser")

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


async def scrape_url(url: str, max_chars: int = 6000) -> WebPage:
    """Fetch a URL with retry and anti-block techniques."""
    # Skip non-HTTP URLs
    if not url.startswith("http"):
        return WebPage(url=url, title="", text="", error="Invalid URL")

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=12.0,
                headers=_get_headers(),
            ) as client:
                resp = await client.get(url)

                if resp.status_code == 200:
                    return _extract_content(resp.text, url, max_chars)

                # Retry on 403/429 with different User-Agent
                if resp.status_code in (403, 429) and attempt == 0:
                    continue

                # Try Google Cache as fallback on 403
                if resp.status_code == 403 and attempt == 1:
                    cache_page = await _try_google_cache(url, max_chars)
                    if cache_page and cache_page.text:
                        return cache_page

                return WebPage(url=url, title="", text="", error=f"HTTP {resp.status_code}")

        except Exception as e:
            if attempt == 0:
                continue
            return WebPage(url=url, title="", text="", error=str(e)[:200])

    return WebPage(url=url, title="", text="", error="All attempts failed")


async def _try_google_cache(url: str, max_chars: int) -> Optional[WebPage]:
    """Try fetching from Google's web cache."""
    try:
        cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}"
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
            headers=_get_headers(),
        ) as client:
            resp = await client.get(cache_url)
            if resp.status_code == 200:
                page = _extract_content(resp.text, url, max_chars)
                if page.text and len(page.text) > 100:
                    return page
    except Exception:
        pass
    return None


async def scrape_urls(urls: List[str], max_chars: int = 6000) -> List[WebPage]:
    """Scrape multiple URLs concurrently with rate limiting."""
    import asyncio

    # Limit concurrency to avoid being blocked
    semaphore = asyncio.Semaphore(5)

    async def _limited_scrape(url: str) -> WebPage:
        async with semaphore:
            return await scrape_url(url, max_chars)

    tasks = [_limited_scrape(url) for url in urls]
    return await asyncio.gather(*tasks)
