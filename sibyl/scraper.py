"""Web scraper — extract clean text content from URLs."""
from __future__ import annotations

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


async def scrape_url(url: str, max_chars: int = 5000) -> WebPage:
    """Fetch a URL and extract clean text content."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Sibyl/1.0)"},
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return WebPage(url=url, title="", text="", error=f"HTTP {resp.status_code}")

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()

            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Extract main content (prefer article/main tags)
            main = soup.find("article") or soup.find("main") or soup.find("body")
            text = main.get_text(separator="\n", strip=True) if main else ""

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)[:max_chars]

            return WebPage(url=url, title=title, text=text)

    except Exception as e:
        return WebPage(url=url, title="", text="", error=str(e)[:200])


async def scrape_urls(urls: List[str], max_chars: int = 5000) -> List[WebPage]:
    """Scrape multiple URLs concurrently."""
    import asyncio
    tasks = [scrape_url(url, max_chars) for url in urls]
    return await asyncio.gather(*tasks)
