"""Web scraper — extract clean text content from URLs with anti-block techniques."""
from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from typing import List, Mapping, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from .url_safety import validate_public_url


@dataclass
class WebPage:
    url: str
    title: str
    text: str
    error: Optional[str] = None


@dataclass(frozen=True)
class _ResponseSnapshot:
    status_code: int
    headers: Mapping[str, str]
    text: str


class ResponseTooLargeError(Exception):
    pass


_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


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


async def _bounded_get(
    client,
    url: str,
    *,
    headers: dict,
    timeout: float,
    max_response_bytes: int = _MAX_RESPONSE_BYTES,
) -> _ResponseSnapshot:
    async with client.stream(
        "GET",
        url,
        headers=headers,
        timeout=timeout,
        follow_redirects=False,
    ) as response:
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_response_bytes:
                    raise ResponseTooLargeError(
                        f"Response exceeds {max_response_bytes} bytes"
                    )
            except ValueError:
                pass

        content = bytearray()
        async for chunk in response.aiter_bytes():
            content.extend(chunk)
            if len(content) > max_response_bytes:
                raise ResponseTooLargeError(
                    f"Response exceeds {max_response_bytes} bytes"
                )
        encoding = response.encoding or "utf-8"
        return _ResponseSnapshot(
            status_code=response.status_code,
            headers=dict(response.headers),
            text=bytes(content).decode(encoding, errors="replace"),
        )


def _clean_lines(text: str, max_chars: int) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) > 5:  # Skip very short lines (likely UI elements)
            lines.append(line)
    return "\n".join(lines)[:max_chars]


def _bs4_body_text(soup: BeautifulSoup, max_chars: int) -> str:
    """Legacy BeautifulSoup body extraction (noise-stripped soup)."""
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
    return _clean_lines(text, max_chars)


def _extract_content(html: str, url: str, max_chars: int, extractor: str = "bs4") -> WebPage:
    """Parse HTML and extract clean text. CPU-bound — run off the event loop.

    ``extractor='trafilatura'`` runs trafilatura (favor_recall) and takes the
    LONGER of trafilatura vs the BeautifulSoup body — a length-comparison
    fallback, so a short-but-non-empty trafilatura result can never shrink the
    usable text below what bs4 would have extracted. Title always comes from the
    cheap <title> tag.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                      "noscript", "iframe", "form", "button"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    bs4_text = _bs4_body_text(soup, max_chars)
    text = bs4_text

    if extractor == "trafilatura":
        try:
            import trafilatura
            traf = trafilatura.extract(
                html, favor_recall=True, output_format="txt",
                include_comments=False, include_tables=True,
            ) or ""
            traf_text = _clean_lines(traf, max_chars)
            if len(traf_text) > len(bs4_text):
                text = traf_text
        except Exception:
            pass  # any trafilatura failure keeps the bs4 result

    return WebPage(url=url, title=title, text=text)


class _JinaGate:
    """Bounds JS-render (r.jina.ai) calls so a thin-content sweep can't stall the
    batch or trip the keyless rate limit: ≤2 concurrent, and a min-interval
    between calls (keyless only; 0 when JINA_API_KEY is set). Created per
    scrape_urls batch, so the semaphore binds to the running loop."""
    def __init__(self):
        self.sem = asyncio.Semaphore(2)
        self.last = 0.0
        self.min_interval = 0.0 if os.environ.get("JINA_API_KEY") else 3.0

    async def render(self, url, max_chars, client):
        import time
        async with self.sem:
            wait = self.min_interval - (time.monotonic() - self.last)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last = time.monotonic()
            return await _try_jina(url, max_chars, client)


def _is_html_ish(resp) -> bool:
    ct = (resp.headers.get("content-type", "") or "").lower()
    return (not ct) or ("html" in ct) or ("text" in ct)


async def scrape_url(
    url: str,
    max_chars: int = 6000,
    client: Optional[httpx.AsyncClient] = None,
    extractor: str = "bs4",
    jina_fallback: bool = False,
    js_render: bool = False,
    js_render_threshold: int = 500,
    jina_gate: Optional["_JinaGate"] = None,
) -> WebPage:
    """Fetch a URL with retry and anti-block techniques.

    Reuses a shared ``client`` when provided so connections/TLS are pooled
    across a batch; otherwise creates a short-lived one.
    """
    unsafe_reason = await validate_public_url(url)
    if unsafe_reason:
        return WebPage(url=url, title="", text="", error=f"Unsafe URL: {unsafe_reason}")

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=12.0)

    async def _render_if_thin(page: WebPage, resp) -> WebPage:
        # A 200 that extracts thin is usually a JS shell — render it via Jina and
        # keep the longer text (keyless; never returns less than we already have).
        if not (js_render and len(page.text) < js_render_threshold and _is_html_ish(resp)):
            return page
        rendered = (await jina_gate.render(url, max_chars, client)) if jina_gate \
            else await _try_jina(url, max_chars, client)
        if rendered and len(rendered.text) > len(page.text):
            return rendered
        return page

    async def _get_with_safe_redirects(headers):
        current_url = url
        for redirect_count in range(6):
            response = await _bounded_get(
                client,
                current_url,
                headers=headers,
                timeout=8.0,
            )
            if response.status_code not in {301, 302, 303, 307, 308}:
                return response, current_url, ""
            if redirect_count == 5:
                return response, current_url, "Too many redirects"
            location = response.headers.get("location", "")
            if not location:
                return response, current_url, "Redirect response is missing a Location header"
            redirect_url = urljoin(current_url, location)
            unsafe_redirect = await validate_public_url(redirect_url)
            if unsafe_redirect:
                return response, current_url, f"Unsafe redirect URL: {unsafe_redirect}"
            current_url = redirect_url
        raise AssertionError("redirect loop exceeded its bound")

    try:
        for attempt in range(2):
            try:
                # Rotate User-Agent per request (shared client → set on request)
                resp, final_url, redirect_error = await _get_with_safe_redirects(
                    _get_headers()
                )
                if redirect_error:
                    return WebPage(url=url, title="", text="", error=redirect_error)

                if resp.status_code == 200:
                    page = await asyncio.to_thread(
                        _extract_content,
                        resp.text,
                        final_url,
                        max_chars,
                        extractor,
                    )
                    return await _render_if_thin(page, resp)

                # Retry 403/429 once with a different User-Agent (may clear).
                if resp.status_code in (403, 429) and attempt == 0:
                    continue

                # On a hard block we won't (further) retry — 401/451 don't
                # benefit from a UA swap, 403/429 have already been retried —
                # optionally recover via Jina Reader (opt-in).
                if resp.status_code in (401, 403, 429, 451) and jina_fallback:
                    jina_page = await _try_jina(url, max_chars, client)
                    if jina_page and jina_page.text:
                        return jina_page

                return WebPage(url=url, title="", text="", error=f"HTTP {resp.status_code}")

            except httpx.TimeoutException:
                # A slow site won't get faster on a second try with the same
                # timeout — fail fast so it doesn't stall the whole scrape batch.
                return WebPage(url=url, title="", text="", error="timeout")
            except ResponseTooLargeError as exc:
                return WebPage(url=url, title="", text="", error=str(exc))
            except Exception as e:
                if attempt == 0:
                    continue
                return WebPage(url=url, title="", text="", error=str(e)[:200])

        return WebPage(url=url, title="", text="", error="All attempts failed")
    finally:
        if own_client:
            await client.aclose()


async def _try_jina(
    url: str,
    max_chars: int,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[WebPage]:
    """Fallback via Jina Reader (r.jina.ai): server-side renders JS and returns
    clean markdown for pages our direct fetch couldn't get (JS/thin/403 — not
    hard auth-walls). Uses JINA_API_KEY when set (the keyless tier's ~20 RPM
    trips instantly under concurrency). Timeout is capped BELOW the primary 8s
    scrape timeout so this branch can't become the slowest path in the batch."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=7.0)
    try:
        headers = {"X-Return-Format": "markdown", "Accept": "text/plain"}
        key = os.environ.get("JINA_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resp = await _bounded_get(
            client,
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=7.0,
        )
        if resp.status_code == 200:
            # Jina already returns clean markdown — no HTML parsing needed.
            text = "\n".join(l.strip() for l in resp.text.splitlines() if len(l.strip()) > 5)[:max_chars]
            if text and len(text) > 100:
                title = url
                for line in resp.text.splitlines():
                    if line.startswith("Title:"):
                        title = line[6:].strip()
                        break
                return WebPage(url=url, title=title, text=text)
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
    extractor: str = "bs4",
    jina_fallback: bool = False,
    js_render: bool = False,
    js_render_threshold: int = 500,
) -> List[WebPage]:
    """Scrape multiple URLs concurrently over a single pooled client."""
    semaphore = asyncio.Semaphore(concurrency)
    jina_gate = _JinaGate() if js_render else None

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
            return await scrape_url(url, max_chars, client=client, extractor=extractor,
                                    jina_fallback=jina_fallback, js_render=js_render,
                                    js_render_threshold=js_render_threshold, jina_gate=jina_gate)

    try:
        return await asyncio.gather(*[_limited_scrape(url) for url in urls])
    finally:
        if own_client:
            await client.aclose()
