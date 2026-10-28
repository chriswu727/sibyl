"""Read a public web page or PDF as plain text over the safe network boundary."""

from __future__ import annotations

import asyncio
import re
import zlib
from dataclasses import dataclass
from typing import List, Mapping, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, Comment, Doctype

from .safe_http import PinnedDNSAsyncClient
from .url_safety import resolve_public_url

MAX_PAGE_CHARS = 400_000


@dataclass
class WebPage:
    url: str
    title: str
    text: str
    error: Optional[str] = None
    source_type: str = "web"
    extraction_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ResponseSnapshot:
    status_code: int
    headers: Mapping[str, str]
    text: str
    content: bytes = b""


class ResponseTooLargeError(Exception):
    pass


_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


# Every request identifies Sibyl, as many publishers (Wikimedia among them) ask
# of automated clients. Sibyl never imitates a browser or switches identity to
# get past a refusal; a refused page stays refused.
USER_AGENT = "Sibyl/0.5 (+https://github.com/chriswu727/sibyl)"


def _get_headers() -> dict:
    return {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
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

        content_encoding = (
            response.headers.get("content-encoding", "identity").lower().strip()
        )
        if content_encoding not in {"identity", "gzip", "deflate"}:
            raise ValueError("Unsupported content encoding")
        decoder = None
        if content_encoding != "identity":
            decoder = zlib.decompressobj(31 if content_encoding == "gzip" else 15)
        content = bytearray()
        wire_size = 0
        # MockTransport may deliver an already-buffered response; real streams stay raw.
        stream = (
            response.aiter_bytes()
            if response.is_stream_consumed
            else response.aiter_raw()
        )
        if response.is_stream_consumed:
            decoder = None
        async for chunk in stream:
            wire_size += len(chunk)
            if wire_size > max_response_bytes:
                raise ResponseTooLargeError(
                    f"Response exceeds {max_response_bytes} bytes"
                )
            remaining = max_response_bytes - len(content)
            decoded = decoder.decompress(chunk, remaining + 1) if decoder else chunk
            if len(decoded) > remaining:
                raise ResponseTooLargeError(
                    f"Response exceeds {max_response_bytes} bytes"
                )
            content.extend(decoded)
        if decoder and not decoder.eof:
            raise ValueError("Truncated compressed response")
        if decoder and decoder.unused_data:
            raise ValueError("Trailing compressed response data")
        encoding = response.encoding or "utf-8"
        raw = bytes(content)
        is_pdf = (
            raw.startswith(b"%PDF-")
            or "application/pdf" in response.headers.get("content-type", "").lower()
        )
        return _ResponseSnapshot(
            status_code=response.status_code,
            headers=dict(response.headers),
            text="" if is_pdf else raw.decode(encoding, errors="replace"),
            content=raw,
        )


def _clean_lines(text: str, max_chars: int) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)[:max_chars]


_CONTENT_SELECTOR = (
    ".model-card, .markdown-body, main, [role=main], article, [itemprop=articleBody]"
)


def _extract_title(soup: BeautifulSoup) -> str:
    candidates = [soup.title.get_text(" ", strip=True) if soup.title else ""]
    for selector in ('meta[property="og:title"]', 'meta[name="twitter:title"]'):
        tag = soup.select_one(selector)
        candidates.append(str(tag.get("content", "")) if tag else "")
    candidates.extend(tag.get_text(" ", strip=True) for tag in soup.find_all("h1"))
    for candidate in candidates:
        title = " ".join(candidate.split())
        if title.casefold() in {
            "",
            "untitled",
            "undefined",
            "null",
            "loading",
            "document",
        }:
            continue
        if re.fullmatch(
            r"[\w:.-]+[-_:](?:empty|placeholder|undefined|null)", title, re.I
        ):
            continue
        return title
    return ""


def _normalize_html_whitespace(soup: BeautifulSoup) -> bool:
    normalized_code = False
    for node in list(soup.find_all(string=True)):
        if isinstance(node, (Comment, Doctype)):
            continue
        text = str(node)
        if node.find_parent(["pre", "code"]):
            cleaned = re.sub(r"[^\S\r\n]{80,}", " ", text)
            normalized_code |= cleaned != text
        else:
            cleaned = re.sub(r"\s+", " ", text)
        if cleaned != text:
            node.replace_with(cleaned)
    return normalized_code


def _bs4_body_text(soup: BeautifulSoup, max_chars: int) -> str:
    candidates = []
    for selector in (
        ".model-card, .markdown-body",
        "main, [role=main]",
        "article, [itemprop=articleBody]",
        "div[class*=content], div[class*=article], div[class*=post], div[class*=entry], div[class*=body]",
        "body",
    ):
        group = soup.select(selector, limit=128)
        if not group:
            continue
        largest = max(group, key=lambda tag: len(tag.get_text(" ", strip=True)))
        candidates.append(largest)
        if len(largest.get_text(" ", strip=True)) >= 200:
            break
    main = (
        max(candidates, key=lambda tag: len(tag.get_text(" ", strip=True)))
        if candidates
        else None
    )
    if main:
        for table in reversed(list(main.find_all("table"))):
            rows = []
            for row in table.find_all("tr"):
                cells = []
                for cell in row.find_all(["td", "th"], recursive=False):
                    value = cell.get_text(" ", strip=True).replace("|", "\\|")
                    spans = [
                        f"{key}={cell[key]}"
                        for key in ("rowspan", "colspan")
                        if cell.get(key)
                    ]
                    cells.append(
                        value + (" [" + ", ".join(spans) + "]" if spans else "")
                    )
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")
            table.replace_with("\n" + "\n".join(rows) + "\n")
    text = main.get_text(separator="\n", strip=True) if main else ""
    return _clean_lines(text, max_chars)


def _extract_content(html: str, url: str, max_chars: int) -> WebPage:
    """Parse HTML and extract clean text. CPU-bound: run off the event loop."""
    soup = BeautifulSoup(html, "lxml")
    title = _extract_title(soup)
    for tag in reversed(
        soup(["script", "style", "noscript", "iframe", "input", "select", "textarea"])
    ):
        tag.decompose()
    # Malformed navigation and page-wide ASP.NET forms can contain the article.
    for tag in reversed(soup(["nav", "footer", "header", "aside", "form"])):
        if tag.select_one(_CONTENT_SELECTOR):
            tag.unwrap()
        else:
            tag.decompose()
    for button in soup.find_all("button"):
        if button.find_parent("table"):
            button.unwrap()
        else:
            button.decompose()

    warnings = []
    if _normalize_html_whitespace(soup):
        warnings.append(
            "Excessive whitespace in preformatted text was normalized; consult the original for exact code formatting."
        )
    text = _bs4_body_text(soup, max_chars + 1)
    if len(text) > max_chars:
        warnings.append(f"HTML text truncated to {max_chars} characters.")
    return WebPage(
        url=url,
        title=title,
        text=text[:max_chars],
        extraction_warnings=tuple(warnings),
    )


def _is_html_ish(resp) -> bool:
    ct = (resp.headers.get("content-type", "") or "").lower()
    return (not ct) or ("html" in ct) or ("text" in ct)


async def scrape_url(
    url: str,
    max_chars: int = MAX_PAGE_CHARS,
    client: Optional[httpx.AsyncClient] = None,
) -> WebPage:
    """Fetch one public URL as text, refusing unsafe destinations and redirects.

    Reuses a provided pinned client; ordinary HTTPX clients are replaced with
    a short-lived pinned client so DNS validation cannot be bypassed.
    """
    unsafe_reason, resolved_addresses = await resolve_public_url(url)
    if unsafe_reason:
        return WebPage(url=url, title="", text="", error=f"Unsafe URL: {unsafe_reason}")

    own_client = client is None or (
        isinstance(client, httpx.AsyncClient)
        and not isinstance(client, PinnedDNSAsyncClient)
    )
    if own_client:
        client = PinnedDNSAsyncClient(timeout=12.0)
    if isinstance(client, PinnedDNSAsyncClient):
        client.pin_url(url, resolved_addresses)

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
                return (
                    response,
                    current_url,
                    "Redirect response is missing a Location header",
                )
            redirect_url = urljoin(current_url, location)
            unsafe_redirect, redirect_addresses = await resolve_public_url(redirect_url)
            if unsafe_redirect:
                return response, current_url, f"Unsafe redirect URL: {unsafe_redirect}"
            if isinstance(client, PinnedDNSAsyncClient):
                client.pin_url(redirect_url, redirect_addresses)
            current_url = redirect_url
        raise AssertionError("redirect loop exceeded its bound")

    try:
        for attempt in range(2):
            try:
                resp, final_url, redirect_error = await _get_with_safe_redirects(
                    _get_headers()
                )
                if redirect_error:
                    return WebPage(url=url, title="", text="", error=redirect_error)

                if resp.status_code == 200:
                    raw = getattr(resp, "content", b"")
                    is_pdf = (
                        "application/pdf"
                        in resp.headers.get("content-type", "").lower()
                    )
                    if is_pdf or (isinstance(raw, bytes) and raw.startswith(b"%PDF-")):
                        from .pdf_reader import read_pdf

                        result = await read_pdf(
                            raw if isinstance(raw, bytes) else b"", max_chars
                        )
                        return WebPage(
                            url=final_url,
                            title=result.get("title", ""),
                            text=result.get("text", ""),
                            error=result.get("error"),
                            source_type="pdf",
                            extraction_warnings=tuple(result.get("warnings", [])),
                        )
                    if not _is_html_ish(resp):
                        content_type = str(
                            resp.headers.get("content-type", "unknown")
                        ).split(";", 1)[0]
                        return WebPage(
                            url=url,
                            title="",
                            text="",
                            error=f"Unsupported content type: {content_type}",
                        )
                    return await asyncio.to_thread(
                        _extract_content, resp.text, final_url, max_chars
                    )

                # A refusal stays a refusal: it is never retried or routed around.
                return WebPage(
                    url=url, title="", text="", error=f"HTTP {resp.status_code}"
                )

            except httpx.TimeoutException:
                return WebPage(url=url, title="", text="", error="timeout")
            except ResponseTooLargeError as exc:
                return WebPage(url=url, title="", text="", error=str(exc))
            except Exception:
                if attempt == 0:
                    continue
                return WebPage(
                    url=url, title="", text="", error="Page retrieval failed"
                )

        return WebPage(url=url, title="", text="", error="All attempts failed")
    finally:
        if own_client:
            await client.aclose()


async def scrape_urls(
    urls: List[str],
    max_chars: int = MAX_PAGE_CHARS,
    concurrency: int = 4,
) -> List[WebPage]:
    """Read several URLs concurrently over one pooled, DNS-pinned client."""
    semaphore = asyncio.Semaphore(concurrency)
    client = PinnedDNSAsyncClient(concurrency=concurrency, timeout=12.0)

    async def _limited_scrape(url: str) -> WebPage:
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    scrape_url(url, max_chars, client=client), timeout=20.0
                )
            except asyncio.TimeoutError:
                return WebPage(url=url, title="", text="", error="timeout")

    try:
        return await asyncio.gather(*[_limited_scrape(url) for url in urls])
    finally:
        await client.aclose()
