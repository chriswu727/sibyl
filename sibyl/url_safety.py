"""Safety checks for URLs fetched by Sibyl."""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit


_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".home", ".lan")
_BLOCKED_HOSTNAMES = {"localhost", "local", "internal", "home", "lan"}
_ALLOWED_PORTS = {80, 443}


def _parsed_ip(hostname: str):
    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        pass
    try:
        return ipaddress.ip_address(socket.inet_aton(hostname))
    except (OSError, ValueError):
        return None


def _unsafe_address_reason(address) -> str:
    if (
        not address.is_global
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        return "Private, loopback, link-local, and reserved IP addresses are not allowed."
    return ""


def _resolve_hostname(hostname: str, port: int):
    records = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(record[4][0] for record in records))


def unsafe_url_reason(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return "URL must be a non-empty string."
    if any(ord(character) < 32 for character in url):
        return "URL contains control characters."

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return "URL is malformed."
    if parsed.scheme.lower() not in {"http", "https"}:
        return "Only http and https URLs are allowed."
    if not parsed.hostname:
        return "URL must include a hostname."
    if parsed.username is not None or parsed.password is not None:
        return "URLs containing credentials are not allowed."
    if port is not None and port not in _ALLOWED_PORTS:
        return "Only ports 80 and 443 are allowed."

    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname in _BLOCKED_HOSTNAMES or hostname.endswith(_BLOCKED_HOST_SUFFIXES):
        return "Local and internal hostnames are not allowed."
    address = _parsed_ip(hostname)
    if address is not None:
        return _unsafe_address_reason(address)
    return ""


async def validate_public_url(url: str) -> str:
    reason = unsafe_url_reason(url)
    if reason:
        return reason

    parsed = urlsplit(url)
    hostname = parsed.hostname.rstrip(".").casefold()
    if _parsed_ip(hostname) is not None:
        return ""
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        addresses = await asyncio.to_thread(_resolve_hostname, hostname, port)
    except (OSError, socket.gaierror):
        return "Hostname could not be resolved."
    if not addresses:
        return "Hostname did not resolve to an address."
    for value in addresses:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return "Hostname resolved to an invalid IP address."
        reason = _unsafe_address_reason(address)
        if reason:
            return reason
    return ""
