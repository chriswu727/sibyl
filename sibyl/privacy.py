"""Presentation-only URL sanitization; never changes hashed evidence text."""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_SECRET_KEYS = {
    "key",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "auth",
    "authorization",
    "password",
    "secret",
    "signature",
    "sig",
    "code",
    "credential",
}


def display_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        host = parts.netloc.rsplit("@", 1)[-1]
        query = [
            (
                key,
                "REDACTED"
                if any(secret in key.lower() for secret in _SECRET_KEYS)
                else value,
            )
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        ]
        return urlunsplit((parts.scheme, host, parts.path, urlencode(query), ""))
    except ValueError:
        return "[invalid URL]"
