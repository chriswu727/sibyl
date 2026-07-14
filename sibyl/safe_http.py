"""HTTP transport that connects only to pre-validated public IP addresses."""
from __future__ import annotations

import ipaddress
import os
import ssl
import threading
from typing import AsyncIterable, Dict, Iterable, NoReturn, Optional, Tuple, Type
from urllib.parse import urlsplit

import certifi
import httpcore
import httpx


def _normalize_hostname(hostname: str) -> str:
    return hostname.rstrip(".").encode("idna").decode("ascii").lower()


class PinnedDNSNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(
        self,
        backend: Optional[httpcore.AsyncNetworkBackend] = None,
    ) -> None:
        self._backend = backend or httpcore.AnyIOBackend()
        self._pins: Dict[str, Tuple[str, ...]] = {}
        self._pins_lock = threading.Lock()

    def pin(self, hostname: str, addresses: Iterable[str]) -> None:
        public_addresses = []
        for value in addresses:
            address = ipaddress.ip_address(value)
            if not address.is_global:
                raise ValueError(f"Refusing to pin non-public IP address: {value}")
            normalized = str(address)
            if normalized not in public_addresses:
                public_addresses.append(normalized)
        if not public_addresses:
            raise ValueError("At least one public IP address is required")
        with self._pins_lock:
            self._pins[_normalize_hostname(hostname)] = tuple(public_addresses)

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
        socket_options=None,
    ) -> httpcore.AsyncNetworkStream:
        with self._pins_lock:
            addresses = self._pins.get(_normalize_hostname(host), ())
        if not addresses:
            raise httpcore.ConnectError(
                f"Hostname {host!r} has not been validated and pinned"
            )

        last_error = None
        for address in addresses:
            try:
                return await self._backend.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def connect_unix_socket(
        self,
        path: str,
        timeout: Optional[float] = None,
        socket_options=None,
    ) -> httpcore.AsyncNetworkStream:
        raise httpcore.ConnectError(
            "Unix sockets are not supported by the safe transport"
        )

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


_CORE_EXCEPTION_MAP: Tuple[
    Tuple[Type[Exception], Type[httpx.HTTPError]], ...
] = (
    (httpcore.ConnectTimeout, httpx.ConnectTimeout),
    (httpcore.ReadTimeout, httpx.ReadTimeout),
    (httpcore.WriteTimeout, httpx.WriteTimeout),
    (httpcore.PoolTimeout, httpx.PoolTimeout),
    (httpcore.ConnectError, httpx.ConnectError),
    (httpcore.ReadError, httpx.ReadError),
    (httpcore.WriteError, httpx.WriteError),
    (httpcore.LocalProtocolError, httpx.LocalProtocolError),
    (httpcore.RemoteProtocolError, httpx.RemoteProtocolError),
    (httpcore.UnsupportedProtocol, httpx.UnsupportedProtocol),
    (httpcore.TimeoutException, httpx.TimeoutException),
    (httpcore.NetworkError, httpx.NetworkError),
    (httpcore.ProtocolError, httpx.ProtocolError),
    (httpcore.ConnectionNotAvailable, httpx.ConnectError),
)


def _ssl_context() -> ssl.SSLContext:
    if os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
    if os.environ.get("SSL_CERT_DIR"):
        return ssl.create_default_context(capath=os.environ["SSL_CERT_DIR"])
    return ssl.create_default_context(cafile=certifi.where())


def _raise_mapped_exception(exc: Exception, request: httpx.Request) -> NoReturn:
    mapped = next(
        target
        for source, target in _CORE_EXCEPTION_MAP
        if isinstance(exc, source)
    )
    raise mapped(str(exc), request=request) from exc


class _CoreResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream: AsyncIterable[bytes], request: httpx.Request) -> None:
        self._stream = stream
        self._request = request

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                yield chunk
        except tuple(error for error, _ in _CORE_EXCEPTION_MAP) as exc:
            _raise_mapped_exception(exc, self._request)

    async def aclose(self) -> None:
        if hasattr(self._stream, "aclose"):
            try:
                await self._stream.aclose()
            except tuple(error for error, _ in _CORE_EXCEPTION_MAP) as exc:
                _raise_mapped_exception(exc, self._request)


class PinnedDNSAsyncHTTPTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        *,
        limits: httpx.Limits,
        network_backend: Optional[httpcore.AsyncNetworkBackend] = None,
    ) -> None:
        self.network_backend = PinnedDNSNetworkBackend(network_backend)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=_ssl_context(),
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            network_backend=self.network_backend,
        )

    def pin_url(self, url: str, addresses: Iterable[str]) -> None:
        hostname = urlsplit(url).hostname
        if hostname is None:
            raise ValueError("URL must include a hostname")
        self.network_backend.pin(hostname, addresses)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        try:
            response = await self._pool.handle_async_request(core_request)
        except tuple(error for error, _ in _CORE_EXCEPTION_MAP) as exc:
            _raise_mapped_exception(exc, request)

        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_CoreResponseStream(response.stream, request),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


class PinnedDNSAsyncClient(httpx.AsyncClient):
    def __init__(
        self,
        *,
        concurrency: int = 8,
        timeout: float = 12.0,
        network_backend: Optional[httpcore.AsyncNetworkBackend] = None,
    ) -> None:
        limits = httpx.Limits(
            max_connections=concurrency * 2,
            max_keepalive_connections=concurrency,
        )
        self.pinned_transport = PinnedDNSAsyncHTTPTransport(
            limits=limits,
            network_backend=network_backend,
        )
        super().__init__(
            transport=self.pinned_transport,
            follow_redirects=False,
            timeout=timeout,
        )

    def pin_url(self, url: str, addresses: Iterable[str]) -> None:
        self.pinned_transport.pin_url(url, addresses)
