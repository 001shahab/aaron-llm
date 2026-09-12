# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""HTTP client construction and the translation of transport failures.

This is the only module that configures ``httpx``, so timeouts, proxies and default
headers have exactly one home.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from ._version import __version__
from .errors import (
    ConnectionError,
    LocalProviderUnavailable,
    TimeoutError,
    TransportError,
)

DEFAULT_TIMEOUT = 60.0
DEFAULT_CONNECT_TIMEOUT = 10.0

USER_AGENT = f"aaron-llm/{__version__}"


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """How to build the underlying HTTP client.

    Attributes:
        timeout: Total seconds for a whole request, including reading the body.
        connect_timeout: Seconds allowed to establish the connection.
        proxy: Proxy URL, or None to use the environment when ``trust_env`` is set.
        verify: TLS verification. Turning this off is a deliberate act; the library
            never does it for you.
        headers: Extra headers sent on every request from this client.
        trust_env: Whether to honour ``HTTP_PROXY`` and friends.
        follow_redirects: Off by default, so that a provider redirecting to another
            host is a visible error rather than a silent change of endpoint.
    """

    timeout: float = DEFAULT_TIMEOUT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    proxy: str | None = None
    verify: bool = True
    headers: dict[str, str] = field(default_factory=dict)
    trust_env: bool = True
    follow_redirects: bool = False

    def httpx_timeout(self) -> httpx.Timeout:
        """The httpx timeout for this configuration."""
        return httpx.Timeout(self.timeout, connect=self.connect_timeout)

    def with_overrides(
        self, *, timeout: float | None, connect_timeout: float | None
    ) -> TransportConfig:
        """Return a copy with per call timeout overrides applied."""
        changes: dict[str, Any] = {}
        if timeout is not None:
            changes["timeout"] = timeout
        if connect_timeout is not None:
            changes["connect_timeout"] = connect_timeout
        return replace(self, **changes) if changes else self

    def client_kwargs(self) -> dict[str, Any]:
        """Arguments common to both httpx clients. Never contains a credential."""
        return {
            "timeout": self.httpx_timeout(),
            "headers": {"user-agent": USER_AGENT, "accept": "application/json", **self.headers},
            "proxy": self.proxy,
            "verify": self.verify,
            "trust_env": self.trust_env,
            "follow_redirects": self.follow_redirects,
        }


class Transport:
    """Lazily builds and owns one :class:`httpx.Client`.

    Nothing is opened until the first call, so importing Aaron and constructing a
    client makes no connections at all.
    """

    def __init__(
        self, config: TransportConfig | None = None, *, client: httpx.Client | None = None
    ) -> None:
        self.config = config or TransportConfig()
        self._client = client
        self._owned = client is None

    @property
    def client(self) -> httpx.Client:
        """The shared client, created on first use."""
        if self._client is None:
            self._client = httpx.Client(**self.config.client_kwargs())
        return self._client

    def close(self) -> None:
        """Close the client if this transport created it."""
        if self._client is not None and self._owned:
            self._client.close()
            self._client = None


class AsyncTransport:
    """Lazily builds and owns one :class:`httpx.AsyncClient`."""

    def __init__(
        self, config: TransportConfig | None = None, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self.config = config or TransportConfig()
        self._client = client
        self._owned = client is None

    @property
    def client(self) -> httpx.AsyncClient:
        """The shared client, created on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(**self.config.client_kwargs())
        return self._client

    async def aclose(self) -> None:
        """Close the client if this transport created it."""
        if self._client is not None and self._owned:
            await self._client.aclose()
            self._client = None


def translate_httpx_error(
    exc: Exception, *, provider: str, model: str, base_url: str, local: bool = False
) -> TransportError:
    """Turn an httpx exception into the matching Aaron transport error.

    Args:
        exc: The exception httpx raised.
        provider: Provider name, for the error's context fields.
        model: Canonical model string.
        base_url: The endpoint that was being called.
        local: True for a provider running on the caller's machine, which turns a
            refused connection into advice on how to start it.

    Returns:
        A :class:`~aaron.errors.TransportError` subclass. The original exception is
        never swallowed; the call site attaches it with ``raise ... from exc``.
    """
    context: dict[str, Any] = {"provider": provider, "model": model}
    if isinstance(exc, httpx.TimeoutException):
        return TimeoutError(f"request to {base_url} timed out", **context)
    if isinstance(exc, httpx.ConnectError):
        if local:
            return LocalProviderUnavailable(
                f"could not reach {provider} at {base_url}. Start it first, for example "
                f"'ollama serve', or point at another host with OLLAMA_HOST or base_url=",
                **context,
            )
        return ConnectionError(f"could not connect to {base_url}: {exc}", **context)
    if isinstance(exc, httpx.HTTPError):
        return ConnectionError(f"transport failure calling {base_url}: {exc}", **context)
    return TransportError(f"unexpected transport failure calling {base_url}: {exc}", **context)
