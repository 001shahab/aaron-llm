# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""The provider protocol and the lazy provider registry.

A provider is a small translator: it turns a :class:`~aaron.request.ChatRequest`
into an HTTP request and turns the reply back into a
:class:`~aaron.types.Response` or a typed error. It holds no state and makes no
policy decisions.

Built in providers are imported on first use. Third party providers register
through the ``aaron.providers`` entry point group and are also loaded lazily, so
an installed plugin costs nothing until its name is used.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..errors import AaronError, UnknownProvider
from ..registry import Capabilities
from ..request import ChatRequest
from ..stream import StreamEvent
from ..types import Response, SecretValue

if TYPE_CHECKING:
    from .base import EventParser

__all__ = [
    "PreparedRequest",
    "Provider",
    "available_providers",
    "get_provider",
    "register_provider",
]


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """An outgoing HTTP request, fully built but not yet sent.

    Credentials live in ``auth``, separate from ``headers``, so that printing,
    logging or serialising this object cannot leak one. Only
    :meth:`sendable_headers` reassembles them, and only the transport calls it.

    Attributes:
        headers: Non secret headers.
        body: The JSON body, including anything from ``provider_options``.
        auth: Secret headers, wrapped so their repr is masked.
    """

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)
    auth: dict[str, SecretValue] = field(default_factory=dict)
    stream: bool = False

    def sendable_headers(self) -> dict[str, str]:
        """Headers with credentials filled in. Never log the result."""
        return {**self.headers, **{name: value.get() for name, value in self.auth.items()}}

    def masked(self) -> PreparedRequest:
        """Return a copy safe to print, with every credential replaced by a mask.

        This is what :meth:`aaron.Aaron.dry_run` returns, so a compliance reviewer
        can inspect exactly what would go over the wire without handling secrets.
        """
        masked_headers = {**self.headers, **dict.fromkeys(self.auth, "****")}
        return replace(self, headers=masked_headers, auth={})


@runtime_checkable
class Provider(Protocol):
    """What a provider must implement.

    Attributes:
        name: The prefix used in a model string, for example ``"openai"``.
        default_base_url: Endpoint root used when the caller gives none.
        env_key: Environment variable holding the credential. Empty when the
            provider needs none, as with a local Ollama.
        local: True when the provider runs on the caller's own machine, which turns
            a refused connection into an error that says how to start it.
        requires_key: Whether a missing credential is an error. False for a local
            Ollama and for a compatible endpoint that may well have no auth.
    """

    name: str
    default_base_url: str
    env_key: str
    local: bool
    requires_key: bool

    def build_request(self, req: ChatRequest, *, stream: bool) -> PreparedRequest:
        """Translate a request into an inspectable HTTP request."""
        ...

    def parse_response(self, payload: dict[str, Any], *, req: ChatRequest) -> Response:
        """Turn a successful provider payload into a normalised response."""
        ...

    def iter_events(self, lines: Iterable[bytes], *, req: ChatRequest) -> Iterator[StreamEvent]:
        """Turn raw response lines into normalised events, ending with one done event."""
        ...

    def event_parser(self, req: ChatRequest) -> EventParser:
        """Return a push parser for one streaming call.

        This exists so that the async client can drive translation from
        ``aiter_lines`` without a provider having to write its streaming logic
        twice. Subclasses of :class:`~aaron.providers.base.BaseProvider` get both
        this and :meth:`iter_events` for free from one ``chunk_events`` method.
        """
        ...

    def map_error(self, status: int, payload: dict[str, Any] | None, text: str) -> AaronError:
        """Map an error response onto the Aaron exception hierarchy."""
        ...

    def capabilities(self, model: str) -> Capabilities:
        """What the named model can do, according to the registry."""
        ...


# Built in providers, imported on first use to keep import time flat.
_BUILTIN: dict[str, tuple[str, str]] = {
    "openai": (".openai", "OpenAIProvider"),
    "anthropic": (".anthropic", "AnthropicProvider"),
    "google": (".google", "GoogleProvider"),
    "ollama": (".ollama", "OllamaProvider"),
    "openai_compat": (".openai_compat", "OpenAICompatProvider"),
}

_instances: dict[str, Provider] = {}


def register_provider(provider: Provider) -> None:
    """Register a provider instance under its own name.

    Args:
        provider: Any object satisfying :class:`Provider`.
    """
    _instances[provider.name] = provider


def get_provider(name: str) -> Provider:
    """Look up a provider by name, importing it on first use.

    Args:
        name: The provider prefix from a model string.

    Returns:
        The provider instance, cached for the life of the process.

    Raises:
        UnknownProvider: No built in, registered or entry point provider matches.
    """
    cached = _instances.get(name)
    if cached is not None:
        return cached

    if name in _BUILTIN:
        module_name, class_name = _BUILTIN[name]
        from importlib import import_module

        module = import_module(module_name, package=__name__)
        provider: Provider = getattr(module, class_name)()
        _instances[name] = provider
        return provider

    plugin = _load_entry_point(name)
    if plugin is not None:
        _instances[name] = plugin
        return plugin

    raise UnknownProvider(
        f"unknown provider {name!r}. Available: {', '.join(available_providers())}. "
        "For any OpenAI compatible endpoint use "
        "'openai_compat/<model>' with an explicit base_url."
    )


def _load_entry_point(name: str) -> Provider | None:
    """Find a third party provider in the ``aaron.providers`` entry point group."""
    from importlib.metadata import entry_points

    for entry in entry_points(group="aaron.providers"):
        if entry.name != name:
            continue
        loaded = entry.load()
        instance = loaded() if isinstance(loaded, type) else loaded
        if not isinstance(instance, Provider):
            raise UnknownProvider(
                f"entry point {entry.name!r} does not implement the aaron Provider protocol"
            )
        return instance
    return None


def available_providers() -> list[str]:
    """Every provider name that can be resolved right now, sorted."""
    from importlib.metadata import entry_points

    plugins = {entry.name for entry in entry_points(group="aaron.providers")}
    return sorted(set(_BUILTIN) | set(_instances) | plugins)
