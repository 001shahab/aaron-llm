# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Helpers shared by every provider: wire decoding, option merging, error mapping.

Streaming is built around a push parser rather than a generator. A provider only
implements :meth:`BaseProvider.chunk_events`, which turns one decoded chunk into
events; the shared :class:`ChunkParser` owns the line decoding, the
:class:`~aaron.stream.StreamAssembler` and the terminal done event. That is what
lets the sync client drive it from a blocking iterator and the async client drive
it from ``aiter_lines`` without either provider code or the event union being
written twice.

When a provider file approaches its 400 line limit, the answer is to move logic
here, never into another provider file.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ..errors import (
    AaronError,
    AuthenticationError,
    ContentFilterError,
    ContextLengthExceeded,
    InvalidRequest,
    PermissionError,
    ProviderError,
    RateLimitError,
    ServerError,
    ServiceOverloaded,
    TimeoutError,
    UnknownModel,
)
from ..registry import Capabilities
from ..request import ChatRequest
from ..stream import StreamAssembler, StreamEvent
from ..types import StopReason

log = logging.getLogger("aaron")


# Substrings that mean "your prompt is too long", across providers.
_CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "prompt is too long",
    "reduce the length",
    "input token count",
    "exceeds the maximum",
)

_FILTER_MARKERS = (
    "content filter",
    "content_filter",
    "content management policy",
    "safety",
    "blocked",
    "responsible ai",
    "recitation",
)


@dataclass(frozen=True, slots=True)
class SSEMessage:
    """One server sent event: its optional name and its accumulated data."""

    event: str | None
    data: str


class SSEDecoder:
    """Incremental server sent event decoder.

    Feed it one line at a time and it returns complete events. Handles multi line
    ``data:`` fields, named events, comment keep alives and the ``[DONE]`` sentinel,
    which is consumed rather than surfaced.
    """

    def __init__(self) -> None:
        self._name: str | None = None
        self._data: list[str] = []

    def feed(self, raw: bytes) -> list[SSEMessage]:
        """Decode one line.

        Args:
            raw: The line, without its terminator.

        Returns:
            Zero or one complete event, as a list so callers can extend.
        """
        line = raw.decode("utf-8", errors="replace").rstrip("\r")
        if not line:
            return self._flush()
        if line.startswith(":"):
            return []  # comment or keep alive
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._name = value
        return []

    def finish(self) -> list[SSEMessage]:
        """Emit any event left unterminated by a missing final blank line."""
        return self._flush()

    def _flush(self) -> list[SSEMessage]:
        if not self._data:
            self._name = None
            return []
        payload = "\n".join(self._data)
        name, self._name, self._data = self._name, None, []
        if payload.strip() == "[DONE]":
            return []
        return [SSEMessage(event=name, data=payload)]


class NDJSONDecoder:
    """Incremental newline delimited JSON decoder, as Ollama streams."""

    def feed(self, raw: bytes) -> list[SSEMessage]:
        """Decode one line into at most one message carrying the raw JSON text."""
        text = raw.decode("utf-8", errors="replace").strip()
        return [SSEMessage(event=None, data=text)] if text else []

    def finish(self) -> list[SSEMessage]:
        """Nothing is buffered, so there is never anything left over."""
        return []


def merge_options(body: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Merge ``provider_options`` into a request body.

    Nested mappings are merged key by key, so a caller can set one field of an
    object the provider built. Anything else replaces what was there, which is what
    makes this a usable escape hatch: the caller always wins.

    Args:
        body: The body the provider built.
        options: The caller's provider specific options.

    Returns:
        A new merged body.
    """
    merged = dict(body)
    for key, value in options.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = merge_options(existing, value)
        else:
            merged[key] = value
    return merged


class EventParser(Protocol):
    """A push parser turning raw response lines into normalised events."""

    def feed(self, line: bytes) -> list[StreamEvent]:
        """Consume one line and return any events it completed."""
        ...

    def finish(self) -> list[StreamEvent]:
        """Return any trailing events, ending with exactly one done event."""
        ...


class ChunkParser:
    """Drives a provider's chunk translation and owns the assembler.

    Because the terminal done event is emitted here rather than in a provider, the
    guarantee that every stream ends with exactly one done event holds for every
    provider by construction, including third party ones built on
    :class:`BaseProvider`.
    """

    def __init__(self, provider: BaseProvider, req: ChatRequest) -> None:
        self._provider = provider
        self._req = req
        self._assembler = StreamAssembler(req)
        self._decoder: SSEDecoder | NDJSONDecoder = (
            NDJSONDecoder() if provider.wire == "ndjson" else SSEDecoder()
        )
        self._state: dict[str, Any] = {}
        self._finished = False

    def feed(self, line: bytes) -> list[StreamEvent]:
        """Decode one line and translate whatever chunks it completed.

        Args:
            line: The raw line, without its terminator.

        Returns:
            The events to hand to the caller, possibly none.
        """
        events: list[StreamEvent] = []
        for message in self._decoder.feed(line):
            events.extend(self._translate(message))
        return events

    def finish(self) -> list[StreamEvent]:
        """Flush the decoder, close any open tool calls, and emit the done event."""
        if self._finished:
            return []
        self._finished = True
        events: list[StreamEvent] = []
        for message in self._decoder.finish():
            events.extend(self._translate(message))
        for event in self._assembler.close_tool_calls():
            self._assembler.add(event)
            events.append(event)
        events.append(self._assembler.done())
        return events

    def _translate(self, message: SSEMessage) -> list[StreamEvent]:
        try:
            chunk = json.loads(message.data)
        except json.JSONDecodeError:
            log.debug("skipping undecodable stream chunk: %r", message.data[:120])
            return []
        if not isinstance(chunk, dict):
            return []
        return list(
            self._provider.chunk_events(
                chunk, assembler=self._assembler, state=self._state, event=message.event
            )
        )


class BaseProvider:
    """Shared behaviour for the built in providers.

    A subclass sets the class attributes and implements ``build_request``,
    ``parse_response`` and ``chunk_events``.
    """

    name: str = ""
    default_base_url: str = ""
    env_key: str = ""
    local: bool = False
    requires_key: bool = True
    wire: Literal["sse", "ndjson"] = "sse"

    def capabilities(self, model: str) -> Capabilities:
        """Capabilities for ``provider/model``, taken from the shipped registry.

        Args:
            model: A canonical ``provider/model`` string.

        Returns:
            The capabilities, permissive when the model is unknown.
        """
        from ..registry import default_registry

        return default_registry().capabilities(model)

    def base_url(self, req: ChatRequest) -> str:
        """The endpoint root for this request, without a trailing slash."""
        return (req.base_url or self.default_base_url).rstrip("/")

    # Streaming -------------------------------------------------------------------

    def event_parser(self, req: ChatRequest) -> ChunkParser:
        """Build a push parser for one streaming call.

        The async client drives this directly. Synchronous callers normally use
        :meth:`iter_events` instead.

        Args:
            req: The originating request.

        Returns:
            A parser bound to this provider and request.
        """
        return ChunkParser(self, req)

    def iter_events(self, lines: Iterable[bytes], *, req: ChatRequest) -> Iterator[StreamEvent]:
        """Translate an iterable of response lines into normalised events.

        Args:
            lines: Raw response lines.
            req: The originating request.

        Yields:
            Every translated event, ending with exactly one done event.
        """
        parser = self.event_parser(req)
        for line in lines:
            yield from parser.feed(line)
        yield from parser.finish()

    def chunk_events(
        self,
        chunk: dict[str, Any],
        *,
        assembler: StreamAssembler,
        state: dict[str, Any],
        event: str | None,
    ) -> Iterator[StreamEvent]:
        """Turn one decoded chunk into events, folding each into the assembler.

        Args:
            chunk: One decoded JSON chunk from the provider.
            assembler: The shared assembler. Every yielded event must also be added
                to it, and metadata such as usage passed to ``assembler.note``.
            state: Scratch space that persists across chunks for one call.
            event: The SSE event name, when the provider uses named events.

        Yields:
            The events this chunk produced.
        """
        raise NotImplementedError

    # Error mapping ---------------------------------------------------------------

    def map_error(self, status: int, payload: dict[str, Any] | None, text: str) -> AaronError:
        """Map an error response onto the exception hierarchy.

        Args:
            status: HTTP status code.
            payload: Decoded JSON error body, when there was one.
            text: Raw body text, used when the body was not JSON.

        Returns:
            The matching :class:`~aaron.errors.AaronError` subclass.
        """
        message, code, request_id, retry_after = self.error_fields(payload, text)
        return classify(
            status,
            message=message,
            code=code,
            provider=self.name,
            payload=payload,
            request_id=request_id,
            retry_after=retry_after,
        )

    def error_fields(
        self, payload: dict[str, Any] | None, text: str
    ) -> tuple[str, str | None, str | None, float | None]:
        """Pull message, code, request id and retry hint out of an error body.

        The default understands the ``{"error": {"message": ..., "code": ...}}``
        shape used by OpenAI and most compatible endpoints. A provider with a
        different shape overrides this.
        """
        if not payload:
            return (text or "provider returned an error with an empty body"), None, None, None
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or text or "provider error")
            code = error.get("code") or error.get("type")
            return message, (str(code) if code else None), _request_id(payload), None
        if isinstance(error, str):
            return error, None, _request_id(payload), None
        message = str(payload.get("message") or payload.get("detail") or text or "provider error")
        return message, None, _request_id(payload), None


def _request_id(payload: dict[str, Any]) -> str | None:
    for key in ("request_id", "requestId", "id"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def classify(
    status: int,
    *,
    message: str,
    code: str | None,
    provider: str,
    payload: dict[str, Any] | None,
    request_id: str | None = None,
    retry_after: float | None = None,
    model: str | None = None,
) -> AaronError:
    """Choose the error class for a status code and message.

    Status codes alone are not enough: a context length overflow and a malformed
    body are both 400, and a safety refusal can be 400 or 403, so the message is
    consulted for those cases.

    Args:
        status: HTTP status code.
        message: Provider message.
        code: Provider error code, when there is one.
        provider: Provider name.
        payload: Raw error body, attached to the exception.
        request_id: Provider request id.
        retry_after: Seconds from a ``Retry-After`` header.
        model: Canonical model string.

    Returns:
        An :class:`~aaron.errors.AaronError` subclass instance.
    """
    context: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "status_code": status,
        "request_id": request_id,
        "raw": payload,
    }
    haystack = f"{message} {code or ''}".lower()

    if status == 429:
        return RateLimitError(message, retry_after=retry_after, **context)
    if status == 401:
        return AuthenticationError(message, **context)
    if status == 403:
        if _matches(haystack, _FILTER_MARKERS):
            return ContentFilterError(message, **context)
        return PermissionError(message, **context)
    if status == 404:
        if "model" in haystack:
            return UnknownModel(message, **context)
        return InvalidRequest(message, **context)
    if status in (408, 504):
        return TimeoutError(message, **context)
    if status in (502, 503, 529):
        return ServiceOverloaded(message, **context)
    if status >= 500:
        return ServerError(message, **context)
    if status in (400, 413, 422):
        if _matches(haystack, _CONTEXT_MARKERS):
            return ContextLengthExceeded(message, **context)
        if _matches(haystack, _FILTER_MARKERS):
            return ContentFilterError(message, **context)
        return InvalidRequest(message, **context)
    if 400 <= status < 500:
        return InvalidRequest(message, **context)
    return ProviderError(message, **context)


def _matches(haystack: str, markers: tuple[str, ...]) -> bool:
    return any(marker in haystack for marker in markers)


def normalise_stop_reason(raw: str | None, *, mapping: dict[str, StopReason]) -> StopReason:
    """Map a provider's finish reason onto the shared vocabulary.

    Args:
        raw: The provider's own value, which may be None.
        mapping: Provider specific translations.

    Returns:
        The normalised reason. An unrecognised value becomes ``"other"`` rather than
        ``"stop"``, so a new provider value is visible instead of silently wrong.
    """
    if raw is None:
        return "stop"
    return mapping.get(raw, "other")
