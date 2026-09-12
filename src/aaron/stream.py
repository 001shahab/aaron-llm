# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Streaming events, and the one assembler that every provider shares.

Providers translate their own wire format into the event union below and hand each
event to a :class:`StreamAssembler`. No provider builds a :class:`Response` of its
own, which is what keeps the final object identical across providers.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from types import TracebackType
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from .errors import ToolArgumentError
from .registry import Registry
from .request import ChatRequest, estimate_output_tokens, estimate_tokens
from .types import Message, Response, StopReason, TextPart, ToolCall, Usage


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextEvent(_Event):
    """A fragment of assistant text."""

    type: Literal["text"] = "text"
    text: str


class ThinkingEvent(_Event):
    """A fragment of reasoning text, for models that expose it."""

    type: Literal["thinking"] = "thinking"
    text: str


class ToolCallStartEvent(_Event):
    """A tool call has begun. Arguments arrive in later delta events."""

    type: Literal["tool_call_start"] = "tool_call_start"
    id: str
    name: str


class ToolCallDeltaEvent(_Event):
    """A partial JSON fragment of a tool call's arguments. Callers may ignore it."""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    id: str
    arguments_json: str


class ToolCallEndEvent(_Event):
    """A tool call is complete and its arguments are parsed."""

    type: Literal["tool_call_end"] = "tool_call_end"
    call: ToolCall


class DoneEvent(BaseModel):
    """The last event of every stream, carrying the fully assembled response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["done"] = "done"
    response: Response


StreamEvent = Annotated[
    TextEvent
    | ThinkingEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent,
    Field(discriminator="type"),
]


# A tool call accumulating across delta events: its name, and its argument fragments.
_Partial = dict[str, Any]


class StreamAssembler:
    """Accumulates stream events into a single :class:`Response`.

    The assembler is deliberately forgiving about event order. Any interleaving of
    valid events assembles without raising, because a provider that reorders its
    chunks must not turn into a crash in the caller's loop. The one exception is
    tool call arguments that are not valid JSON, which raise
    :class:`~aaron.errors.ToolArgumentError` rather than silently becoming ``{}``.
    """

    def __init__(self, req: ChatRequest, *, registry: Registry | None = None) -> None:
        self._req = req
        self._registry = registry or req.registry
        self._started = time.monotonic()
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._open: dict[str, _Partial] = {}
        self._calls: list[ToolCall] = []
        self._usage: Usage | None = None
        self._stop_reason: StopReason | None = None
        self._id: str | None = None
        self._resolved_model: str | None = None
        self._raw: dict[str, Any] = {}
        self.emitted_text = False

    def add(self, event: StreamEvent) -> None:
        """Fold one event into the accumulating response.

        Args:
            event: Any event except :class:`DoneEvent`, which the assembler makes.
        """
        if isinstance(event, TextEvent):
            self._text.append(event.text)
            if event.text:
                self.emitted_text = True
        elif isinstance(event, ThinkingEvent):
            self._thinking.append(event.text)
        elif isinstance(event, ToolCallStartEvent):
            self._open.setdefault(event.id, _new_partial())["name"] = event.name
        elif isinstance(event, ToolCallDeltaEvent):
            # A delta may arrive before its start event; keep it either way.
            self._open.setdefault(event.id, _new_partial())["fragments"].append(
                event.arguments_json
            )
        elif isinstance(event, ToolCallEndEvent):
            self._open.pop(event.call.id, None)
            self._calls = [call for call in self._calls if call.id != event.call.id]
            self._calls.append(event.call)

    def note(
        self,
        *,
        id: str | None = None,
        resolved_model: str | None = None,
        stop_reason: StopReason | None = None,
        usage: Usage | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        """Record metadata that arrives outside the event stream.

        Usage often only appears in the final chunk, and the response id only in
        the first. Providers call this as they see each, and the last non ``None``
        value wins.

        Args:
            id: Provider response id.
            resolved_model: Exactly what the provider reported back.
            stop_reason: Normalised stop reason.
            usage: Token counts, merged over anything recorded earlier.
            raw: Provider payload fragments to keep on ``Response.raw``.
        """
        if id is not None:
            self._id = id
        if resolved_model is not None:
            self._resolved_model = resolved_model
        if stop_reason is not None:
            self._stop_reason = stop_reason
        if usage is not None:
            self._usage = _merge_usage(self._usage, usage)
        if raw is not None:
            self._raw.update(raw)

    def close_tool_calls(self) -> list[ToolCallEndEvent]:
        """Finish every tool call that received deltas but no explicit end event.

        Returns:
            The end events the provider should yield, in the order the calls
            started.

        Raises:
            ToolArgumentError: A model produced arguments that are not a JSON
                object. The raw string is attached to the error.
        """
        events: list[ToolCallEndEvent] = []
        for call_id, partial in list(self._open.items()):
            name = str(partial["name"])
            call = ToolCall(
                id=call_id,
                name=name,
                arguments=parse_tool_arguments(
                    "".join(partial["fragments"]), tool_name=name, req=self._req
                ),
            )
            self._open.pop(call_id, None)
            self._calls.append(call)
            events.append(ToolCallEndEvent(call=call))
        return events

    @property
    def text(self) -> str:
        """Everything accumulated so far."""
        return "".join(self._text)

    def done(self) -> DoneEvent:
        """Assemble the terminal event.

        Returns:
            A :class:`DoneEvent` whose response has usage and cost filled in. When the
            provider never reported output tokens they are estimated from the
            generated text, and ``Cost.estimated`` is set.
        """
        latency_ms = int((time.monotonic() - self._started) * 1000)
        content: list[TextPart] = [TextPart(text=self.text)] if self._text else []
        message = Message(role="assistant", content=list(content), tool_calls=list(self._calls))

        usage, estimated = self._resolve_usage()
        stop_reason = self._stop_reason or ("tool_use" if self._calls else "stop")
        raw = dict(self._raw)
        if self._thinking:
            raw["thinking"] = "".join(self._thinking)

        response = Response(
            id=self._id or f"aaron-{uuid.uuid4()}",
            model=self._req.model,
            resolved_model=self._resolved_model or self._req.model_name,
            message=message,
            stop_reason=stop_reason,
            usage=usage,
            cost=self._registry.cost(self._req.model, usage, estimated=estimated),
            latency_ms=latency_ms,
            raw=raw,
        )
        return DoneEvent(response=response)

    def _resolve_usage(self) -> tuple[Usage, bool]:
        """Fill in whatever the provider did not report, and say so."""
        reported = self._usage
        if reported is not None and reported.output_tokens > 0 and reported.input_tokens > 0:
            return reported, False
        estimated_output = estimate_output_tokens(self.text) + sum(
            estimate_output_tokens(json.dumps(call.arguments, default=str)) for call in self._calls
        )
        return (
            Usage(
                input_tokens=(reported.input_tokens if reported else 0)
                or estimate_tokens(self._req.messages),
                output_tokens=(reported.output_tokens if reported else 0) or estimated_output,
                cached_input_tokens=reported.cached_input_tokens if reported else 0,
                reasoning_tokens=reported.reasoning_tokens if reported else 0,
            ),
            True,
        )


def _new_partial() -> _Partial:
    return {"name": "", "fragments": []}


def _merge_usage(current: Usage | None, incoming: Usage) -> Usage:
    """Take the larger of each counter, since providers report cumulative totals."""
    if current is None:
        return incoming
    return Usage(
        input_tokens=max(current.input_tokens, incoming.input_tokens),
        output_tokens=max(current.output_tokens, incoming.output_tokens),
        cached_input_tokens=max(current.cached_input_tokens, incoming.cached_input_tokens),
        reasoning_tokens=max(current.reasoning_tokens, incoming.reasoning_tokens),
    )


def parse_tool_arguments(
    raw: str, *, tool_name: str, req: ChatRequest | None = None
) -> dict[str, Any]:
    """Parse a model's tool arguments into a dict.

    Args:
        raw: The argument string exactly as the model produced it.
        tool_name: The tool being called, for the error message.
        req: The originating request, used to attach provider and model context.

    Returns:
        The parsed arguments. An empty or whitespace only string means no
        arguments, which is legitimate for a tool that takes none.

    Raises:
        ToolArgumentError: The string is not valid JSON, or is JSON that is not an
            object. The raw string is attached as ``raw_arguments``.
    """
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _bad_arguments(
            f"arguments that are not valid JSON: {exc}", raw, tool_name, req
        ) from exc
    if not isinstance(parsed, dict):
        raise _bad_arguments(
            f"{type(parsed).__name__} arguments, expected an object", raw, tool_name, req
        )
    return parsed


def _bad_arguments(
    detail: str, raw: str, tool_name: str, req: ChatRequest | None
) -> ToolArgumentError:
    return ToolArgumentError(
        f"tool {tool_name!r} produced {detail}",
        raw_arguments=raw,
        tool_name=tool_name,
        provider=req.provider if req else None,
        model=req.model if req else None,
    )


class Stream:
    """A synchronous stream of events that owns its HTTP connection.

    The stream holds a live connection until it is exhausted or closed. A caller
    who breaks out of the loop early must close it, either explicitly or by using
    the stream as a context manager:

    ```python
    with client.stream("ollama/llama3.1", "Hello") as stream:
        for event in stream:
            if event.type == "text":
                print(event.text, end="")
                break
    ```

    Iterating to the end closes it automatically.
    """

    def __init__(self, events: Iterator[StreamEvent], close: Callable[[], None]) -> None:
        self._events = events
        self._close = close
        self._closed = False
        self.response: Response | None = None

    def __iter__(self) -> Iterator[StreamEvent]:
        return self

    def __next__(self) -> StreamEvent:
        try:
            event = next(self._events)
        except StopIteration:
            self.close()
            raise
        if isinstance(event, DoneEvent):
            self.response = event.response
        return event

    def close(self) -> None:
        """Release the connection and finalise the audit record. Idempotent."""
        if not self._closed:
            self._closed = True
            self._close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class AsyncStream:
    """The async twin of :class:`Stream`, usable with ``async for`` and ``async with``."""

    def __init__(self, events: AsyncIterator[StreamEvent], close: Callable[[], Any]) -> None:
        self._events = events
        self._close = close
        self._closed = False
        self.response: Response | None = None

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __anext__(self) -> StreamEvent:
        try:
            event = await self._events.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        if isinstance(event, DoneEvent):
            self.response = event.response
        return event

    async def aclose(self) -> None:
        """Release the connection and finalise the audit record. Idempotent."""
        if not self._closed:
            self._closed = True
            result = self._close()
            if result is not None:
                await result

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
