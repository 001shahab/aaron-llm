# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Ollama, against ``POST /api/chat`` on ``http://localhost:11434``.

No API key, no cost, and nothing leaves the machine, which makes it the provider to
reach for in a test, in a quickstart, and as the fallback in a residency policy.

Ollama streams by default, so ``stream`` is always sent explicitly. A refused
connection becomes :class:`~aaron.errors.LocalProviderUnavailable` with instructions,
because "connection refused" is not a useful thing to show a user who simply has not
started the server.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ..errors import InvalidRequest
from ..request import ChatRequest, estimate_output_tokens, estimate_tokens
from ..stream import (
    StreamAssembler,
    StreamEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEndEvent,
)
from ..types import (
    DocumentPart,
    ImagePart,
    Message,
    Response,
    StopReason,
    TextPart,
    ToolCall,
    Usage,
)
from . import PreparedRequest
from .base import BaseProvider, merge_options, normalise_stop_reason

_STOP_REASONS: dict[str, StopReason] = {
    "stop": "stop",
    "length": "length",
    "load": "other",
    "unload": "other",
}


class OllamaProvider(BaseProvider):
    """Translates to and from the Ollama chat API."""

    name = "ollama"
    default_base_url = "http://localhost:11434"
    env_key = ""
    local = True
    requires_key = False
    wire = "ndjson"

    def build_request(self, req: ChatRequest, *, stream: bool) -> PreparedRequest:
        """Build the ``/api/chat`` request.

        ``keep_alive`` and the ``options`` object, which is where Ollama keeps
        ``num_ctx``, ``num_predict``, ``top_k`` and the rest, are reachable through
        ``provider_options``.

        Args:
            req: The normalised request.
            stream: Whether to receive newline delimited JSON chunks.

        Returns:
            An inspectable prepared request.
        """
        options: dict[str, Any] = {}
        if req.max_tokens is not None:
            options["num_predict"] = req.max_tokens
        if req.temperature is not None:
            options["temperature"] = req.temperature
        if req.top_p is not None:
            options["top_p"] = req.top_p
        if req.stop:
            options["stop"] = req.stop

        body: dict[str, Any] = {
            "model": req.model_name,
            "messages": [self.encode(message) for message in req.messages],
            "stream": stream,
        }
        if options:
            body["options"] = options
        if req.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in req.tools
            ]
        if req.json_schema is not None:
            body["format"] = req.json_schema.schema
        elif req.json_mode:
            body["format"] = "json"

        return PreparedRequest(
            method="POST",
            url=f"{self.base_url(req)}/api/chat",
            headers={"content-type": "application/json", **req.extra_headers},
            body=merge_options(body, req.provider_options),
            stream=stream,
        )

    def encode(self, message: Message) -> dict[str, Any]:
        """Encode one message. Images ride alongside the text, not inside it.

        Raises:
            InvalidRequest: The message carries a document. The Ollama chat API has
                nowhere to put one, and dropping it silently would answer a question
                about a file the model never saw.
        """
        if any(isinstance(part, DocumentPart) for part in message.content):
            raise InvalidRequest(
                "ollama cannot accept a document. Extract the text yourself and send "
                "it as text, or use a provider with document support.",
                provider=self.name,
            )
        entry: dict[str, Any] = {"role": message.role, "content": message.text}
        images = [part.data for part in message.content if isinstance(part, ImagePart)]
        if images:
            entry["images"] = images
        if message.tool_calls:
            entry["tool_calls"] = [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in message.tool_calls
            ]
        if message.role == "tool":
            entry["tool_name"] = message.name or ""
        return entry

    def parse_response(self, payload: dict[str, Any], *, req: ChatRequest) -> Response:
        """Turn an ``/api/chat`` payload into a normalised response."""
        raw_message = payload.get("message") or {}
        text = str(raw_message.get("content") or "")
        parts = [TextPart(text=text)] if text else []
        calls = _parse_tool_calls(raw_message.get("tool_calls"))
        thinking = raw_message.get("thinking")

        reported = payload.get("eval_count") is not None
        usage = (
            _parse_usage(payload)
            if reported
            else Usage(
                input_tokens=estimate_tokens(req.messages),
                output_tokens=estimate_output_tokens(text),
            )
        )
        raw = dict(payload)
        if thinking:
            raw["thinking"] = thinking
        return Response(
            id=f"aaron-{req.request_id}",  # Ollama does not return an id of its own
            model=req.model,
            resolved_model=str(payload.get("model") or req.model_name),
            message=Message(role="assistant", content=list(parts), tool_calls=calls),
            stop_reason=_stop_reason(payload, calls),
            usage=usage,
            cost=req.registry.cost(req.model, usage, estimated=not reported),
            latency_ms=int(int(payload.get("total_duration") or 0) / 1_000_000),
            raw=raw,
        )

    def chunk_events(
        self,
        chunk: dict[str, Any],
        *,
        assembler: StreamAssembler,
        state: dict[str, Any],
        event: str | None,
    ) -> Iterator[StreamEvent]:
        """Translate one newline delimited chunk."""
        if chunk.get("error"):
            raise self.map_error(200, chunk, json.dumps(chunk))

        assembler.note(
            resolved_model=str(chunk["model"]) if chunk.get("model") else None,
            usage=_parse_usage(chunk) if chunk.get("eval_count") is not None else None,
        )
        message = chunk.get("message") or {}
        text = message.get("content")
        if isinstance(text, str) and text:
            yield _add(assembler, TextEvent(text=text))
        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking:
            yield _add(assembler, ThinkingEvent(text=thinking))

        # Ollama sends each tool call complete in a single chunk, with no id.
        counter = state.get("calls", 0)
        for call in _parse_tool_calls(message.get("tool_calls"), start=counter):
            counter += 1
            yield _add(assembler, ToolCallEndEvent(call=call))
        state["calls"] = counter

        if chunk.get("done"):
            assembler.note(
                stop_reason=normalise_stop_reason(chunk.get("done_reason"), mapping=_STOP_REASONS)
            )

    def error_fields(
        self, payload: dict[str, Any] | None, text: str
    ) -> tuple[str, str | None, str | None, float | None]:
        """Ollama reports ``{"error": "message"}``, sometimes with a model hint."""
        if payload and isinstance(payload.get("error"), str):
            message = str(payload["error"])
            return message, None, None, None
        return super().error_fields(payload, text)

    def map_error(self, status: int, payload: dict[str, Any] | None, text: str) -> Any:
        """Map an Ollama error, turning a missing model into actionable advice."""
        error = super().map_error(status, payload, text)
        message = error.message.lower()
        # Ollama's own wording is "try pulling it first", which never names the
        # command, so the check is for the command rather than for the word "pull".
        if "not found" in message and "ollama pull" not in message:
            error.message = (
                f"{error.message}. Run 'ollama pull <model>' to download it, "
                "or 'ollama list' to see what is already installed."
            )
        return error


def _add(assembler: StreamAssembler, event: StreamEvent) -> StreamEvent:
    assembler.add(event)
    return event


def _stop_reason(payload: dict[str, Any], calls: list[ToolCall]) -> StopReason:
    """A tool call means tool_use, whatever Ollama put in done_reason."""
    reason = normalise_stop_reason(payload.get("done_reason"), mapping=_STOP_REASONS)
    return "tool_use" if calls and reason in ("stop", "other") else reason


def _parse_tool_calls(raw: Any, *, start: int = 0) -> list[ToolCall]:
    """Read Ollama tool calls, which arrive with parsed arguments and no id."""
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for offset, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        arguments = function.get("arguments")
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{start + offset}"),
                name=str(function.get("name") or ""),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return calls


def _parse_usage(payload: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=int(payload.get("prompt_eval_count") or 0),
        output_tokens=int(payload.get("eval_count") or 0),
    )
