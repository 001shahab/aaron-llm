# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Anthropic, against ``POST /v1/messages``.

System prompts go to the native ``system`` field, tool results become
``tool_result`` content blocks, and consecutive same role turns are merged because
the API rejects them. Prompt caching is reached through ``provider_options``, for
example ``provider_options={"cache_control": {"type": "ephemeral"}}``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ..errors import ProviderError
from ..request import ChatRequest, merge_consecutive
from ..stream import (
    StreamAssembler,
    StreamEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    parse_tool_arguments,
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

API_VERSION = "2023-06-01"

PDF_BETA = "pdfs-2024-09-25"

# Anthropic requires max_tokens, so a caller who omits it gets a sane ceiling.
DEFAULT_MAX_TOKENS = 4096

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_use",
    "refusal": "content_filter",
    "pause_turn": "other",
}


def _has_document(req: ChatRequest) -> bool:
    """Whether any message carries a PDF, which needs the documents beta header."""
    return any(isinstance(part, DocumentPart) for m in req.messages for part in m.content)


class AnthropicProvider(BaseProvider):
    """Translates to and from the Anthropic messages API."""

    name = "anthropic"
    default_base_url = "https://api.anthropic.com/v1"
    env_key = "ANTHROPIC_API_KEY"
    local = False
    wire = "sse"

    def build_request(self, req: ChatRequest, *, stream: bool) -> PreparedRequest:
        """Build the messages request.

        Args:
            req: The normalised request.
            stream: Whether to ask for an SSE stream.

        Returns:
            An inspectable prepared request.
        """
        body: dict[str, Any] = {
            "model": req.model_name,
            "max_tokens": req.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": [
                self.encode(message) for message in merge_consecutive(req.non_system_messages)
            ],
        }
        if req.system_text:
            body["system"] = req.system_text
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.stop:
            body["stop_sequences"] = req.stop
        if req.tools:
            body["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in req.tools
            ]
            choice = _tool_choice(req.tool_choice)
            if choice is not None:
                body["tool_choice"] = choice
        if req.json_schema is not None or req.json_mode:
            # The messages API has no response_format, so the documented approach is
            # a single tool the model is forced to call. extract() validates locally
            # either way, so a refusal to use it surfaces as a validation error.
            body.update(_json_via_tool(req))
        if stream:
            body["stream"] = True

        return PreparedRequest(
            method="POST",
            url=f"{self.base_url(req)}/messages",
            headers={
                "content-type": "application/json",
                "anthropic-version": API_VERSION,
                # Only opt into a beta when the request actually needs it.
                **({"anthropic-beta": PDF_BETA} if _has_document(req) else {}),
                "x-request-id": req.request_id,
                **req.extra_headers,
            },
            body=merge_options(body, req.provider_options),
            auth=({"x-api-key": req.api_key} if req.api_key else {}),
            stream=stream,
        )

    def encode(self, message: Message) -> dict[str, Any]:
        """Encode one message into an Anthropic message object."""
        if message.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.text,
                    }
                ],
            }

        blocks: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": part.data,
                        },
                    }
                )
            elif isinstance(part, DocumentPart):
                blocks.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": part.data,
                        },
                    }
                )
        for call in message.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )
        role = "assistant" if message.role == "assistant" else "user"
        return {"role": role, "content": blocks}

    def parse_response(self, payload: dict[str, Any], *, req: ChatRequest) -> Response:
        """Turn a messages payload into a normalised response.

        Args:
            payload: The decoded JSON body.
            req: The originating request.

        Returns:
            The normalised response. Thinking blocks are preserved on ``raw``.

        Raises:
            ProviderError: The payload has no content array.
        """
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise ProviderError(
                "Anthropic returned no content array",
                provider=self.name,
                model=req.model,
                raw=payload,
            )
        parts: list[TextPart] = []
        calls: list[ToolCall] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and block.get("text"):
                parts.append(TextPart(text=str(block["text"])))
            elif kind == "tool_use":
                calls.append(
                    ToolCall(
                        id=str(block.get("id") or ""),
                        name=str(block.get("name") or ""),
                        arguments=_as_object(block.get("input"), str(block.get("name") or ""), req),
                    )
                )

        usage = _parse_usage(payload.get("usage"))
        return Response(
            id=str(payload.get("id") or f"aaron-{req.request_id}"),
            model=req.model,
            resolved_model=str(payload.get("model") or req.model_name),
            message=Message(role="assistant", content=list(parts), tool_calls=calls),
            stop_reason=normalise_stop_reason(payload.get("stop_reason"), mapping=_STOP_REASONS),
            usage=usage,
            cost=req.registry.cost(req.model, usage, estimated=payload.get("usage") is None),
            latency_ms=0,
            raw=payload,
        )

    def chunk_events(
        self,
        chunk: dict[str, Any],
        *,
        assembler: StreamAssembler,
        state: dict[str, Any],
        event: str | None,
    ) -> Iterator[StreamEvent]:
        """Translate one SSE chunk of the messages stream."""
        kind = str(chunk.get("type") or event or "")
        blocks: dict[int, str] = state.setdefault("blocks", {})

        if kind == "message_start":
            message = chunk.get("message") or {}
            assembler.note(
                id=str(message.get("id")) if message.get("id") else None,
                resolved_model=str(message.get("model")) if message.get("model") else None,
                usage=_parse_usage(message.get("usage")) if message.get("usage") else None,
            )
        elif kind == "content_block_start":
            index = int(chunk.get("index", 0))
            block = chunk.get("content_block") or {}
            blocks[index] = str(block.get("type") or "text")
            if block.get("type") == "tool_use":
                yield _add(
                    assembler,
                    ToolCallStartEvent(
                        id=str(block.get("id") or f"call_{index}"),
                        name=str(block.get("name") or ""),
                    ),
                )
                state.setdefault("tool_ids", {})[index] = str(block.get("id") or f"call_{index}")
        elif kind == "content_block_delta":
            yield from self._delta_events(chunk, assembler, state)
        elif kind == "message_delta":
            delta = chunk.get("delta") or {}
            assembler.note(
                stop_reason=normalise_stop_reason(delta.get("stop_reason"), mapping=_STOP_REASONS)
                if delta.get("stop_reason")
                else None,
                usage=_parse_usage(chunk["usage"]) if chunk.get("usage") else None,
            )
        elif kind == "error":
            raise self.map_error(200, chunk, json.dumps(chunk))

    def _delta_events(
        self, chunk: dict[str, Any], assembler: StreamAssembler, state: dict[str, Any]
    ) -> Iterator[StreamEvent]:
        delta = chunk.get("delta") or {}
        kind = delta.get("type")
        if kind == "text_delta" and delta.get("text"):
            yield _add(assembler, TextEvent(text=str(delta["text"])))
        elif kind == "thinking_delta" and delta.get("thinking"):
            yield _add(assembler, ThinkingEvent(text=str(delta["thinking"])))
        elif kind == "input_json_delta":
            index = int(chunk.get("index", 0))
            call_id = state.get("tool_ids", {}).get(index, f"call_{index}")
            fragment = str(delta.get("partial_json") or "")
            if fragment:
                yield _add(assembler, ToolCallDeltaEvent(id=call_id, arguments_json=fragment))

    def error_fields(
        self, payload: dict[str, Any] | None, text: str
    ) -> tuple[str, str | None, str | None, float | None]:
        """Read Anthropic's ``{"type": "error", "error": {"type", "message"}}`` shape."""
        if not payload:
            return (text or "Anthropic returned an empty error body"), None, None, None
        error = payload.get("error")
        if isinstance(error, dict):
            return (
                str(error.get("message") or text or "Anthropic error"),
                str(error.get("type")) if error.get("type") else None,
                str(payload.get("request_id")) if payload.get("request_id") else None,
                None,
            )
        return super().error_fields(payload, text)


def _add(assembler: StreamAssembler, event: StreamEvent) -> StreamEvent:
    assembler.add(event)
    return event


def _tool_choice(choice: str | None) -> dict[str, Any] | None:
    if choice is None or choice == "auto":
        return None
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    return {"type": "tool", "name": choice}


def _json_via_tool(req: ChatRequest) -> dict[str, Any]:
    """Force a single tool call whose input is the requested JSON document."""
    schema = req.json_schema.schema if req.json_schema else {"type": "object"}
    name = req.json_schema.name if req.json_schema else "json_output"
    return {
        "tools": [
            {
                "name": name,
                "description": "Return the requested structured document.",
                "input_schema": schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": name},
    }


def _as_object(value: Any, tool_name: str, req: ChatRequest) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return parse_tool_arguments(value, tool_name=tool_name, req=req)
    return {}


def _parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    return Usage(
        input_tokens=int(raw.get("input_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        cached_input_tokens=int(raw.get("cache_read_input_tokens") or 0),
    )
