# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Google Gemini, against ``generateContent`` and ``streamGenerateContent``.

The shapes differ more here than anywhere else: turns are ``contents`` with
``parts``, the assistant role is ``model``, tools are ``functionDeclarations``, and
tool results are ``functionResponse`` parts. Streaming uses ``?alt=sse`` so the same
server sent event decoder works as for the other providers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

from ..errors import ProviderError
from ..request import ChatRequest, merge_consecutive
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
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "MALFORMED_FUNCTION_CALL": "error",
    "OTHER": "other",
}

# Gemini rejects JSON Schema keywords it does not implement.
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {"additionalProperties", "$schema", "$defs", "definitions", "title", "default", "examples"}
)


class GoogleProvider(BaseProvider):
    """Translates to and from the Gemini generative language API."""

    name = "google"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    env_key = "GOOGLE_API_KEY"
    local = False
    wire = "sse"

    def build_request(self, req: ChatRequest, *, stream: bool) -> PreparedRequest:
        """Build a generateContent or streamGenerateContent request.

        Args:
            req: The normalised request.
            stream: Whether to use the streaming endpoint with ``alt=sse``.

        Returns:
            An inspectable prepared request. The key travels in the
            ``x-goog-api-key`` header, never in the query string, so it cannot end
            up in a proxy access log.
        """
        generation: dict[str, Any] = {}
        if req.max_tokens is not None:
            generation["maxOutputTokens"] = req.max_tokens
        if req.temperature is not None:
            generation["temperature"] = req.temperature
        if req.top_p is not None:
            generation["topP"] = req.top_p
        if req.stop:
            generation["stopSequences"] = req.stop
        if req.json_schema is not None:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = _clean_schema(req.json_schema.schema)
        elif req.json_mode:
            generation["responseMimeType"] = "application/json"

        # Gemini pairs a result with its call by function name, not by an id, so the
        # name has to be recovered from the assistant turn that made the call.
        tool_names = {call.id: call.name for m in req.messages for call in m.tool_calls}
        body: dict[str, Any] = {
            "contents": [
                self.encode(message, tool_names)
                for message in merge_consecutive(req.non_system_messages)
            ]
        }
        if generation:
            body["generationConfig"] = generation
        if req.system_text:
            body["systemInstruction"] = {"parts": [{"text": req.system_text}]}
        if req.tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": _clean_schema(tool.parameters),
                        }
                        for tool in req.tools
                    ]
                }
            ]
            config = _tool_config(req.tool_choice)
            if config is not None:
                body["toolConfig"] = config

        method = "streamGenerateContent" if stream else "generateContent"
        suffix = "?alt=sse" if stream else ""
        return PreparedRequest(
            method="POST",
            url=f"{self.base_url(req)}/models/{req.model_name}:{method}{suffix}",
            headers={
                "content-type": "application/json",
                "x-request-id": req.request_id,
                **req.extra_headers,
            },
            body=merge_options(body, req.provider_options),
            auth=({"x-goog-api-key": req.api_key} if req.api_key else {}),
            stream=stream,
        )

    def encode(self, message: Message, tool_names: Mapping[str, str]) -> dict[str, Any]:
        """Encode one message into a Gemini ``contents`` entry.

        Args:
            message: The message to encode.
            tool_names: Call id to function name, used to label a tool result.

        Returns:
            One ``contents`` entry.
        """
        if message.role == "tool":
            call_id = message.tool_call_id or ""
            return {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": message.name or tool_names.get(call_id) or call_id or "tool",
                            "response": _wrap_response(message.text),
                        }
                    }
                ],
            }

        parts: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append({"text": part.text})
            elif isinstance(part, ImagePart | DocumentPart):
                parts.append({"inlineData": {"mimeType": part.media_type, "data": part.data}})
        for call in message.tool_calls:
            parts.append({"functionCall": {"name": call.name, "args": call.arguments}})
        return {"role": "model" if message.role == "assistant" else "user", "parts": parts}

    def parse_response(self, payload: dict[str, Any], *, req: ChatRequest) -> Response:
        """Turn a generateContent payload into a normalised response.

        Raises:
            InvalidRequest: The prompt itself was blocked, which Gemini reports as a
                ``promptFeedback.blockReason`` with no candidates at all.
        """
        candidates = payload.get("candidates") or []
        if not candidates:
            blocked = (payload.get("promptFeedback") or {}).get("blockReason")
            if blocked:
                raise self.map_error(
                    400, {"error": {"message": f"prompt blocked by safety filter: {blocked}"}}, ""
                )
            raise ProviderError(
                "Gemini returned no candidates", provider=self.name, model=req.model, raw=payload
            )

        candidate = candidates[0]
        parts, calls, thinking = _split_parts(candidate.get("content") or {})
        usage = _parse_usage(payload.get("usageMetadata"))
        raw = dict(payload)
        if thinking:
            raw["thinking"] = thinking
        return Response(
            id=str(payload.get("responseId") or f"aaron-{req.request_id}"),
            model=req.model,
            resolved_model=str(payload.get("modelVersion") or req.model_name),
            message=Message(role="assistant", content=list(parts), tool_calls=calls),
            stop_reason=normalise_stop_reason(candidate.get("finishReason"), mapping=_STOP_REASONS),
            usage=usage,
            cost=req.registry.cost(
                req.model, usage, estimated=payload.get("usageMetadata") is None
            ),
            latency_ms=0,
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
        """Translate one streamed candidate delta."""
        if chunk.get("error"):
            raise self.map_error(200, chunk, json.dumps(chunk))

        assembler.note(
            id=str(chunk["responseId"]) if chunk.get("responseId") else None,
            resolved_model=str(chunk["modelVersion"]) if chunk.get("modelVersion") else None,
            usage=_parse_usage(chunk["usageMetadata"]) if chunk.get("usageMetadata") else None,
        )
        for candidate in chunk.get("candidates") or []:
            parts, calls, thinking = _split_parts(candidate.get("content") or {})
            for part in parts:
                yield _add(assembler, TextEvent(text=part.text))
            if thinking:
                yield _add(assembler, ThinkingEvent(text=thinking))
            # Gemini sends each function call complete in one chunk, so there is
            # nothing to accumulate and the end event can be emitted immediately.
            for index, call in enumerate(calls):
                numbered = call if call.id else call.model_copy(update={"id": f"call_{index}"})
                yield _add(assembler, ToolCallEndEvent(call=numbered))
            if candidate.get("finishReason"):
                assembler.note(
                    stop_reason=normalise_stop_reason(
                        candidate["finishReason"], mapping=_STOP_REASONS
                    )
                )

    def error_fields(
        self, payload: dict[str, Any] | None, text: str
    ) -> tuple[str, str | None, str | None, float | None]:
        """Read Google's ``{"error": {"code", "message", "status"}}`` shape."""
        if not payload:
            return (text or "Gemini returned an empty error body"), None, None, None
        error = payload.get("error")
        if isinstance(error, dict):
            return (
                str(error.get("message") or text or "Gemini error"),
                str(error.get("status")) if error.get("status") else None,
                None,
                None,
            )
        return super().error_fields(payload, text)


def _add(assembler: StreamAssembler, event: StreamEvent) -> StreamEvent:
    assembler.add(event)
    return event


def _split_parts(content: dict[str, Any]) -> tuple[list[TextPart], list[ToolCall], str]:
    """Split a Gemini content object into text, tool calls and reasoning text."""
    parts: list[TextPart] = []
    calls: list[ToolCall] = []
    thinking: list[str] = []
    for index, part in enumerate(content.get("parts") or []):
        if not isinstance(part, dict):
            continue
        if part.get("thought") and part.get("text"):
            thinking.append(str(part["text"]))
        elif part.get("text"):
            parts.append(TextPart(text=str(part["text"])))
        elif isinstance(part.get("functionCall"), dict):
            call = part["functionCall"]
            arguments = call.get("args")
            calls.append(
                ToolCall(
                    id=str(call.get("id") or f"call_{index}"),
                    name=str(call.get("name") or ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
    return parts, calls, "".join(thinking)


def _wrap_response(text: str) -> dict[str, Any]:
    """Gemini requires a functionResponse payload to be an object."""
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {"result": text}
    return decoded if isinstance(decoded, dict) else {"result": decoded}


def _tool_config(choice: str | None) -> dict[str, Any] | None:
    if choice is None or choice == "auto":
        return None
    if choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if choice == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}
    return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [choice]}}


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip JSON Schema keywords Gemini rejects, recursively."""
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_schema(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    return Usage(
        input_tokens=int(raw.get("promptTokenCount") or 0),
        output_tokens=int(raw.get("candidatesTokenCount") or 0),
        cached_input_tokens=int(raw.get("cachedContentTokenCount") or 0),
        reasoning_tokens=int(raw.get("thoughtsTokenCount") or 0),
    )
