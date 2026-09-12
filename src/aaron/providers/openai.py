# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""OpenAI, against ``POST /v1/chat/completions``.

Written against the documented HTTP API, with no vendor SDK. The reasoning families
take ``max_completion_tokens`` instead of ``max_tokens`` and reject sampling
parameters, which is handled here rather than pushed onto the caller.
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
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    parse_tool_arguments,
)
from ..types import (
    DocumentPart,
    ImagePart,
    Message,
    Response,
    SecretValue,
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
    "max_tokens": "length",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
}

# Families that take max_completion_tokens and reject temperature and top_p.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


class OpenAIProvider(BaseProvider):
    """Translates to and from the OpenAI chat completions API."""

    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    env_key = "OPENAI_API_KEY"
    local = False
    wire = "sse"

    def build_request(self, req: ChatRequest, *, stream: bool) -> PreparedRequest:
        """Build the chat completions request.

        Args:
            req: The normalised request.
            stream: Whether to ask for a server sent event stream.

        Returns:
            An inspectable prepared request, with the credential kept separate.
        """
        body: dict[str, Any] = {
            "model": req.model_name,
            "messages": [entry for message in req.messages for entry in self.encode(message)],
        }
        reasoning = self.is_reasoning_model(req.model_name)

        if req.max_tokens is not None:
            body["max_completion_tokens" if reasoning else "max_tokens"] = req.max_tokens
        if req.temperature is not None and not reasoning:
            body["temperature"] = req.temperature
        if req.top_p is not None and not reasoning:
            body["top_p"] = req.top_p
        if req.stop:
            body["stop"] = req.stop
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
            body["tool_choice"] = _tool_choice(req.tool_choice)
        if req.json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": req.json_schema.name,
                    "schema": req.json_schema.schema,
                    "strict": req.json_schema.strict,
                },
            }
        elif req.json_mode:
            body["response_format"] = {"type": "json_object"}
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}

        return PreparedRequest(
            method="POST",
            url=f"{self.base_url(req)}/chat/completions",
            headers={
                "content-type": "application/json",
                "idempotency-key": req.request_id,
                **req.extra_headers,
            },
            body=merge_options(body, req.provider_options),
            auth=self.auth(req),
            stream=stream,
        )

    def auth(self, req: ChatRequest) -> dict[str, SecretValue]:
        """Bearer token header, or nothing when the endpoint needs no key."""
        if req.api_key is None:
            return {}
        return {"authorization": SecretValue(f"Bearer {req.api_key.get()}")}

    @staticmethod
    def is_reasoning_model(model_name: str) -> bool:
        """Whether the model belongs to a family with the reasoning parameter set."""
        return model_name.startswith(_REASONING_PREFIXES)

    def encode(self, message: Message) -> list[dict[str, Any]]:
        """Encode one message into the OpenAI shape."""
        if message.role == "tool":
            return [{"role": "tool", "tool_call_id": message.tool_call_id, "content": message.text}]

        if message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.text or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, default=str),
                        },
                    }
                    for call in message.tool_calls
                ]
            return [entry]

        if message.role == "system":
            return [{"role": "system", "content": message.text}]

        content: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.media_type};base64,{part.data}"},
                    }
                )
            elif isinstance(part, DocumentPart):
                content.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": "document.pdf",
                            "file_data": f"data:{part.media_type};base64,{part.data}",
                        },
                    }
                )
        named = {"name": message.name} if message.name else {}
        return [{"role": message.role, "content": content, **named}]

    def parse_response(self, payload: dict[str, Any], *, req: ChatRequest) -> Response:
        """Turn a chat completion payload into a normalised response.

        Args:
            payload: The decoded JSON body.
            req: The originating request.

        Returns:
            The normalised response, with the untouched payload on ``raw``.

        Raises:
            InvalidRequest: The payload carries no choices.
            ToolArgumentError: A tool call's arguments are not a JSON object.
        """
        choices = payload.get("choices") or []
        if not choices:
            raise InvalidRequest(
                "OpenAI returned no choices", provider=self.name, model=req.model, raw=payload
            )
        choice = choices[0]
        raw_message = choice.get("message") or {}
        message = Message(
            role="assistant",
            content=list(_text_parts(raw_message.get("content"))),
            tool_calls=_parse_tool_calls(raw_message.get("tool_calls"), req),
        )
        reported = payload.get("usage") is not None
        usage = (
            _parse_usage(payload.get("usage"))
            if reported
            else Usage(
                input_tokens=estimate_tokens(req.messages),
                output_tokens=estimate_output_tokens(message.text),
            )
        )
        return Response(
            id=str(payload.get("id") or f"aaron-{req.request_id}"),
            model=req.model,
            resolved_model=str(payload.get("model") or req.model_name),
            message=message,
            stop_reason=normalise_stop_reason(choice.get("finish_reason"), mapping=_STOP_REASONS),
            usage=usage,
            cost=req.registry.cost(req.model, usage, estimated=not reported),
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
        """Translate one streamed chunk.

        Raises:
            ProviderError: The stream carried an error object instead of a chunk,
                which is how a mid stream failure is reported on a 200 response.
        """
        if chunk.get("error"):
            raise self.map_error(200, chunk, json.dumps(chunk))

        assembler.note(
            id=str(chunk["id"]) if chunk.get("id") else None,
            resolved_model=str(chunk["model"]) if chunk.get("model") else None,
            usage=_parse_usage(chunk["usage"]) if chunk.get("usage") else None,
        )
        # OpenAI identifies streamed tool calls by array index, not by id.
        index_to_id: dict[int, str] = state.setdefault("tool_call_ids", {})

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                yield _add(assembler, TextEvent(text=text))
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                yield _add(assembler, ThinkingEvent(text=reasoning))

            for item in delta.get("tool_calls") or []:
                if isinstance(item, dict):
                    yield from _tool_call_events(item, assembler, index_to_id)

            finish = choice.get("finish_reason")
            if finish:
                assembler.note(stop_reason=normalise_stop_reason(finish, mapping=_STOP_REASONS))


def _add(assembler: StreamAssembler, event: StreamEvent) -> StreamEvent:
    """Fold an event into the assembler and hand it straight back for yielding."""
    assembler.add(event)
    return event


def _tool_call_events(
    item: dict[str, Any], assembler: StreamAssembler, index_to_id: dict[int, str]
) -> Iterator[StreamEvent]:
    index = int(item.get("index", 0))
    function = item.get("function") or {}
    if item.get("id"):
        index_to_id[index] = str(item["id"])
    call_id = index_to_id.setdefault(index, f"call_{index}")
    if function.get("name"):
        yield _add(assembler, ToolCallStartEvent(id=call_id, name=str(function["name"])))
    arguments = function.get("arguments")
    if isinstance(arguments, str) and arguments:
        yield _add(assembler, ToolCallDeltaEvent(id=call_id, arguments_json=arguments))


def _tool_choice(choice: str | None) -> Any:
    if choice is None or choice == "auto":
        return "auto"
    if choice in ("none", "required"):
        return choice
    return {"type": "function", "function": {"name": choice}}


def _text_parts(content: Any) -> Iterator[TextPart]:
    """Accept both a plain string and the parts list some compatible endpoints send."""
    if isinstance(content, str) and content:
        yield TextPart(text=content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                yield TextPart(text=str(item["text"]))


def _parse_tool_calls(raw: Any, req: ChatRequest) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        name = str(function.get("name") or "")
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{index}"),
                name=name,
                arguments=parse_tool_arguments(
                    str(function.get("arguments") or ""), tool_name=name, req=req
                ),
            )
        )
    return calls


def _parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    return Usage(
        input_tokens=int(raw.get("prompt_tokens") or 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cached_input_tokens=int(prompt_details.get("cached_tokens") or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
    )
