# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""The normalised request every provider receives.

A :class:`ChatRequest` is fully resolved: the model string is split, credentials
and base URL are decided, and messages are real :class:`~aaron.types.Message`
objects. Providers only translate it, they never make policy or configuration
decisions of their own.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .errors import InvalidRequest
from .registry import Registry, default_registry
from .types import Message, SecretValue, TextPart, Tool

# What a caller may pass in place of a message list.
Prompt = str | Message | Sequence[Message]

ToolChoice = Literal["auto", "none", "required"] | str

_MAX_TEMPERATURE = 2.0


@dataclass(frozen=True, slots=True)
class JsonSchemaSpec:
    """A JSON Schema the model must conform to.

    Args:
        name: Schema name, required by some providers.
        schema: A JSON Schema object.
        strict: Ask the provider to enforce the schema natively where it can.
    """

    name: str
    schema: dict[str, Any]
    strict: bool = True


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """A fully resolved chat request, ready for a provider to translate.

    Attributes:
        model: The canonical ``provider/model`` string the caller asked for, after
            alias resolution. This is what appears on ``Response.model``.
        provider: Provider name, the part before the first slash.
        model_name: The bare model id to send to the provider.
        base_url: Resolved endpoint root, without a trailing slash.
        api_key: The credential, masked in every repr.
        provider_options: Merged verbatim into the outgoing JSON body. This is the
            escape hatch that keeps the abstraction from blocking a provider
            feature.
    """

    model: str
    provider: str
    model_name: str
    messages: list[Message]
    base_url: str
    api_key: SecretValue | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    tool_choice: ToolChoice | None = None
    json_schema: JsonSchemaSpec | None = None
    json_mode: bool = False
    provider_options: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout: float | None = None
    connect_timeout: float | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: dict[str, str] = field(default_factory=dict)
    registry: Registry = field(default_factory=default_registry, repr=False, compare=False)

    def with_messages(self, messages: Sequence[Message]) -> ChatRequest:
        """Return a copy carrying different messages, used by the redaction pass."""
        return replace(self, messages=list(messages))

    @property
    def system_text(self) -> str:
        """Every system message, joined with blank lines. Empty when there is none."""
        return "\n\n".join(m.text for m in self.messages if m.role == "system" and m.text)

    @property
    def non_system_messages(self) -> list[Message]:
        """The conversation without system messages."""
        return [m for m in self.messages if m.role != "system"]

    @property
    def input_chars(self) -> int:
        """Total characters of text across every message, used for audit and budgets."""
        return sum(
            len(part.text)
            for message in self.messages
            for part in message.content
            if isinstance(part, TextPart)
        )

    def __repr__(self) -> str:
        return (
            f"ChatRequest(model={self.model!r}, messages={len(self.messages)}, "
            f"tools={len(self.tools)}, base_url={self.base_url!r})"
        )


def split_model(model: str) -> tuple[str, str]:
    """Split a canonical model string into its provider and model name.

    Only the first slash is significant, so a namespaced model id such as
    ``openai_compat/mistralai/mistral-large`` keeps its slash.

    Args:
        model: A ``provider/model`` string.

    Returns:
        The provider name and the bare model name.

    Raises:
        InvalidRequest: The string has no provider prefix or an empty half.
    """
    if "/" not in model:
        raise InvalidRequest(
            f"model must be 'provider/model', got {model!r}. "
            "Register an alias with client.alias() if you want a short name."
        )
    provider, _, name = model.partition("/")
    if not provider or not name:
        raise InvalidRequest(f"model must be 'provider/model', got {model!r}")
    return provider, name


def normalise_messages(prompt: Prompt) -> list[Message]:
    """Coerce whatever the caller passed into a list of messages.

    Args:
        prompt: A plain string, a single message, or a sequence of messages.

    Returns:
        A new list of messages, never empty.

    Raises:
        InvalidRequest: The prompt was empty, or contained a non message item.
    """
    if isinstance(prompt, str):
        if not prompt:
            raise InvalidRequest("prompt string is empty")
        return [Message.user(prompt)]
    if isinstance(prompt, Message):
        return [prompt]
    messages = list(prompt)
    if not messages:
        raise InvalidRequest("messages is empty")
    for index, message in enumerate(messages):
        if not isinstance(message, Message):
            raise InvalidRequest(
                f"messages[{index}] is {type(message).__name__}, expected aaron.Message"
            )
    return messages


def validate(req: ChatRequest) -> None:
    """Check a request for problems no provider should have to think about.

    Args:
        req: The request to check.

    Raises:
        InvalidRequest: A field is out of range, or the conversation is malformed.
    """
    if req.temperature is not None and not 0.0 <= req.temperature <= _MAX_TEMPERATURE:
        raise InvalidRequest(f"temperature must be between 0 and {_MAX_TEMPERATURE}")
    if req.top_p is not None and not 0.0 <= req.top_p <= 1.0:
        raise InvalidRequest("top_p must be between 0 and 1")
    if req.max_tokens is not None and req.max_tokens <= 0:
        raise InvalidRequest("max_tokens must be positive")
    if not req.non_system_messages:
        raise InvalidRequest("a request needs at least one non system message")

    names = [tool.name for tool in req.tools]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise InvalidRequest(f"duplicate tool names: {sorted(duplicates)}")

    if req.json_schema is not None and req.tools:
        raise InvalidRequest(
            "a json schema and tools cannot be combined, "
            "run the tool loop first and then extract from the result"
        )

    for message in req.messages:
        if message.role == "tool" and not message.tool_call_id:
            raise InvalidRequest("a tool message needs tool_call_id")
        if message.role != "assistant" and message.tool_calls:
            raise InvalidRequest(f"a {message.role} message cannot carry tool_calls")

    _validate_tool_pairing(req.messages)


def _validate_tool_pairing(messages: Sequence[Message]) -> None:
    """Every tool result must answer a tool call that appeared earlier."""
    seen: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            seen.update(call.id for call in message.tool_calls)
        elif message.role == "tool" and message.tool_call_id not in seen:
            raise InvalidRequest(
                f"tool result {message.tool_call_id!r} does not answer any earlier tool call"
            )


def merge_consecutive(messages: Sequence[Message]) -> list[Message]:
    """Merge neighbouring messages that share a role.

    Anthropic and Google reject two user turns in a row, so providers that need
    this call it during translation. Messages carrying tool calls or tool results
    are never merged, because their ids must stay distinguishable.

    Args:
        messages: The conversation.

    Returns:
        A new list where mergeable neighbours have been combined.
    """
    merged: list[Message] = []
    for message in messages:
        mergeable = not message.tool_calls and message.tool_call_id is None
        if merged and mergeable and merged[-1].role == message.role and not merged[-1].tool_calls:
            previous = merged[-1]
            if previous.tool_call_id is None:
                merged[-1] = previous.model_copy(
                    update={"content": [*previous.content, *message.content]}
                )
                continue
        merged.append(message)
    return merged


def estimate_tokens(messages: Iterable[Message]) -> int:
    """Estimate the input token count without a tokeniser.

    Four characters per token is the usual rule of thumb for English text, and an
    attachment is charged a flat approximation of its encoded size. This is only
    used for policy budgets and for filling in usage a provider never reported, so
    a cheap estimate that never needs a model download is the right trade.

    Args:
        messages: The conversation to measure.

    Returns:
        An approximate token count, always at least 1 for a non empty input.
    """
    total = 0
    for message in messages:
        total += 4  # per message role and delimiter overhead
        for part in message.content:
            if isinstance(part, TextPart):
                total += len(part.text) // 4 + 1
            else:
                # Base64 grows the payload by a third; providers bill images far
                # more cheaply than that, so scale it down hard.
                total += len(part.data) // 750 + 1
        for call in message.tool_calls:
            total += len(json.dumps(call.arguments, default=str)) // 4 + len(call.name) // 4 + 4
    return max(total, 1)


def estimate_output_tokens(text: str) -> int:
    """Estimate output tokens from generated text, for providers that omit usage."""
    return max(len(text) // 4, 1) if text else 0


def body_sha256(body: dict[str, Any]) -> str:
    """Hash a request or response body for the audit trail.

    The hash is over a canonical JSON encoding, so the same logical body always
    produces the same digest regardless of key order.

    Args:
        body: The JSON body.

    Returns:
        A lowercase hex sha256 digest.
    """
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
