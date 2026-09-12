# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Core data types shared by every provider.

Nothing in this module knows about a specific provider. Providers translate to and
from these shapes, so a caller can move between them without touching their code.
"""

from __future__ import annotations

import base64
import binascii
import inspect
import json
import re
import typing
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import InvalidRequest

Role = Literal["system", "user", "assistant", "tool"]

StopReason = Literal["stop", "length", "tool_use", "content_filter", "error", "other"]

# Media types we can prove from magic bytes, mapped from the signature prefix.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

_SUFFIX_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".pdf": "application/pdf",
}

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/\s]*={0,2}$")


class SecretValue:
    """A string whose repr is masked, used for credentials.

    The value is only reachable through :meth:`get`, which makes every read site
    grep-able. ``repr``, ``str`` and pydantic serialisation all mask it, so a key
    cannot reach a log line, an exception or an audit record by accident.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        """Return the raw secret. Never log the result."""
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def masked(self) -> str:
        """Return a short masked form, safe to display."""
        if not self._value:
            return "****"
        return f"****{self._value[-4:]}" if len(self._value) > 8 else "****"

    def __repr__(self) -> str:
        return f"SecretValue({self.masked()})"

    def __str__(self) -> str:
        return self.masked()


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextPart(_Frozen):
    """A run of plain text."""

    type: Literal["text"] = "text"
    text: str


class ImagePart(_Frozen):
    """An image, always inline base64. URL passthrough is not supported in v1."""

    type: Literal["image"] = "image"
    media_type: str
    data: str


class DocumentPart(_Frozen):
    """A document, currently PDF only, always inline base64."""

    type: Literal["document"] = "document"
    media_type: str
    data: str
    name: str | None = None


ContentPart = Annotated[TextPart | ImagePart | DocumentPart, Field(discriminator="type")]


class ToolCall(_Frozen):
    """A tool invocation requested by the model, with arguments already parsed."""

    id: str
    name: str
    arguments: dict[str, Any]


def _blob_name(source: str | bytes | Path) -> str | None:
    """The file name, when the attachment came from a path we can name."""
    if isinstance(source, bytes):
        return None
    name = Path(source).name
    return name if name and Path(source).suffix else None


def _read_blob(source: str | bytes | Path, *, kind: str) -> tuple[str, str]:
    """Turn a path, raw bytes or a base64 string into ``(media_type, base64_data)``."""
    if isinstance(source, bytes):
        return _sniff_media_type(source, kind=kind), base64.b64encode(source).decode("ascii")

    if isinstance(source, Path):
        return _read_path(source, kind=kind)

    if source.startswith("data:"):
        header, _, payload = source.partition(",")
        if not payload or ";base64" not in header:
            raise InvalidRequest("only base64 data URLs are supported")
        return header.removeprefix("data:").split(";", 1)[0] or "application/octet-stream", payload

    if source.startswith(("http://", "https://")):
        raise InvalidRequest(
            f"remote {kind} URLs are not supported in v1, download the bytes and pass them instead"
        )

    candidate = Path(source)
    # A short string with a known media suffix is a path; anything else that looks
    # like base64 is treated as already encoded data.
    if candidate.suffix.lower() in _SUFFIX_MEDIA_TYPES or candidate.exists():
        return _read_path(candidate, kind=kind)

    if _BASE64_RE.match(source):
        try:
            raw = base64.b64decode(source, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidRequest(f"{kind} string is neither a readable path nor base64") from exc
        return _sniff_media_type(raw, kind=kind), source

    raise InvalidRequest(f"{kind} string is neither a readable path nor base64: {source[:40]!r}")


def _read_path(path: Path, *, kind: str) -> tuple[str, str]:
    if not path.is_file():
        raise InvalidRequest(f"{kind} file not found: {path}")
    raw = path.read_bytes()
    media_type = _SUFFIX_MEDIA_TYPES.get(path.suffix.lower()) or _sniff_media_type(raw, kind=kind)
    return media_type, base64.b64encode(raw).decode("ascii")


def _sniff_media_type(raw: bytes, *, kind: str) -> str:
    if kind == "document":
        if raw.startswith(b"%PDF"):
            return "application/pdf"
        raise InvalidRequest("only PDF documents are supported in v1")
    for signature, media_type in _IMAGE_SIGNATURES:
        if raw.startswith(signature):
            return media_type
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    raise InvalidRequest(
        "could not determine image media type, supported types are png, jpeg, webp and gif"
    )


class Message(BaseModel):
    """One turn in a conversation.

    Prefer the classmethod constructors over building this directly; they accept
    plain strings and file paths and handle base64 encoding and media sniffing.

    Frozen, like every type here. Use ``model_copy(update=...)`` to derive a variant.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: list[ContentPart] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, text: str) -> Message:
        """Build a system message.

        Args:
            text: The system instruction.

        Returns:
            A message with role ``system``.
        """
        return cls(role="system", content=[TextPart(text=text)])

    @classmethod
    def user(
        cls,
        text: str = "",
        images: Sequence[str | bytes | Path] = (),
        documents: Sequence[str | bytes | Path] = (),
    ) -> Message:
        """Build a user message, encoding any attachments.

        Args:
            text: The prompt text. May be empty when only attachments are sent.
            images: Local paths, raw bytes, base64 strings or base64 data URLs.
            documents: The same, for PDFs.

        Returns:
            A message with role ``user``.

        Raises:
            InvalidRequest: An attachment could not be read, or its media type is
                unsupported, or a remote URL was passed.
        """
        parts: list[ContentPart] = []
        if text:
            parts.append(TextPart(text=text))
        for image in images:
            media_type, data = _read_blob(image, kind="image")
            parts.append(ImagePart(media_type=media_type, data=data))
        for document in documents:
            media_type, data = _read_blob(document, kind="document")
            parts.append(DocumentPart(media_type=media_type, data=data, name=_blob_name(document)))
        if not parts:
            raise InvalidRequest("a user message needs text, an image or a document")
        return cls(role="user", content=parts)

    @classmethod
    def assistant(cls, text: str = "", tool_calls: Sequence[ToolCall] = ()) -> Message:
        """Build an assistant message, for replaying a conversation.

        Args:
            text: Assistant text, may be empty when the turn was only tool calls.
            tool_calls: Tool calls the assistant made.

        Returns:
            A message with role ``assistant``.
        """
        parts: list[ContentPart] = [TextPart(text=text)] if text else []
        return cls(role="assistant", content=parts, tool_calls=list(tool_calls))

    @classmethod
    def tool(cls, tool_call_id: str, result: Any) -> Message:
        """Build a tool result message.

        Args:
            tool_call_id: The id of the :class:`ToolCall` being answered.
            result: The result. Strings are passed through, anything else is JSON
                encoded with a readable fallback for non serialisable objects.

        Returns:
            A message with role ``tool``.
        """
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        return cls(role="tool", content=[TextPart(text=text)], tool_call_id=tool_call_id)

    @property
    def text(self) -> str:
        """Concatenate every text part in this message."""
        return "".join(part.text for part in self.content if isinstance(part, TextPart))


_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_schema_for(annotation: Any) -> dict[str, Any]:
    """Map a python annotation onto a small JSON Schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = typing.get_origin(annotation)
    if origin is Literal:
        options = list(typing.get_args(annotation))
        schema: dict[str, Any] = {"enum": options}
        if all(isinstance(option, str) for option in options):
            schema["type"] = "string"
        return schema
    if origin in (list, Sequence):
        args = typing.get_args(annotation)
        items = _json_schema_for(args[0]) if args else {}
        return {"type": "array", "items": items} if items else {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    if origin is typing.Union or origin is type(int | str):
        # Optional[X] is Union[X, None]; drop the None and describe what is left.
        members = tuple(a for a in typing.get_args(annotation) if a is not type(None))
        if len(members) == 1:
            return _json_schema_for(members[0])
        return {"anyOf": [_json_schema_for(member) for member in members]}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _inline_model_schema(annotation)
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    return {}


def _inline_model_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a model's JSON Schema with ``$defs`` inlined, as providers require."""
    schema = model.model_json_schema(ref_template="#/$defs/{model}")
    defs = schema.pop("$defs", {})
    resolved = _resolve_refs(schema, defs, depth=0)
    if not isinstance(resolved, dict):  # pragma: no cover - a model schema is always an object
        raise InvalidRequest(f"unexpected schema shape for {model.__name__}")
    resolved.pop("title", None)
    return resolved


def _resolve_refs(node: Any, defs: dict[str, Any], *, depth: int) -> Any:
    if depth > 12:  # guard against a self referencing model
        return {"type": "object"}
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = defs.get(ref.removeprefix("#/$defs/"), {})
            merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
            return _resolve_refs(merged, defs, depth=depth + 1)
        return {key: _resolve_refs(value, defs, depth=depth + 1) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(item, defs, depth=depth + 1) for item in node]
    return node


class Tool(BaseModel):
    """A function the model may call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        """Reject a name no provider would accept, here rather than in five providers."""
        if not _TOOL_NAME_RE.match(value):
            raise InvalidRequest(
                f"tool name {value!r} is invalid: use 1 to 64 characters of letters, "
                "digits, underscore or hyphen"
            )
        return value

    @field_validator("parameters")
    @classmethod
    def _check_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Reject a schema that is not an object, which every provider requires."""
        if value.get("type", "object") != "object":
            raise InvalidRequest("tool parameters must be a JSON Schema object")
        return value

    @classmethod
    def from_function(cls, fn: Callable[..., Any]) -> Tool:
        """Build a tool from a python function's signature and docstring.

        Parameter types come from the annotations, descriptions from ``Args:``
        lines in a Google style docstring, and required-ness from the absence of a
        default.

        Args:
            fn: The function to describe. It is never called here.

        Returns:
            A tool whose ``parameters`` is a JSON Schema object.
        """
        signature = inspect.signature(fn)
        hints = typing.get_type_hints(fn)
        summary, arg_docs = _parse_docstring(inspect.getdoc(fn) or "")
        if not summary:
            raise InvalidRequest(
                f"{fn.__name__} needs a docstring: its summary is the description the "
                "model uses to decide whether to call the tool"
            )

        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, parameter in signature.parameters.items():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            annotation = hints.get(name, parameter.annotation)
            if annotation is inspect.Parameter.empty:
                raise InvalidRequest(
                    f"parameter {name!r} of {fn.__name__} has no type annotation, so its "
                    "schema would be a guess"
                )
            schema = _json_schema_for(annotation)
            if name in arg_docs:
                schema["description"] = arg_docs[name]
            properties[name] = schema
            if parameter.default is inspect.Parameter.empty:
                required.append(name)

        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        return cls(name=fn.__name__, description=summary, parameters=parameters)

    @classmethod
    def from_model(
        cls, model: type[BaseModel], *, name: str | None = None, description: str | None = None
    ) -> Tool:
        """Build a tool from a pydantic model.

        Args:
            model: The model describing the arguments. Its class name and docstring
                are used unless overridden.
            name: Tool name the model will call. Defaults to the class name.
            description: What the tool does. Defaults to the class docstring.

        Returns:
            A tool whose ``parameters`` is the model's inlined JSON Schema, with every
            ``$ref`` resolved because several providers reject them.
        """
        return cls(
            name=name or model.__name__,
            description=description or (inspect.getdoc(model) or "").strip(),
            parameters=_inline_model_schema(model),
        )


_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

_ARG_RE = re.compile(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")

_SECTION_HEADS = frozenset(
    {"returns:", "return:", "raises:", "yields:", "examples:", "example:", "notes:", "note:"}
)


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google style docstring into its first paragraph and its ``Args`` entries.

    Args:
        doc: A dedented docstring, or an empty string.

    Returns:
        The summary paragraph, and a mapping of parameter name to description. A
        description that continues onto the next line is joined onto one.
    """
    summary: list[str] = []
    arg_docs: dict[str, str] = {}
    section = "summary"
    current: str | None = None

    for raw in doc.splitlines():
        line = raw.strip()
        lowered = line.lower()
        if lowered in ("args:", "arguments:", "parameters:"):
            section, current = "args", None
        elif lowered in _SECTION_HEADS:
            section, current = "other", None
        elif section == "args":
            match = _ARG_RE.match(line)
            if match:
                current = match.group(1).lstrip("*")
                arg_docs[current] = match.group(2).strip()
            elif current and line:
                arg_docs[current] = f"{arg_docs[current]} {line}".strip()
        elif section == "summary":
            if not line and summary:
                section = "other"  # the summary is the first paragraph only
            elif line:
                summary.append(line)
    return " ".join(summary), arg_docs


class Usage(BaseModel):
    """Token counts for one call."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Input plus output. Cached and reasoning tokens are already counted in those."""
        return self.input_tokens + self.output_tokens


class Cost(BaseModel):
    """What the call cost, in US dollars.

    ``estimated`` is True whenever any input to the calculation was a guess: an
    unpriced model, or token counts the provider never reported.
    """

    model_config = ConfigDict(frozen=True)

    usd: float = 0.0
    estimated: bool = False


class Response(BaseModel):
    """The result of one completed chat call."""

    model_config = ConfigDict(extra="forbid")

    id: str
    model: str
    resolved_model: str
    message: Message
    stop_reason: StopReason
    usage: Usage
    cost: Cost
    latency_ms: int
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)
    audit_id: str = ""

    @property
    def text(self) -> str:
        """Concatenate every text part of the assistant message."""
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Tool calls the model requested, with arguments already parsed."""
        return self.message.tool_calls
