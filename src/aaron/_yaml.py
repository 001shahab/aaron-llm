# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""A deliberately small YAML reader, used when PyYAML is not installed.

The shipped ``models.yaml`` must load with only the two runtime dependencies, so
this understands the subset that it and the policy files are written in:

- a mapping of keys at the top level
- one level of indented ``field: value`` under a key
- inline collections, ``[a, b]`` and ``{a: 1}``
- block sequences of scalars, ``- item``
- scalars: quoted and bare strings, numbers, booleans, null
- comments, and the ``---`` document marker

That is enough for every file Aaron ships or documents. When PyYAML is importable it
is used instead, so a user's own file may use the full language. Anything outside the
subset raises :class:`YamlSubsetError`, which names the line and points at the
``aaron-llm[yaml]`` extra.
"""

from __future__ import annotations

import json
from typing import Any


class YamlSubsetError(ValueError):
    """The document used YAML this reader does not implement."""

    def __init__(self, message: str, line_number: int) -> None:
        super().__init__(
            f"line {line_number}: {message}. "
            "Install the optional extra 'aaron-llm[yaml]' to parse the full YAML language."
        )
        self.line_number = line_number


def load_mapping(text: str) -> dict[str, Any]:
    """Parse a YAML mapping document into a dict.

    Args:
        text: The document source.

    Returns:
        The top level mapping, empty when the document is blank.

    Raises:
        YamlSubsetError: The document is not a mapping, or uses unsupported syntax.
    """
    try:
        # Optional dependency, imported inside the function so that importing aaron
        # never requires it.
        import yaml
    except ImportError:
        return _parse(_significant_lines(text))
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise YamlSubsetError(f"expected a mapping, got {type(loaded).__name__}", 1)
    return {str(key): value for key, value in loaded.items()}


def _significant_lines(text: str) -> list[tuple[int, int, str]]:
    """Return ``(line_number, indent, content)`` for every line that carries data."""
    out: list[tuple[int, int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = _strip_comment(raw)
        content = stripped.strip()
        if not content or content in ("---", "..."):
            continue
        if "\t" in stripped[: len(stripped) - len(stripped.lstrip())]:
            raise YamlSubsetError("tab indentation is not supported", number)
        out.append((number, len(stripped) - len(stripped.lstrip()), content))
    return out


def _parse(lines: list[tuple[int, int, str]]) -> dict[str, Any]:
    """Read the top level mapping, handing each key's indented block to :func:`_block`."""
    result: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        number, indent, content = lines[index]
        index += 1
        key, value, has_value = _split(content, number)
        if has_value:
            result[key] = _scalar(value, number)
            continue
        # Everything more deeply indented belongs to this key, as does a block
        # sequence at the same indentation, which is how YAML allows it to be written.
        start = index
        while index < len(lines) and (
            lines[index][1] > indent or (lines[index][1] == indent and lines[index][2][:2] == "- ")
        ):
            index += 1
        result[key] = _block(lines[start:index])
    return result


def _block(block: list[tuple[int, int, str]]) -> Any:
    """Turn one key's block into a sequence, a flat mapping, or None when empty."""
    if not block:
        return None
    if all(content.startswith("- ") for _, _, content in block):
        return [_scalar(content[2:].strip(), number) for number, _, content in block]

    mapping: dict[str, Any] = {}
    for number, _, content in block:
        key, value, has_value = _split(content, number)
        if not has_value:
            raise YamlSubsetError("nesting deeper than one level is not supported", number)
        mapping[key] = _scalar(value, number)
    return mapping


def _split(content: str, number: int) -> tuple[str, str, bool]:
    """Split ``key: value`` on the first colon outside quotes and brackets."""
    quote: str | None = None
    depth = 0
    for index, char in enumerate(content):
        if quote is not None:
            quote = None if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == ":" and depth == 0:
            after = content[index + 1 :]
            if after and not after.startswith(" "):
                continue  # part of a model id such as llama3.1:8b
            return _unquote(content[:index].strip(), number), after.strip(), bool(after.strip())
    raise YamlSubsetError("expected 'key: value'", number)


def _strip_comment(raw: str) -> str:
    """Drop a trailing comment, respecting quotes."""
    quote: str | None = None
    for index, char in enumerate(raw):
        if quote is not None:
            quote = None if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char == "#" and (index == 0 or raw[index - 1] in " \t"):
            return raw[:index]
    return raw


_CONSTANTS: dict[str, Any] = {"true": True, "false": False, "null": None, "~": None, "": None}


def _scalar(text: str, number: int) -> Any:
    """Convert one scalar or inline collection."""
    if text.startswith(("[", "{")):
        return _flow(text, number)
    if text.startswith(("|", ">", "&", "*", "!")):
        raise YamlSubsetError("block scalars, anchors and tags are not supported", number)
    if text[:1] in "\"'":
        return _unquote(text, number)
    if text.lower() in _CONSTANTS:
        return _CONSTANTS[text.lower()]
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            continue
    return text


def _flow(text: str, number: int) -> Any:
    """Parse an inline collection, falling back to splitting bare words on commas."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if not (text.startswith("[") and text.endswith("]")):
        raise YamlSubsetError("unsupported inline collection", number)
    inner = text[1:-1].strip()
    return [_scalar(item.strip(), number) for item in inner.split(",") if item.strip()]


def _unquote(text: str, number: int) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        if text[0] == "'":
            return text[1:-1].replace("''", "'")
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise YamlSubsetError("malformed double quoted string", number) from exc
        return str(decoded)
    return text
