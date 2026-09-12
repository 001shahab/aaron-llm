# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Pre send redactors.

A redactor rewrites outbound text before it reaches a provider. The shipped ones
cover the patterns that come up most often in a GDPR review. They are deliberately
simple: a redactor is a safety net, not a substitute for not sending personal data
in the first place, and the README says so plainly.

Write your own by implementing :class:`Redactor`. It is a protocol, so any object
with a matching ``name`` and ``redact`` will do.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Redactor(Protocol):
    """Rewrites one piece of outbound text."""

    @property
    def name(self) -> str:
        """Label recorded in the audit record, never the pattern itself."""
        ...

    def redact(self, text: str) -> tuple[str, int]:
        """Return the rewritten text and how many replacements were made."""
        ...


@dataclass(frozen=True)
class RegexRedactor:
    """Replace every match of a regular expression.

    Args:
        pattern: The expression to match.
        replacement: What to put in its place.
        name: Label recorded in the audit record.
        flags: Regex flags.
    """

    pattern: str
    replacement: str = "[REDACTED]"
    name: str = "regex"
    flags: int = 0

    def redact(self, text: str) -> tuple[str, int]:
        """Apply the expression.

        Args:
            text: The outbound text.

        Returns:
            The rewritten text and the number of replacements made.
        """
        return re.subn(self.pattern, self.replacement, text, flags=self.flags)


@dataclass(frozen=True)
class EmailRedactor(RegexRedactor):
    """Replace email addresses."""

    pattern: str = r"\b[\w.%+-]+@[\w-]+(?:\.[\w-]+)+\b"
    replacement: str = "[EMAIL]"
    name: str = "email"
    flags: int = re.IGNORECASE


@dataclass(frozen=True)
class IpAddressRedactor(RegexRedactor):
    """Replace IPv4 and common IPv6 literals, which are personal data under the GDPR."""

    pattern: str = r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b"
    replacement: str = "[IP]"
    name: str = "ip"


_BY_NAME: dict[str, Callable[[], Redactor]] = {
    "email": EmailRedactor,
    "ip": IpAddressRedactor,
}


def build_redactor(spec: str | dict[str, object]) -> Redactor:
    """Build a redactor from a policy file entry.

    A string names a shipped redactor. A mapping with a ``pattern`` key builds a
    :class:`RegexRedactor`, so a policy file can carry a bespoke expression without
    any Python.

    Args:
        spec: ``"email"``, or ``{"pattern": r"\\b\\d{11}\\b", "replacement": "[ID]"}``.

    Returns:
        The redactor.

    Raises:
        ValueError: The name is not a shipped redactor, or the mapping has no
            pattern.
    """
    if isinstance(spec, str):
        builtin = _BY_NAME.get(spec.strip().lower())
        if builtin is None:
            raise ValueError(
                f"unknown redactor {spec!r}. Available: {', '.join(sorted(_BY_NAME))}, "
                "or give a mapping with a 'pattern' key."
            )
        return builtin()

    pattern = spec.get("pattern")
    if not isinstance(pattern, str):
        raise ValueError("a redactor mapping needs a string 'pattern' key")
    replacement = spec.get("replacement", "[REDACTED]")
    name = spec.get("name", "regex")
    return RegexRedactor(
        pattern=pattern,
        replacement=str(replacement),
        name=str(name),
        flags=re.IGNORECASE if spec.get("ignore_case") else 0,
    )
