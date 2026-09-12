# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""The policy object and its evaluation.

Evaluation happens entirely before any network call, in a fixed order, so that a
rejection is cheap and its reason is unambiguous:

1. deny list, then allow list, because a deny always wins
2. residency
3. capabilities
4. input token budget
5. cost budget
6. redaction

A rejection is a :class:`~aaron.errors.PolicyViolation` carrying structured
``model``, ``rule`` and ``detail`` fields, never just a sentence, so a caller can
branch on the reason and an auditor can count them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from ..errors import ConfigurationError, InvalidRequest, PolicyViolation
from ..registry import CAPABILITY_NAMES, Registry, glob_match
from ..request import ChatRequest, estimate_tokens, split_model
from ..types import ContentPart, Message, TextPart
from . import residency as residency_module
from .redaction import Redactor, build_redactor

OnViolation = Literal["raise", "fallback"]


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of evaluating one candidate model.

    Attributes:
        allowed: Whether this candidate may be used.
        rule: The rule that rejected it, empty when allowed.
        detail: Why, in words a compliance reader can act on.
    """

    allowed: bool
    rule: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """What the policy decided for a call.

    Attributes:
        request: The request to send, with redactions already applied.
        redactions: How many replacements the redactors made, for the audit record.
        rejected: Every candidate that was tried and refused, in order.
    """

    request: ChatRequest
    redactions: int
    rejected: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Policy:
    """Rules a call must satisfy before it is allowed to leave the process.

    Attributes:
        allow: Glob patterns of permitted models. Empty means allow everything.
        deny: Glob patterns of forbidden models. A deny always beats an allow.
        residency: Required processing jurisdiction, for example ``"eu"``. An
            unknown region is rejected rather than assumed compliant.
        max_usd_per_call: Ceiling on the worst case cost of one call.
        max_input_tokens: Ceiling on the estimated input size.
        require_capabilities: Capability names the model must have.
        fallback: Models to try, in order, when ``on_violation`` is ``"fallback"``.
        redactors: Applied to every outbound text part, in order.
        on_violation: ``"raise"`` to fail immediately, ``"fallback"`` to walk the
            fallback list and re evaluate every rule for each candidate.
    """

    allow: Sequence[str] = ()
    deny: Sequence[str] = ()
    residency: str | None = None
    max_usd_per_call: float | None = None
    max_input_tokens: int | None = None
    require_capabilities: Sequence[str] = ()
    fallback: Sequence[str] = ()
    redactors: Sequence[Redactor] = ()
    on_violation: OnViolation = "raise"

    def __post_init__(self) -> None:
        if self.on_violation not in ("raise", "fallback"):
            raise ConfigurationError(
                f"on_violation must be 'raise' or 'fallback', got {self.on_violation!r}"
            )
        unknown = sorted(set(self.require_capabilities) - CAPABILITY_NAMES)
        if unknown:
            raise ConfigurationError(
                f"unknown capability names in require_capabilities: {', '.join(unknown)}. "
                f"Known names: {', '.join(sorted(CAPABILITY_NAMES))}"
            )
        if self.on_violation == "fallback" and not self.fallback:
            raise ConfigurationError(
                "on_violation='fallback' needs a non empty fallback list, "
                "otherwise there is nothing to fall back to"
            )

    # Construction ----------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> Policy:
        """Load a policy from a YAML file so it can live in version control.

        Args:
            path: Path to the policy file.

        Returns:
            The policy.

        Raises:
            ConfigurationError: The file is missing, is not a mapping, or contains
                a key that is not a policy field.
        """
        from .._yaml import load_mapping

        file = Path(path)
        if not file.is_file():
            raise ConfigurationError(f"policy file not found: {file}")
        return cls.from_mapping(load_mapping(file.read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Policy:
        """Build a policy from a plain mapping, as loaded from a file.

        Args:
            data: Field names to values. ``redactors`` accepts shipped redactor
                names or mappings with a ``pattern`` key.

        Returns:
            The policy.

        Raises:
            ConfigurationError: A key is not a policy field, or a redactor spec is
                invalid.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(
                f"unknown policy fields: {sorted(unknown)}. Valid fields: {sorted(known)}"
            )
        fields: dict[str, Any] = dict(data)
        raw_redactors = fields.get("redactors")
        if raw_redactors:
            # build_redactor raises ConfigurationError itself, naming the bad entry.
            fields["redactors"] = tuple(build_redactor(spec) for spec in raw_redactors)
        for key in ("allow", "deny", "require_capabilities", "fallback"):
            value = fields.get(key)
            if isinstance(value, str):
                fields[key] = (value,)
            elif value is not None:
                fields[key] = tuple(value)
        return cls(**fields)

    # Evaluation ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """The rules in force, as plain data for the audit record.

        Returns:
            A JSON serialisable mapping. Redactors appear by name only, never by
            pattern, because a pattern can itself describe personal data.
        """
        return {
            "allow": list(self.allow),
            "deny": list(self.deny),
            "residency": self.residency,
            "max_usd_per_call": self.max_usd_per_call,
            "max_input_tokens": self.max_input_tokens,
            "require_capabilities": list(self.require_capabilities),
            "fallback": list(self.fallback),
            "redactors": [getattr(r, "name", type(r).__name__) for r in self.redactors],
            "on_violation": self.on_violation,
        }

    def evaluate(self, model: str, req: ChatRequest, *, registry: Registry) -> Decision:
        """Check one candidate model against every rule.

        Args:
            model: The canonical ``provider/model`` string to check.
            req: The request, used for the token and cost estimates.
            registry: Metadata source for region, capabilities and price.

        Returns:
            A :class:`Decision`. The first failing rule is the one reported.
        """
        for pattern in self.deny:
            if _glob(model, pattern):
                return Decision(
                    allowed=False,
                    rule="deny",
                    detail=f"{model} matches deny pattern {pattern!r}",
                )

        if self.allow and not any(_glob(model, pattern) for pattern in self.allow):
            return Decision(
                allowed=False,
                rule="allow",
                detail=f"{model} matches none of the allow patterns {list(self.allow)}",
            )

        if self.residency is not None:
            region = registry.region(model)
            if not residency_module.satisfies(self.residency, region):
                return Decision(
                    allowed=False,
                    rule="residency",
                    detail=residency_module.describe(self.residency, region, model),
                )

        if self.require_capabilities:
            capabilities = registry.capabilities(model)
            missing = [
                name for name in self.require_capabilities if not capabilities.supports(name)
            ]
            if missing:
                return Decision(
                    allowed=False,
                    rule="capabilities",
                    detail=(
                        f"{model} does not support {missing} according to the registry. "
                        f"It supports: {capabilities.names()}."
                    ),
                )

        input_tokens = estimate_tokens(req.messages)
        if self.max_input_tokens is not None and input_tokens > self.max_input_tokens:
            return Decision(
                allowed=False,
                rule="max_input_tokens",
                detail=(
                    f"estimated {input_tokens} input tokens exceeds the limit of "
                    f"{self.max_input_tokens}"
                ),
            )

        if self.max_usd_per_call is not None:
            worst_case = registry.estimate_cost(model, input_tokens, req.max_tokens)
            if worst_case > self.max_usd_per_call:
                return Decision(
                    allowed=False,
                    rule="max_usd_per_call",
                    detail=(
                        f"worst case cost ${worst_case:.4f} for {model} exceeds the limit of "
                        f"${self.max_usd_per_call:.4f}. Lower max_tokens or choose a "
                        f"cheaper model."
                    ),
                )

        return Decision(allowed=True)

    def apply(self, req: ChatRequest, *, registry: Registry) -> PolicyResult:
        """Evaluate the request, walking fallbacks if allowed, then redact.

        Args:
            req: The resolved request.
            registry: Metadata source.

        Returns:
            The request to send, the redaction count, and any rejected candidates.

        Raises:
            PolicyViolation: The requested model was refused and either
                ``on_violation`` is ``"raise"`` or every fallback was refused too.
                The exception lists each candidate with its rule and reason.
        """
        rejected: list[tuple[str, str, str]] = []
        decision = self.evaluate(req.model, req, registry=registry)
        chosen = req

        if not decision.allowed:
            rejected.append((req.model, decision.rule, decision.detail))
            if self.on_violation == "raise":
                raise PolicyViolation(
                    f"policy refused {req.model}: {decision.detail}",
                    model=req.model,
                    rule=decision.rule,
                    detail=decision.detail,
                    candidates=rejected,
                )
            chosen = self._walk_fallbacks(req, registry=registry, rejected=rejected)

        redacted, count = self.redact(chosen.messages)
        return PolicyResult(
            request=chosen.with_messages(redacted) if count else chosen,
            redactions=count,
            rejected=rejected,
        )

    def _walk_fallbacks(
        self, req: ChatRequest, *, registry: Registry, rejected: list[tuple[str, str, str]]
    ) -> ChatRequest:
        """Try each fallback in order, re evaluating every rule for each."""
        for candidate in self.fallback:
            try:
                provider, model_name = split_model(candidate)
            except InvalidRequest as exc:  # a malformed fallback is config, not a violation
                raise ConfigurationError(f"invalid fallback model {candidate!r}: {exc}") from exc
            attempt = replace(req, model=candidate, provider=provider, model_name=model_name)
            decision = self.evaluate(candidate, attempt, registry=registry)
            if decision.allowed:
                return attempt
            rejected.append((candidate, decision.rule, decision.detail))

        first = rejected[0]
        raise PolicyViolation(
            f"policy refused {first[0]} and every fallback",
            model=first[0],
            rule=first[1],
            detail=first[2],
            candidates=rejected,
        )

    def redact(self, messages: Sequence[Message]) -> tuple[list[Message], int]:
        """Run every redactor over every text part, in order.

        Args:
            messages: The conversation to rewrite.

        Returns:
            The rewritten messages and the total number of replacements. When there
            are no redactors the messages are returned unchanged.
        """
        if not self.redactors:
            return list(messages), 0

        total = 0
        out: list[Message] = []
        for message in messages:
            parts: list[ContentPart] = []
            changed = False
            for part in message.content:
                if not isinstance(part, TextPart):
                    parts.append(part)
                    continue
                text = part.text
                for redactor in self.redactors:
                    text, count = redactor.redact(text)
                    total += count
                if text != part.text:
                    changed = True
                    parts.append(TextPart(text=text))
                else:
                    parts.append(part)
            out.append(message.model_copy(update={"content": parts}) if changed else message)
        return out, total


def _glob(value: str, pattern: str) -> bool:
    """Match a model string against a pattern where only ``*`` is special.

    Model ids contain dots, colons and dashes that must stay literal, so the
    registry's matcher is used rather than :func:`fnmatch.fnmatch`, which would also
    treat ``?`` and ``[]`` as wildcards.
    """
    return glob_match(value, pattern)
