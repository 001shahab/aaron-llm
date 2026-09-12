# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""The Aaron exception hierarchy.

Every error raised by this library derives from :class:`AaronError` and carries
the provider, model, HTTP status, provider request id and raw error payload where
those are known. Provider errors are never collapsed into a generic exception.
"""

from __future__ import annotations

from typing import Any


class AaronError(Exception):
    """Base class for everything this library raises.

    Args:
        message: Human readable description. Must never contain credentials.
        provider: Provider name, for example ``"openai"``.
        model: Canonical ``provider/model`` string the caller asked for.
        status_code: HTTP status code when the error came from a response.
        request_id: Provider side request id, useful in a support ticket.
        raw: Untouched provider error payload.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.request_id = request_id
        self.raw = raw

    def __str__(self) -> str:
        context = {
            "provider": self.provider,
            "model": self.model,
            "status": self.status_code,
            "request_id": self.request_id,
        }
        detail = " ".join(f"{k}={v}" for k, v in context.items() if v is not None)
        return f"{self.message} ({detail})" if detail else self.message

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"


class ConfigurationError(AaronError):
    """The client or the call was configured in a way that cannot work."""


class MissingAPIKey(ConfigurationError):
    """No credential was found for the provider the call needs."""


class UnknownProvider(ConfigurationError):
    """The model string named a provider that is not registered."""


class UnknownModel(ConfigurationError):
    """A model was rejected as unknown.

    Note that an unknown model is normally permitted, with permissive
    capabilities and estimated cost. This is raised only when a caller has asked
    for strict registry behaviour.
    """


class PolicyViolation(AaronError):
    """A call was blocked by the policy layer before any network access.

    Args:
        message: Summary of the violation.
        model: The model that was rejected.
        rule: Machine readable rule name, for example ``"residency"``.
        detail: Why the rule rejected this model.
        candidates: For each model tried, the ``(model, rule, detail)`` rejection.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        rule: str | None = None,
        detail: str | None = None,
        candidates: list[tuple[str, str, str]] | None = None,
    ) -> None:
        super().__init__(message, model=model)
        self.rule = rule
        self.detail = detail
        self.candidates: list[tuple[str, str, str]] = candidates or []

    def __str__(self) -> str:
        bits = [self.message]
        if self.rule:
            bits.append(f"(model={self.model} rule={self.rule} detail={self.detail})")
        if self.candidates:
            listed = "; ".join(f"{m}: {r}: {d}" for m, r, d in self.candidates)
            bits.append(f"rejected candidates: {listed}")
        return " ".join(bits)


class RequestError(AaronError):
    """The request itself was wrong, retrying it unchanged will not help."""


class InvalidRequest(RequestError):
    """The provider rejected the request body as malformed or unsupported."""


class ContextLengthExceeded(RequestError):
    """The prompt, or the prompt plus the requested output, is too long."""


class ToolArgumentError(RequestError):
    """A model emitted tool call arguments that are not valid JSON objects.

    Args:
        message: Summary of the parse failure.
        raw_arguments: The exact string the model produced, kept for debugging.
        tool_name: Name of the tool that was called.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_arguments: str,
        tool_name: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model)
        self.raw_arguments = raw_arguments
        self.tool_name = tool_name


class ProviderError(AaronError):
    """The provider returned an error response."""


class AuthenticationError(ProviderError):
    """The credential was missing, malformed or rejected."""


# Deliberately shadows the builtin inside this module; callers see aaron.errors.PermissionError.
class PermissionError(ProviderError):
    """The credential is valid but not allowed to do this."""


class RateLimitError(ProviderError):
    """A rate or quota limit was hit.

    Args:
        retry_after: Seconds to wait, when the provider told us.
    """

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ServerError(ProviderError):
    """The provider failed internally."""


class ServiceOverloaded(ProviderError):
    """The provider is up but shedding load."""


class ContentFilterError(ProviderError):
    """The provider refused on safety grounds."""


class TransportError(AaronError):
    """The call never produced a usable HTTP response."""


# Deliberately shadows the builtin inside this module.
class TimeoutError(TransportError):
    """The call exceeded its timeout."""


# Deliberately shadows the builtin inside this module.
class ConnectionError(TransportError):
    """The connection could not be established or was dropped."""


class LocalProviderUnavailable(ConnectionError):
    """A local provider such as Ollama is not reachable on its configured host."""
