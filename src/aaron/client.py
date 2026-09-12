# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""``Aaron`` and ``AsyncAaron``, the two entry points.

Everything a call needs is resolved here, in this order, highest first:

1. an explicit argument to ``chat`` or ``stream``
2. an explicit argument to the constructor
3. ``aaron.toml`` in the working directory
4. environment variables
5. library defaults

The two classes share configuration, request building, policy evaluation, retry
bookkeeping and audit records. Only the transport loops are written twice, because a
shared async core would drag every synchronous traceback through an event loop the
caller never asked for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import tomllib
from collections.abc import AsyncGenerator, Callable, Generator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError

from .audit import AuditLog, AuditRecord, JsonlSink, Outcome, Sink
from .errors import (
    AaronError,
    ConfigurationError,
    InvalidRequest,
    MissingAPIKey,
    PolicyViolation,
    ProviderError,
    RateLimitError,
    TransportError,
)
from .errors import TimeoutError as AaronTimeout
from .policy import Policy
from .policy.policy import PolicyResult
from .providers import PreparedRequest, Provider, get_provider
from .registry import Registry, default_registry
from .request import (
    ChatRequest,
    JsonSchemaSpec,
    Prompt,
    ToolChoice,
    body_sha256,
    normalise_messages,
    split_model,
    validate,
)
from .retry import RetryPolicy
from .stream import AsyncStream, DoneEvent, Stream, StreamEvent
from .transport import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_TIMEOUT,
    AsyncTransport,
    Transport,
    TransportConfig,
    translate_httpx_error,
)
from .types import DocumentPart, ImagePart, Message, Response, SecretValue, TextPart, Tool

log = logging.getLogger("aaron")

M = TypeVar("M", bound=BaseModel)

KeySource = str | Callable[[], str]

CONFIG_FILE = "aaron.toml"

# How many times extract() may show a model its validation errors and ask again.
DEFAULT_MAX_REPAIRS = 2


class _Call:
    """One call in flight: its request, its timing, and its single audit record.

    Both clients drive this, which is what keeps the retry accounting and the audit
    trail identical in the sync and async paths.
    """

    def __init__(
        self,
        client: _ClientBase,
        req: ChatRequest,
        prepared: PreparedRequest,
        provider: Provider,
        redactions: int,
    ) -> None:
        self.client = client
        self.req = req
        self.prepared = prepared
        self.provider = provider
        self.redactions = redactions
        self.prompt_sha256 = body_sha256(prepared.body)
        self.started = time.monotonic()

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)

    def record(self, **kwargs: Any) -> AuditRecord:
        """Build a record for this call, filling in what the call already knows."""
        return self.client.build_record(
            self.req,
            redactions=self.redactions,
            prompt_sha256=self.prompt_sha256,
            **kwargs,
        )

    def fail(self, error: BaseException, attempts: int) -> None:
        """Write the record for a failed call. The caller re raises."""
        self.client.audit.write(
            self.record(
                outcome=_outcome_for(error),
                error=error,
                attempts=attempts,
                latency_ms=self.elapsed_ms,
            )
        )

    def succeed(self, response: Response, attempts: int) -> Response:
        """Write the record for a successful call and stamp the response with it."""
        latency_ms = self.elapsed_ms
        record = self.record(
            outcome="ok", response=response, attempts=attempts, latency_ms=latency_ms
        )
        self.client.audit.write(record)
        return response.model_copy(update={"latency_ms": latency_ms, "audit_id": record.id})

    def send_kwargs(self) -> dict[str, Any]:
        """The keyword arguments both transports pass to httpx, timeouts included."""
        config = self.client.transport_config.with_overrides(
            timeout=self.req.timeout, connect_timeout=self.req.connect_timeout
        )
        return {
            "json": self.prepared.body,
            "headers": self.prepared.sendable_headers(),
            "timeout": config.httpx_timeout(),
        }


class _StreamRun:
    """Audit bookkeeping for one streaming call.

    The record is created up front so that its id can be stamped onto the response
    inside the done event, and completed in a ``finally`` block so that a caller who
    abandons the stream still leaves exactly one record behind, marked cancelled.
    """

    def __init__(self, call: _Call) -> None:
        self.call = call
        self.record = call.record(outcome="ok")
        self.outcome = "cancelled"
        self.error: BaseException | None = None
        self.response: Response | None = None

    def event(self, event: StreamEvent) -> StreamEvent:
        """Pass an event through, capturing the assembled response from the end."""
        if isinstance(event, DoneEvent):
            self.response = event.response.model_copy(update={"audit_id": self.record.id})
            self.outcome = "ok"
            return DoneEvent(response=self.response)
        return event

    def failed(self, error: BaseException) -> None:
        """Note that the stream died, for the record written on the way out."""
        self.error = error
        self.outcome = _outcome_for(error)

    def finish(self) -> None:
        """Complete and write the record. Always called, exactly once."""
        record, response = self.record, self.response
        record.outcome = _as_outcome(self.outcome)
        record.latency_ms = self.call.elapsed_ms
        record.error_type = type(self.error).__name__ if self.error else None
        if response is not None:
            record.model_resolved = response.resolved_model
            record.usage = response.usage
            record.cost = response.cost
            record.output_chars = len(response.text)
            record.response_sha256 = body_sha256(response.raw) if response.raw else None
            if record.content is not None:
                record.content["output"] = response.text
        self.call.client.audit.write(record)


class _ClientBase:
    """Configuration, request building, policy and audit, shared by both clients."""

    def __init__(
        self,
        *,
        api_keys: Mapping[str, KeySource] | None = None,
        base_urls: Mapping[str, str] | None = None,
        policy: Policy | None = None,
        audit: Sink | AuditLog | str | Path | None = None,
        model_registry: str | Path | Mapping[str, Any] | None = None,
        default_model: str | None = None,
        aliases: Mapping[str, str] | None = None,
        timeout: float | None = None,
        connect_timeout: float | None = None,
        retry: RetryPolicy | None = None,
        headers: Mapping[str, str] | None = None,
        proxy: str | None = None,
        verify: bool = True,
        trust_env: bool = True,
        config_file: str | Path | None = None,
        record_content: bool = False,
    ) -> None:
        self._file_config = _load_config_file(config_file)
        self._api_keys: dict[str, KeySource] = dict(api_keys or {})
        self._base_urls: dict[str, str] = {
            **_mapping(self._file_config.get("base_urls")),
            **(base_urls or {}),
        }
        self._aliases: dict[str, str] = {
            **_mapping(self._file_config.get("aliases")),
            **(aliases or {}),
        }
        self.registry: Registry = default_registry().merge(
            model_registry
            if model_registry is not None
            else self._file_config.get("model_registry")
        )
        self.policy: Policy | None = policy or self._policy_from_config()
        self.audit: AuditLog = self._audit_from(audit, record_content=record_content)
        self.retry: RetryPolicy = retry or RetryPolicy()
        self.default_model: str | None = (
            default_model
            or _str_or_none(self._file_config.get("default_model"))
            or os.environ.get("AARON_DEFAULT_MODEL")
        )
        self.transport_config = TransportConfig(
            timeout=_first_float(
                timeout, self._file_config.get("timeout"), os.environ.get("AARON_TIMEOUT")
            )
            or DEFAULT_TIMEOUT,
            connect_timeout=_first_float(connect_timeout, self._file_config.get("connect_timeout"))
            or DEFAULT_CONNECT_TIMEOUT,
            proxy=proxy,
            verify=verify,
            headers=dict(headers or {}),
            trust_env=trust_env,
        )

    # Configuration ---------------------------------------------------------------

    def _policy_from_config(self) -> Policy | None:
        """Load a policy from the environment, ``aaron.toml``, or the default file."""
        explicit = os.environ.get("AARON_POLICY_FILE") or _str_or_none(
            self._file_config.get("policy_file")
        )
        if explicit:
            return Policy.from_file(explicit)
        inline = self._file_config.get("policy")
        if isinstance(inline, dict):
            return Policy.from_mapping(inline)
        default_path = Path("aaron.policy.yaml")
        return Policy.from_file(default_path) if default_path.is_file() else None

    def _audit_from(
        self, audit: Sink | AuditLog | str | Path | None, *, record_content: bool
    ) -> AuditLog:
        """Turn whatever the caller passed into an :class:`AuditLog`."""
        if isinstance(audit, AuditLog):
            return audit
        if audit is not None and not isinstance(audit, str | Path):
            return AuditLog(audit)
        path = audit or os.environ.get("AARON_AUDIT_PATH") or self._file_config.get("audit_path")
        if not path:
            return AuditLog()
        return AuditLog(JsonlSink(str(path), record_content=record_content))

    def alias(self, name: str, target: str) -> None:
        """Register a short name for a model.

        Args:
            name: The alias, for example ``"fast"``. It must not contain a slash, so
                that an alias can never shadow a ``provider/model`` string.
            target: A canonical model string, or another alias.

        Raises:
            ConfigurationError: The alias is empty or contains a slash.
        """
        if not name or "/" in name:
            raise ConfigurationError(
                f"alias {name!r} must be a non empty name without a slash, "
                "so that it cannot shadow a provider/model string"
            )
        self._aliases[name] = target

    def resolve_model(self, model: str) -> str:
        """Follow aliases until a concrete ``provider/model`` string is reached.

        Args:
            model: An alias or a canonical model string.

        Returns:
            The canonical model string.

        Raises:
            ConfigurationError: The aliases form a cycle.
            InvalidRequest: The result is not a ``provider/model`` string.
        """
        seen: list[str] = []
        current = model
        while current in self._aliases:
            if current in seen:
                raise ConfigurationError(f"alias cycle: {' -> '.join([*seen, current])}")
            seen.append(current)
            current = self._aliases[current]
        split_model(current)  # raises InvalidRequest when it is not provider/model
        return current

    def _key_for(self, provider: Provider) -> SecretValue | None:
        """Resolve a credential at call time. Keys are never cached in a global.

        Raises:
            MissingAPIKey: The provider needs a key and none was found.
        """
        source = self._api_keys.get(provider.name)
        if source is not None:
            value = source() if callable(source) else source
            if not value:
                raise MissingAPIKey(
                    f"the api_keys entry for {provider.name!r} returned an empty value",
                    provider=provider.name,
                )
            return SecretValue(str(value))
        if provider.env_key:
            from_env = os.environ.get(provider.env_key)
            if from_env:
                return SecretValue(from_env)
            if provider.requires_key:
                raise MissingAPIKey(
                    f"no credential for {provider.name}. Set {provider.env_key}, or pass "
                    f"Aaron(api_keys={{'{provider.name}': ...}}) with a string or a callable.",
                    provider=provider.name,
                )
        return None

    def _base_url_for(self, provider: Provider, override: str | None) -> str:
        """Decide the endpoint for this call."""
        if override:
            return _normalise_url(override)
        configured = self._base_urls.get(provider.name)
        if configured:
            return _normalise_url(configured)
        if provider.name == "ollama":
            host = os.environ.get("OLLAMA_HOST")
            if host:
                return _normalise_url(host)
        if not provider.default_base_url:
            raise ConfigurationError(
                f"provider {provider.name!r} has no default endpoint, pass base_url=... "
                f"or Aaron(base_urls={{'{provider.name}': 'https://...'}})",
                provider=provider.name,
            )
        return provider.default_base_url

    # Request building ------------------------------------------------------------

    def build(
        self,
        model: str | None,
        messages: Prompt | None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        json_schema: JsonSchemaSpec | None = None,
        json_mode: bool = False,
        provider_options: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        connect_timeout: float | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> ChatRequest:
        """Resolve everything a provider needs, without sending anything.

        Args:
            model: Alias or ``provider/model``. Falls back to the client default.
            messages: A string, a message, or a sequence of messages.
            max_tokens: Output ceiling.
            temperature: Sampling temperature, 0 to 2.
            top_p: Nucleus sampling cutoff.
            stop: Stop sequences.
            tools: Tools the model may call.
            tool_choice: ``"auto"``, ``"none"``, ``"required"``, or a tool name.
            json_schema: Schema the reply must satisfy.
            json_mode: Ask for any valid JSON object, without a schema.
            provider_options: Merged verbatim into the outgoing body.
            headers: Extra headers for this call.
            base_url: Endpoint override for this call.
            timeout: Total timeout for this call.
            connect_timeout: Connect timeout for this call.
            tags: Audit tags for this call, overriding the client's.

        Returns:
            The resolved request.

        The credential is deliberately not resolved here. It is attached after policy
        has chosen the final model, so that a call the policy refuses, or falls back
        away from, never demands a key for the provider it rejected.

        Raises:
            ConfigurationError: No model was given and no default is configured.
            InvalidRequest: The model string or one of the fields is invalid.
        """
        target = model or self.default_model
        if not target:
            raise ConfigurationError(
                "no model given and no default configured. Pass a model, or set "
                "Aaron(default_model=...), AARON_DEFAULT_MODEL, or default_model in aaron.toml."
            )
        canonical = self.resolve_model(target)
        provider_name, model_name = split_model(canonical)
        provider = get_provider(provider_name)

        req = ChatRequest(
            model=canonical,
            provider=provider_name,
            model_name=model_name,
            messages=normalise_messages(messages if messages is not None else ""),
            base_url=self._base_url_for(provider, base_url),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=list(stop or []),
            tools=list(tools or []),
            tool_choice=tool_choice,
            json_schema=json_schema,
            json_mode=json_mode,
            provider_options=dict(provider_options or {}),
            extra_headers=dict(headers or {}),
            timeout=timeout,
            connect_timeout=connect_timeout,
            tags=self.audit.merge_tags(dict(tags) if tags else None),
            registry=self.registry,
        )
        validate(req)
        return req

    def dry_run(
        self, model: str | None = None, messages: Prompt | None = None, **kwargs: Any
    ) -> PreparedRequest:
        """Build the exact HTTP request that would be sent, and send nothing.

        Credentials are masked, so the result is safe to print, to diff in a review,
        or to attach to a compliance report. Policy is evaluated and redactors are
        applied first, so what you see is what would leave the process.

        Args:
            model: Alias or ``provider/model``.
            messages: The prompt.
            **kwargs: Anything :meth:`build` accepts, plus ``stream``.

        Returns:
            The prepared request, with masked auth headers.

        Raises:
            PolicyViolation: The call would be refused.
            MissingAPIKey: The provider needs a credential and none was found.
        """
        stream = bool(kwargs.pop("stream", False))
        req = self.build(model, messages, **kwargs)
        result = self._apply_policy(req)
        provider = get_provider(result.request.provider)
        bound = self._bind(result.request, provider, requested=req.provider)
        return provider.build_request(bound, stream=stream).masked()

    # Policy and audit ------------------------------------------------------------

    def _apply_policy(self, req: ChatRequest) -> PolicyResult:
        """Evaluate policy, or pass the request through when there is none."""
        if self.policy is None:
            return PolicyResult(request=req, redactions=0)
        return self.policy.apply(req, registry=self.registry)

    def begin(
        self, model: str | None, messages: Prompt | None, kwargs: dict[str, Any], *, stream: bool
    ) -> _Call:
        """Resolve, evaluate policy, and build the HTTP request for one call.

        Returns:
            A tracker holding the request, the prepared HTTP request and the timing.

        Raises:
            PolicyViolation: The call was refused. Its audit record is written first.
            MissingAPIKey: The chosen provider needs a credential and none was found.
        """
        req = self.build(model, messages, **kwargs)
        try:
            result = self._apply_policy(req)
        except PolicyViolation as violation:
            self.audit.write(
                self.build_record(
                    req,
                    outcome="policy_violation",
                    error=violation,
                    detail={"rule": violation.rule, "candidates": violation.candidates},
                )
            )
            raise
        provider = get_provider(result.request.provider)
        chosen = self._bind(result.request, provider, requested=req.provider)
        prepared = provider.build_request(chosen, stream=stream)
        return _Call(self, chosen, prepared, provider, result.redactions)

    def _bind(self, req: ChatRequest, provider: Provider, *, requested: str) -> ChatRequest:
        """Attach the credential now that policy has settled which provider is used.

        When policy fell back to a different provider, the endpoint is resolved again
        from scratch: neither the original provider's default nor a per call
        ``base_url`` meant for it can apply to a provider the caller did not name.

        Args:
            req: The request policy chose.
            provider: The provider that will serve it.
            requested: The provider name the caller originally asked for.

        Returns:
            The request with its credential and endpoint bound to ``provider``.
        """
        if provider.name != requested:
            req = replace(req, base_url=self._base_url_for(provider, None))
        return replace(req, api_key=self._key_for(provider))

    def build_record(
        self,
        req: ChatRequest,
        *,
        outcome: str,
        response: Response | None = None,
        error: BaseException | None = None,
        attempts: int = 1,
        redactions: int = 0,
        prompt_sha256: str = "",
        latency_ms: int | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> AuditRecord:
        """Build the single audit record for one call.

        Never put a credential, a header or a base64 attachment payload in here. The
        leakage test in the suite asserts that by scanning serialised records for the
        fixture key.
        """
        record = AuditRecord(
            model_requested=req.model,
            model_resolved=response.resolved_model if response else None,
            provider=req.provider,
            provider_region=self.registry.region(req.model),
            base_url=req.base_url,
            outcome=_as_outcome(outcome),
            error_type=type(error).__name__ if error else None,
            usage=response.usage if response else None,
            cost=response.cost if response else None,
            latency_ms=latency_ms,
            attempts=attempts,
            policy_snapshot=self.policy.snapshot() if self.policy else {},
            redactions=redactions,
            message_count=len(req.messages),
            input_chars=req.input_chars,
            output_chars=len(response.text) if response else 0,
            prompt_sha256=prompt_sha256,
            response_sha256=body_sha256(response.raw) if response and response.raw else None,
            tags=dict(req.tags),
        )
        if self.audit.record_content:
            record.content = _content_snapshot(req, response, detail)
        elif detail:
            record.content = {"policy": dict(detail)}
        return record

    def close(self) -> None:
        """Close the audit sink. Subclasses also close their HTTP client."""
        self.audit.close()


class Aaron(_ClientBase):
    """The synchronous client.

    ```python
    client = Aaron()
    reply = client.chat("ollama/llama3.1", "Hello")
    print(reply.text)
    ```

    Constructing one is cheap and it owns a connection pool, so build it once and
    keep it. It is safe to share between threads. Close it when you are done, or use
    it as a context manager.
    """

    def __init__(self, **kwargs: Any) -> None:
        client: httpx.Client | None = kwargs.pop("http_client", None)
        super().__init__(**kwargs)
        self._transport = Transport(self.transport_config, client=client)

    @property
    def http(self) -> httpx.Client:
        """The underlying httpx client, created on first use."""
        return self._transport.client

    def chat(
        self, model: str | None = None, messages: Prompt | None = None, **kwargs: Any
    ) -> Response:
        """Send one chat request and wait for the whole reply.

        Args:
            model: Alias or ``provider/model``. Defaults to the client's default.
            messages: A string, a :class:`~aaron.types.Message`, or a sequence.
            **kwargs: Anything :meth:`_ClientBase.build` accepts.

        Returns:
            The completed response, with usage, cost, latency and an audit id.

        Raises:
            PolicyViolation: The policy refused the call.
            ProviderError: The provider returned an error, mapped to its own class.
            TransportError: The call never reached the provider.
        """
        call = self.begin(model, messages, kwargs, stream=False)
        attempt = 0
        while True:
            attempt += 1
            try:
                payload = self._post(call)
                response = call.provider.parse_response(payload, req=call.req)
            except AaronError as error:
                if self.retry.should_retry(error, attempt=attempt):
                    time.sleep(self.retry.delay_for(attempt, error))
                    continue
                call.fail(error, attempt)
                raise
            return call.succeed(response, attempt)

    def _post(self, call: _Call) -> dict[str, Any]:
        """Send one non streaming request and decode it, or raise a typed error."""
        try:
            http_response = self.http.request(
                call.prepared.method, call.prepared.url, **call.send_kwargs()
            )
        except Exception as exc:
            raise _transport_error(exc, call) from exc
        return _decode(call, http_response)

    def stream(
        self, model: str | None = None, messages: Prompt | None = None, **kwargs: Any
    ) -> Stream:
        """Start a streaming call.

        The returned stream holds an open connection. Iterating to the end closes it.
        A caller who breaks out of the loop early must call ``close()`` or use the
        stream as a context manager, otherwise the connection stays open until the
        object is collected.

        Args:
            model: Alias or ``provider/model``.
            messages: The prompt.
            **kwargs: Anything :meth:`_ClientBase.build` accepts.

        Returns:
            A :class:`~aaron.stream.Stream` of events, ending with one done event.

        Raises:
            PolicyViolation: The policy refused the call.
        """
        call = self.begin(model, messages, kwargs, stream=True)
        generator = self._stream_events(call)
        return Stream(generator, close=generator.close)

    def _stream_events(self, call: _Call) -> Generator[StreamEvent, None, None]:
        """Open the connection, translate events, and always write one record."""
        run = _StreamRun(call)
        try:
            with self.http.stream(
                call.prepared.method, call.prepared.url, **call.send_kwargs()
            ) as http_response:
                if http_response.status_code >= 400:
                    http_response.read()
                    raise _error_from(call, http_response)
                lines = (line.encode("utf-8") for line in http_response.iter_lines())
                for event in call.provider.iter_events(lines, req=call.req):
                    yield run.event(event)
        except AaronError as exc:
            run.failed(exc)
            raise
        except httpx.HTTPError as exc:
            error = _transport_error(exc, call)
            run.failed(error)
            raise error from exc
        finally:
            run.finish()

    def extract(
        self,
        model: str | None = None,
        messages: Prompt | None = None,
        *,
        schema: type[M],
        max_repairs: int = DEFAULT_MAX_REPAIRS,
        **kwargs: Any,
    ) -> M:
        """Get a validated pydantic object back instead of text.

        Uses the provider's native schema support where the registry says it exists,
        and JSON mode with a prompted schema where it does not. Either way the result
        is validated locally, because a provider claiming schema support is not the
        same as a model honouring it, and an invalid document is shown back to the
        model with its errors up to ``max_repairs`` times.

        Args:
            model: Alias or ``provider/model``.
            messages: The prompt.
            schema: The pydantic model to produce.
            max_repairs: How many corrective attempts to allow. Zero means fail on
                the first invalid document.
            **kwargs: Anything :meth:`_ClientBase.build` accepts.

        Returns:
            An instance of ``schema``.

        Raises:
            InvalidRequest: No attempt produced a document that validates. The last
                validation error is in the message.
        """
        conversation = _extract_setup(self, model, messages, schema, kwargs)
        for attempt in range(max_repairs + 1):
            response = self.chat(model, conversation, **kwargs)
            parsed, errors = _validate_document(response, schema)
            if parsed is not None:
                return parsed
            if attempt == max_repairs:
                raise _extract_failed(response, schema, attempt + 1, errors)
            conversation = _repair_turn(conversation, response, errors)
        raise AssertionError("unreachable")  # pragma: no cover

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP connection pool and the audit sink."""
        self._transport.close()
        super().close()


class AsyncAaron(_ClientBase):
    """The asynchronous client. The same surface as :class:`Aaron`, with awaits.

    ```python
    client = AsyncAaron()
    reply = await client.chat("openai/gpt-4o", "Hello")
    async for event in client.stream("openai/gpt-4o", "Hello"):
        ...
    await client.aclose()
    ```
    """

    def __init__(self, **kwargs: Any) -> None:
        client: httpx.AsyncClient | None = kwargs.pop("http_client", None)
        super().__init__(**kwargs)
        self._transport = AsyncTransport(self.transport_config, client=client)

    @property
    def http(self) -> httpx.AsyncClient:
        """The underlying httpx async client, created on first use."""
        return self._transport.client

    async def chat(
        self, model: str | None = None, messages: Prompt | None = None, **kwargs: Any
    ) -> Response:
        """Send one chat request and await the whole reply. See :meth:`Aaron.chat`."""
        call = self.begin(model, messages, kwargs, stream=False)
        attempt = 0
        while True:
            attempt += 1
            try:
                payload = await self._post(call)
                response = call.provider.parse_response(payload, req=call.req)
            except AaronError as error:
                if self.retry.should_retry(error, attempt=attempt):
                    await asyncio.sleep(self.retry.delay_for(attempt, error))
                    continue
                call.fail(error, attempt)
                raise
            return call.succeed(response, attempt)

    async def _post(self, call: _Call) -> dict[str, Any]:
        try:
            http_response = await self.http.request(
                call.prepared.method, call.prepared.url, **call.send_kwargs()
            )
        except Exception as exc:
            raise _transport_error(exc, call) from exc
        return _decode(call, http_response)

    def stream(
        self, model: str | None = None, messages: Prompt | None = None, **kwargs: Any
    ) -> AsyncStream:
        """Start a streaming call. See :meth:`Aaron.stream` for the closing rule."""
        call = self.begin(model, messages, kwargs, stream=True)
        generator = self._stream_events(call)
        return AsyncStream(generator, close=generator.aclose)

    async def _stream_events(self, call: _Call) -> AsyncGenerator[StreamEvent, None]:
        """The async twin of :meth:`Aaron._stream_events`.

        Translation is driven line by line through the provider's push parser, so
        nothing is buffered and the first token reaches the caller as soon as it
        arrives.
        """
        run = _StreamRun(call)
        try:
            async with self.http.stream(
                call.prepared.method, call.prepared.url, **call.send_kwargs()
            ) as http_response:
                if http_response.status_code >= 400:
                    await http_response.aread()
                    raise _error_from(call, http_response)
                parser = call.provider.event_parser(call.req)
                async for line in http_response.aiter_lines():
                    for event in parser.feed(line.encode("utf-8")):
                        yield run.event(event)
                for event in parser.finish():
                    yield run.event(event)
        except AaronError as exc:
            run.failed(exc)
            raise
        except httpx.HTTPError as exc:
            error = _transport_error(exc, call)
            run.failed(error)
            raise error from exc
        finally:
            run.finish()

    async def extract(
        self,
        model: str | None = None,
        messages: Prompt | None = None,
        *,
        schema: type[M],
        max_repairs: int = DEFAULT_MAX_REPAIRS,
        **kwargs: Any,
    ) -> M:
        """Get a validated pydantic object back. See :meth:`Aaron.extract`."""
        conversation = _extract_setup(self, model, messages, schema, kwargs)
        for attempt in range(max_repairs + 1):
            response = await self.chat(model, conversation, **kwargs)
            parsed, errors = _validate_document(response, schema)
            if parsed is not None:
                return parsed
            if attempt == max_repairs:
                raise _extract_failed(response, schema, attempt + 1, errors)
            conversation = _repair_turn(conversation, response, errors)
        raise AssertionError("unreachable")  # pragma: no cover

    async def batch(
        self, requests: Sequence[Mapping[str, Any]], *, concurrency: int = 8
    ) -> list[Response | AaronError]:
        """Run many calls with a semaphore and return results in input order.

        Errors are captured rather than raised, so one bad call does not lose the
        results of the others. Anything that is not an :class:`AaronError` still
        propagates, because a bug in the caller's own code should not be swallowed.

        Args:
            requests: One mapping of :meth:`chat` keyword arguments per call.
            concurrency: How many calls may be in flight at once.

        Returns:
            A list as long as ``requests``, each item a response or an error.

        Raises:
            ValueError: ``concurrency`` is not positive.
        """
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        semaphore = asyncio.Semaphore(concurrency)

        async def one(kwargs: Mapping[str, Any]) -> Response | AaronError:
            async with semaphore:
                try:
                    return await self.chat(**kwargs)
                except AaronError as error:
                    return error

        return list(await asyncio.gather(*(one(kwargs) for kwargs in requests)))

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP connection pool and the audit sink."""
        await self._transport.aclose()
        super().close()


# Structured output helpers, shared by both clients ---------------------------------


def _extract_setup(
    client: _ClientBase,
    model: str | None,
    messages: Prompt | None,
    schema: type[M],
    kwargs: dict[str, Any],
) -> list[Message]:
    """Choose native schema or prompted JSON mode, and prime the conversation."""
    canonical = client.resolve_model(model or client.default_model or "")
    conversation = normalise_messages(messages if messages is not None else "")
    spec = JsonSchemaSpec(name=schema.__name__, schema=_schema_of(schema))
    if client.registry.capabilities(canonical).supports("json_schema"):
        kwargs.setdefault("json_schema", spec)
        return conversation
    # No native schema support: ask for JSON mode and put the schema in the prompt.
    kwargs.setdefault("json_mode", True)
    return [
        *conversation,
        Message.user(
            "Reply with a single JSON object and nothing else, matching this JSON Schema:\n"
            + json.dumps(spec.schema, indent=2)
        ),
    ]


def _validate_document(response: Response, schema: type[M]) -> tuple[M | None, str]:
    """Validate a reply against the schema, returning either the object or the errors."""
    try:
        return schema.model_validate_json(_document_of(response)), ""
    except ValidationError as exc:
        return None, "; ".join(
            f"{'.'.join(str(p) for p in error['loc']) or 'root'}: {error['msg']}"
            for error in exc.errors()[:3]
        )


def _repair_turn(conversation: list[Message], response: Response, errors: str) -> list[Message]:
    """Append the invalid reply and a corrective instruction."""
    return [
        *conversation,
        Message.assistant(response.text or "{}"),
        Message.user(
            f"That reply did not validate: {errors}. Reply again with only the "
            f"corrected JSON object, with no markdown around it."
        ),
    ]


def _extract_failed(
    response: Response, schema: type[BaseModel], attempts: int, errors: str
) -> InvalidRequest:
    return InvalidRequest(
        f"{response.model} did not produce a valid {schema.__name__} after "
        f"{attempts} attempts: {errors}",
        provider=response.model.split("/", 1)[0],
        model=response.model,
        raw=response.raw,
    )


def _schema_of(schema: type[BaseModel]) -> dict[str, Any]:
    """A provider ready JSON Schema: references inlined, extra keys forbidden."""
    from .types import _inline_model_schema

    inlined = _inline_model_schema(schema)
    inlined.setdefault("type", "object")
    inlined["additionalProperties"] = False
    return inlined


def _document_of(response: Response) -> str:
    """Pull the JSON document out of a reply, wherever the provider put it.

    Anthropic constrains output by forcing a tool call, so the document arrives as
    tool arguments rather than as text. Everything else returns text, sometimes
    wrapped in a markdown fence.
    """
    if response.text.strip():
        return _strip_fences(response.text)
    if response.tool_calls:
        return json.dumps(response.tool_calls[0].arguments, default=str)
    return ""


def _strip_fences(text: str) -> str:
    """Remove a markdown code fence, which JSON mode models add anyway."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.removeprefix("```")
    newline = body.find("\n")
    if newline != -1 and " " not in body[:newline]:
        body = body[newline + 1 :]  # drop a language tag such as ```json
    return body.removesuffix("```").strip()


# Transport and audit helpers ------------------------------------------------------


def _decode(call: _Call, response: httpx.Response) -> dict[str, Any]:
    """Decode a non streaming response, or raise the provider's mapped error."""
    if response.status_code >= 400:
        raise _error_from(call, response)
    try:
        payload = response.json()
    except ValueError as exc:
        raise _bad_body(
            call, response, f"a body that is not JSON: {response.text[:200]!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise _bad_body(call, response, f"{type(payload).__name__}, expected a JSON object")
    return payload


def _bad_body(call: _Call, response: httpx.Response, detail: str) -> ProviderError:
    """A successful status with a body we cannot use is the provider's fault, not ours."""
    return ProviderError(
        f"{call.provider.name} returned {detail}",
        provider=call.provider.name,
        model=call.req.model,
        status_code=response.status_code,
    )


def _error_from(call: _Call, response: httpx.Response) -> AaronError:
    """Ask the provider to map an error response, and fill in what it cannot know."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    error = call.provider.map_error(
        response.status_code, payload if isinstance(payload, dict) else None, response.text
    )
    error.model = error.model or call.req.model
    error.provider = error.provider or call.provider.name
    error.status_code = error.status_code or response.status_code
    error.request_id = error.request_id or _provider_request_id(response)
    if isinstance(error, RateLimitError) and error.retry_after is None:
        error.retry_after = _retry_after(response)
    return error


def _provider_request_id(response: httpx.Response) -> str | None:
    headers = ("x-request-id", "request-id", "cf-ray", "x-amzn-requestid")
    return next((str(response.headers[h]) for h in headers if response.headers.get(h)), None)


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None  # an HTTP date, not worth parsing for a backoff hint


def _transport_error(exc: Exception, call: _Call) -> TransportError:
    return translate_httpx_error(
        exc,
        provider=call.provider.name,
        model=call.req.model,
        base_url=call.req.base_url,
        local=call.provider.local,
    )


def _as_outcome(value: str) -> Outcome:
    """Narrow a string to the audit outcome literal."""
    if value in ("ok", "policy_violation", "provider_error", "timeout", "cancelled"):
        return cast(Outcome, value)
    return "provider_error"


def _outcome_for(error: BaseException) -> Outcome:
    """Which audit outcome a failure counts as."""
    if isinstance(error, PolicyViolation):
        return "policy_violation"
    return "timeout" if isinstance(error, AaronTimeout) else "provider_error"


def _content_snapshot(
    req: ChatRequest, response: Response | None, detail: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Build the opt in content block, with attachment bytes replaced by digests."""
    content: dict[str, Any] = {
        "messages": [_message_snapshot(message) for message in req.messages],
        "output": response.text if response else "",
    }
    if response is not None and response.tool_calls:
        content["tool_calls"] = [call.model_dump() for call in response.tool_calls]
    if detail:
        content["policy"] = dict(detail)
    return content


def _message_snapshot(message: Message) -> dict[str, Any]:
    """Serialise a message. A base64 payload becomes a digest, never a log line."""
    entry: dict[str, Any] = {
        "role": message.role,
        "content": [_part_snapshot(part) for part in message.content],
    }
    if message.tool_calls:
        entry["tool_calls"] = [call.model_dump() for call in message.tool_calls]
    if message.tool_call_id:
        entry["tool_call_id"] = message.tool_call_id
    return entry


def _part_snapshot(part: TextPart | ImagePart | DocumentPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    return {
        "type": part.type,
        "media_type": part.media_type,
        "bytes": len(part.data),
        "sha256": body_sha256({"data": part.data}),
    }


# Configuration file helpers -------------------------------------------------------


def _load_config_file(path: str | Path | None) -> dict[str, Any]:
    """Read ``aaron.toml`` from the working directory, or an explicit path."""
    file = Path(path) if path is not None else Path(CONFIG_FILE)
    if not file.is_file():
        if path is not None:
            raise ConfigurationError(f"config file not found: {file}")
        return {}
    try:
        data = tomllib.loads(file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"{file} is not valid TOML: {exc}") from exc
    section = data.get("aaron", data)
    return section if isinstance(section, dict) else {}


def _mapping(value: Any) -> dict[str, str]:
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}


def _str_or_none(value: Any) -> str | None:
    return str(value) if isinstance(value, str | int | float) else None


def _first_float(*values: Any) -> float | None:
    """The first value that is a number, ignoring and warning about the rest."""
    for value in (v for v in values if v is not None):
        try:
            return float(value)
        except (TypeError, ValueError):
            log.warning("ignoring non numeric timeout value %r", value)
    return None


def _normalise_url(url: str) -> str:
    """Accept a bare host, as ``OLLAMA_HOST`` is often written."""
    trimmed = url.strip().rstrip("/")
    if not trimmed.startswith(("http://", "https://")):
        trimmed = f"http://{trimmed}"
    return trimmed
