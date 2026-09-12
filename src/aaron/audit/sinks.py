# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Where audit records go.

The default is :class:`NullSink`, which records nothing: a library must not start
writing files to a user's disk because they imported it. Content is never recorded
unless the operator sets ``record_content=True``, and doing so may place personal
data in the log.

A sink failure never breaks the call. A full disk is a reason to lose a log line,
not a reason to lose a completion, so every write goes through
:meth:`AuditLog.write`, which catches, logs once, and continues.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO, runtime_checkable

from ..errors import ConfigurationError
from .record import AuditRecord

if TYPE_CHECKING:  # pragma: no cover - the extra is not installed for type checking
    from opentelemetry.trace import Tracer

log = logging.getLogger("aaron")


@runtime_checkable
class Sink(Protocol):
    """Somewhere audit records can be written.

    Attributes:
        record_content: Whether the client should populate ``AuditRecord.content``
            for this sink. The client asks the sink, so the decision is made once,
            where the destination is known.
    """

    record_content: bool

    def write(self, record: AuditRecord) -> None:
        """Persist one record."""
        ...

    def close(self) -> None:
        """Release any resources. Must be safe to call twice."""
        ...


class NullSink:
    """Discards every record. The default, so Aaron writes nothing uninvited."""

    record_content = False

    def write(self, record: AuditRecord) -> None:
        """Do nothing."""

    def close(self) -> None:
        """Do nothing."""


class JsonlSink:
    """Appends one JSON object per line to a file.

    Args:
        path: File to append to. Parent directories are created.
        record_content: Record prompts and completions as well as metadata. Off by
            default. Turning it on may place personal data in the log, so treat the
            file as a processing record under the GDPR and set a retention period.

    The file handle is opened on first write and every line is flushed, so a crash
    loses at most the line in flight. Writes are serialised with a lock, which makes
    the sink safe to share between threads.
    """

    def __init__(self, path: str | Path, *, record_content: bool = False) -> None:
        self.path = Path(path)
        self.record_content = record_content
        self._lock = threading.Lock()
        self._handle: TextIO | None = None

    def write(self, record: AuditRecord) -> None:
        """Append one record as a single line of JSON, flushed immediately."""
        line = record.to_json()
        with self._lock:
            if self._handle is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self.path.open("a", encoding="utf-8")
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        """Close the file if it was opened."""
        with self._lock:
            handle = self._handle
            self._handle = None
        if handle is not None:
            handle.close()


class CallbackSink:
    """Hands each record to a function, for shipping records somewhere else.

    Args:
        fn: Called with each record. Any exception it raises is caught and logged
            by :meth:`AuditLog.write`, so a broken exporter cannot break a call.
        record_content: Whether to populate ``AuditRecord.content``.
    """

    def __init__(self, fn: Callable[[AuditRecord], None], *, record_content: bool = False) -> None:
        self._fn = fn
        self.record_content = record_content

    def write(self, record: AuditRecord) -> None:
        """Invoke the callback."""
        self._fn(record)

    def close(self) -> None:
        """Nothing to release."""


class OtelSink:
    """Emits one OpenTelemetry span per call, for a team that already has tracing.

    Attribute names follow the GenAI semantic conventions where they exist, so an
    existing dashboard recognises them, with Aaron's own additions under ``aaron.``
    for cost, policy and redaction counts.

    Args:
        tracer: The tracer to use. Defaults to one named ``aaron`` from the global
            provider, which is what an application that has configured OpenTelemetry
            already expects.
        record_content: Whether to put prompts and completions on the span. Off by
            default, and a span is a particularly bad place for personal data because
            it is usually shipped to a third party collector.

    Raises:
        ConfigurationError: OpenTelemetry is not installed. It is an optional extra,
            imported here rather than at module import time so the base install stays
            at two dependencies.
    """

    def __init__(self, tracer: Tracer | None = None, *, record_content: bool = False) -> None:
        self.record_content = record_content
        self._tracer = tracer if tracer is not None else _default_tracer()

    def write(self, record: AuditRecord) -> None:
        """Emit the record as a span covering the real duration of the call."""
        start = int(record.timestamp.timestamp() * 1_000_000_000)
        span = self._tracer.start_span(
            f"chat {record.model_requested}",
            start_time=start,
            attributes=_span_attributes(record),
        )
        if record.error_type:
            span.set_status(_error_status(record))
        if self.record_content and record.content is not None:
            span.add_event("aaron.content", attributes={"aaron.content": str(record.content)})
        span.end(end_time=start + int((record.latency_ms or 0) * 1_000_000))

    def close(self) -> None:
        """Nothing to release. Flushing is the tracer provider's job, not ours."""


def _default_tracer() -> Tracer:
    """Fetch the ``aaron`` tracer, explaining the missing extra if it is not there."""
    try:
        from opentelemetry import trace
    except ImportError as exc:  # pragma: no cover - exercised with a stubbed import
        raise ConfigurationError(
            "OtelSink needs OpenTelemetry. Install it with: pip install 'aaron-llm[otel]'"
        ) from exc
    return trace.get_tracer("aaron")


def _span_attributes(record: AuditRecord) -> dict[str, Any]:
    """Flatten a record into span attributes, credentials and content excluded."""
    attributes: dict[str, Any] = {
        "gen_ai.system": record.provider,
        "gen_ai.request.model": record.model_requested,
        "gen_ai.operation.name": "chat",
        "server.address": record.base_url,
        "aaron.outcome": record.outcome,
        "aaron.attempts": record.attempts,
        "aaron.redactions": record.redactions,
        "aaron.audit_id": record.id,
    }
    if record.model_resolved:
        attributes["gen_ai.response.model"] = record.model_resolved
    if record.provider_region:
        attributes["aaron.provider_region"] = record.provider_region
    if record.usage:
        attributes["gen_ai.usage.input_tokens"] = record.usage.input_tokens
        attributes["gen_ai.usage.output_tokens"] = record.usage.output_tokens
    if record.cost:
        attributes["aaron.cost_usd"] = record.cost.usd
        attributes["aaron.cost_estimated"] = record.cost.estimated
    if record.policy_detail:
        attributes["aaron.policy_rule"] = str(record.policy_detail.get("rule", ""))
    attributes.update({f"aaron.tag.{key}": value for key, value in record.tags.items()})
    return attributes


def _error_status(record: AuditRecord) -> Any:
    """Build an error status, importing the status classes only when one is needed."""
    from opentelemetry.trace import Status, StatusCode

    return Status(StatusCode.ERROR, record.error_type)


class AuditLog:
    """A sink plus the client wide tags, and the guarantee that writes never raise.

    ``client.audit.tag(tenant="acme")`` sets tags for every call from this client;
    ``chat(..., tags={...})`` overrides them per call.
    """

    def __init__(self, sink: Sink | None = None, **tags: str) -> None:
        self.sink: Sink = sink or NullSink()
        self.tags: dict[str, str] = {key: str(value) for key, value in tags.items()}
        self._failed = False

    @property
    def record_content(self) -> bool:
        """Whether the destination sink asked for prompt and completion content."""
        return bool(getattr(self.sink, "record_content", False))

    def tag(self, **tags: str) -> None:
        """Set or replace tags applied to every record from this client.

        Args:
            tags: Arbitrary string labels, for example ``tenant="acme"`` or
                ``purpose="support"``. Values are stringified.
        """
        self.tags.update({key: str(value) for key, value in tags.items()})

    def merge_tags(self, per_call: dict[str, str] | None) -> dict[str, str]:
        """Combine client tags with per call tags, the latter winning."""
        return {**self.tags, **(per_call or {})}

    def write(self, record: AuditRecord) -> None:
        """Write a record, swallowing and logging any sink failure exactly once.

        Args:
            record: The record to persist.
        """
        try:
            self.sink.write(record)
        except Exception:
            if not self._failed:
                self._failed = True
                log.exception(
                    "audit sink %s failed, further failures from this client will be silent. "
                    "Calls continue unaffected.",
                    type(self.sink).__name__,
                )

    def close(self) -> None:
        """Close the underlying sink, ignoring a failure to do so."""
        try:
            self.sink.close()
        except Exception:  # closing must never raise into a caller's teardown
            log.debug("audit sink %s failed to close", type(self.sink).__name__, exc_info=True)
