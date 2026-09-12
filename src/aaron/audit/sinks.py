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
from typing import Protocol, TextIO, runtime_checkable

from .record import AuditRecord

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
