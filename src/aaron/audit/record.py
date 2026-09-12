# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""The audit record.

Every call produces exactly one record, whether it succeeded, was refused by
policy, or failed at the provider. The record is designed to be evidence: it says
what was asked for, what was actually used, where it was processed, what it cost,
and what was stripped before it left, without containing the prompt itself unless
the operator explicitly turned that on.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..types import Cost, Usage

Outcome = Literal["ok", "policy_violation", "provider_error", "timeout", "cancelled"]


def _now() -> datetime:
    return datetime.now(UTC)


class AuditRecord(BaseModel):
    """One call, described for a compliance reader.

    Attributes:
        prompt_sha256: Hash of the exact normalised body that was sent, so a caller
            can prove which prompt produced which answer without storing either.
        policy_snapshot: The rules that were in force at the time of the call.
        policy_detail: For a refused call, the rule that refused it and every
            fallback candidate that was tried. Metadata only, never prompt text, so
            it is present whatever ``record_content`` is set to.
        redactions: How many replacements the redactors made before sending.
        content: The prompt and completion. Populated only when the sink was
            constructed with ``record_content=True``, which may place personal data
            in the log.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=_now)
    model_requested: str
    model_resolved: str | None = None
    provider: str
    provider_region: str | None = None
    base_url: str = ""
    outcome: Outcome = "ok"
    error_type: str | None = None
    usage: Usage | None = None
    cost: Cost | None = None
    latency_ms: int | None = None
    attempts: int = 1
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    policy_detail: dict[str, Any] | None = None
    redactions: int = 0
    message_count: int = 0
    input_chars: int = 0
    output_chars: int = 0
    prompt_sha256: str = ""
    response_sha256: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    content: dict[str, Any] | None = None

    def to_json(self) -> str:
        """Serialise to one line of JSON, as written to a JSONL sink."""
        return self.model_dump_json(exclude_none=False)
