"""Tests for the audit layer.

The audit record is the artefact a compliance reviewer reads, so these tests treat it
as a contract: which fields are always present, what is deliberately absent, and that
a broken sink can never break a call.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest
import respx

from aaron import Aaron, Message, Policy
from aaron.audit import AuditLog, AuditRecord, CallbackSink, JsonlSink, NullSink
from aaron.errors import RateLimitError
from aaron.retry import RetryPolicy
from conftest import load, read_jsonl

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/chat"


@pytest.fixture
def records() -> list[AuditRecord]:
    """A list a CallbackSink appends to."""
    return []


@pytest.fixture
def client(records: list[AuditRecord]) -> Aaron:
    """A client auditing into memory."""
    return Aaron(
        api_keys={"openai": "k"},
        audit=CallbackSink(records.append),
        retry=RetryPolicy(attempts=1),
    )


class TestOneRecordPerCall:
    """Exactly one record per call, whatever happened."""

    @respx.mock
    def test_a_successful_call_writes_one_record(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "hi")
        assert len(records) == 1
        assert records[0].outcome == "ok"

    @respx.mock
    def test_a_failed_call_writes_one_record(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(429, json={"error": {}}))
        with pytest.raises(RateLimitError):
            client.chat("openai/gpt-4o", "hi")

        assert len(records) == 1
        assert records[0].outcome == "provider_error"
        assert records[0].error_type == "RateLimitError"

    @respx.mock
    def test_a_refused_call_writes_one_record_before_raising(
        self, records: list[AuditRecord]
    ) -> None:
        from aaron.errors import PolicyViolation

        instance = Aaron(
            api_keys={"openai": "k"},
            policy=Policy(deny=["openai/*"]),
            audit=CallbackSink(records.append),
        )
        with pytest.raises(PolicyViolation):
            instance.chat("openai/gpt-4o", "hi")

        assert len(records) == 1
        assert records[0].outcome == "policy_violation"
        assert records[0].policy_snapshot["deny"] == ["openai/*"]

    @respx.mock
    def test_a_stream_writes_one_record_when_it_finishes(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        from conftest import sse

        body = sse(
            {"id": "s", "choices": [{"index": 0, "delta": {"content": "hi"}}]},
            {"id": "s", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, text=body))
        list(client.stream("openai/gpt-4o", "hi"))

        assert len(records) == 1
        assert records[0].outcome == "ok"

    @respx.mock
    def test_an_abandoned_stream_is_recorded_as_cancelled(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        from conftest import sse

        body = sse(
            {"id": "s", "choices": [{"index": 0, "delta": {"content": "one"}}]},
            {"id": "s", "choices": [{"index": 0, "delta": {"content": "two"}}]},
            {"id": "s", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, text=body))
        with client.stream("openai/gpt-4o", "hi") as stream:
            next(iter(stream))

        assert len(records) == 1
        assert records[0].outcome == "cancelled"

    @respx.mock
    def test_each_retry_is_counted_on_the_single_record(self, records: list[AuditRecord]) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(500, json={"error": {}}))
        instance = Aaron(
            api_keys={"openai": "k"},
            audit=CallbackSink(records.append),
            retry=RetryPolicy(attempts=3, initial_delay=0.0),
        )
        with pytest.raises(Exception, match=r"."):
            instance.chat("openai/gpt-4o", "hi")

        assert len(records) == 1
        assert records[0].attempts == 3


class TestRecordContents:
    """What every record carries, and what it must not."""

    @respx.mock
    def test_the_fields_a_reviewer_needs_are_all_present(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "What is the capital of Estonia?")
        record = records[0]

        assert record.id
        assert record.timestamp
        assert record.model_requested == "openai/gpt-4o"
        assert record.model_resolved == "gpt-4o-2024-11-20"
        assert record.provider == "openai"
        assert record.provider_region == "us"
        assert record.base_url == "https://api.openai.com/v1"
        assert record.usage is not None
        assert record.cost is not None
        assert record.latency_ms is not None
        assert record.attempts == 1
        assert record.message_count == 1
        assert record.input_chars > 0
        assert record.output_chars > 0
        assert record.prompt_sha256
        assert record.response_sha256

    @respx.mock
    def test_content_is_absent_by_default(self, client: Aaron, records: list[AuditRecord]) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "a secret business plan")

        record = records[0]
        assert record.content is None
        assert "a secret business plan" not in record.to_json()

    @respx.mock
    def test_the_prompt_hash_is_stable_and_the_prompt_is_not_recoverable(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "same prompt")
        client.chat("openai/gpt-4o", "same prompt")
        client.chat("openai/gpt-4o", "different prompt")

        assert records[0].prompt_sha256 == records[1].prompt_sha256
        assert records[0].prompt_sha256 != records[2].prompt_sha256
        assert len(records[0].prompt_sha256) == 64

    @respx.mock
    def test_the_id_is_returned_on_the_response_so_it_can_be_cited(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        response = client.chat("openai/gpt-4o", "hi")
        assert response.audit_id == records[0].id

    @respx.mock
    def test_tags_are_carried_through(self, records: list[AuditRecord]) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        log = AuditLog(CallbackSink(records.append), service="billing")
        instance = Aaron(api_keys={"openai": "k"}, audit=log)
        instance.chat("openai/gpt-4o", "hi", tags={"request": "r-1"})

        assert records[0].tags == {"service": "billing", "request": "r-1"}

    @respx.mock
    def test_a_record_is_json_serialisable_and_round_trips(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "hi")

        text = records[0].to_json()
        assert "\n" not in text, "a JSONL line must not contain a newline"
        again = AuditRecord.model_validate_json(text)
        assert again.id == records[0].id


class TestOptInContent:
    """Recording content is explicit, because it may put personal data in a log."""

    @respx.mock
    def test_content_appears_only_when_asked_for(self, tmp_path: Path) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        log = tmp_path / "with-content.jsonl"
        instance = Aaron(api_keys={"openai": "k"}, audit=log, record_content=True)
        instance.chat("openai/gpt-4o", "the prompt text")
        instance.close()

        record = read_jsonl(log)[0]
        assert record["content"]["messages"][0]["content"][0]["text"] == "the prompt text"
        assert "Tallinn" in json.dumps(record["content"])

    @respx.mock
    def test_redaction_happens_before_content_is_recorded(self, tmp_path: Path) -> None:
        from aaron.policy.redaction import EmailRedactor

        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        log = tmp_path / "redacted.jsonl"
        instance = Aaron(
            api_keys={"openai": "k"},
            audit=log,
            record_content=True,
            policy=Policy(redactors=[EmailRedactor()]),
        )
        instance.chat("openai/gpt-4o", "mail shb@3sholding.com")
        instance.close()

        assert "shb@3sholding.com" not in log.read_text()
        assert "[EMAIL]" in log.read_text()


class TestSinks:
    """A sink is a protocol, and a failing one must never break a call."""

    def test_the_default_sink_discards(self) -> None:
        assert isinstance(Aaron().audit.sink, NullSink)

    @respx.mock
    def test_jsonl_writes_one_line_per_call_and_creates_the_directory(
        self, audit_path: Path
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        instance = Aaron(api_keys={"openai": "k"}, audit=audit_path)
        instance.chat("openai/gpt-4o", "one")
        instance.chat("openai/gpt-4o", "two")
        instance.close()

        assert audit_path.parent.is_dir()
        assert len(read_jsonl(audit_path)) == 2

    @respx.mock
    def test_jsonl_appends_rather_than_truncating(self, audit_path: Path) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        for _ in range(2):
            instance = Aaron(api_keys={"openai": "k"}, audit=audit_path)
            instance.chat("openai/gpt-4o", "hi")
            instance.close()
        assert len(read_jsonl(audit_path)) == 2

    @respx.mock
    def test_a_sink_that_raises_does_not_break_the_call(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def explode(record: AuditRecord) -> None:
            raise RuntimeError("the audit database is down")

        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        instance = Aaron(api_keys={"openai": "k"}, audit=CallbackSink(explode))

        with caplog.at_level(logging.ERROR, logger="aaron"):
            response = instance.chat("openai/gpt-4o", "hi")

        assert response.text, "the call must succeed even though auditing failed"
        assert "the audit database is down" in caplog.text

    @respx.mock
    def test_a_failing_sink_is_logged_once_not_per_call(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def explode(record: AuditRecord) -> None:
            raise RuntimeError("still down")

        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        instance = Aaron(api_keys={"openai": "k"}, audit=CallbackSink(explode))

        with caplog.at_level(logging.ERROR, logger="aaron"):
            for _ in range(5):
                instance.chat("openai/gpt-4o", "hi")

        # caplog.text includes the traceback, so count records rather than substrings.
        failures = [r for r in caplog.records if "audit sink" in r.getMessage()]
        assert len(failures) == 1

    def test_a_jsonl_sink_can_be_closed_twice(self, audit_path: Path) -> None:
        sink = JsonlSink(str(audit_path))
        sink.write(AuditRecord(model_requested="openai/gpt-4o", provider="openai"))
        sink.close()
        sink.close()
        assert len(read_jsonl(audit_path)) == 1


class TestEnvironmentConfiguration:
    """Auditing can be turned on without touching the code."""

    @respx.mock
    def test_the_path_can_come_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = tmp_path / "from-env.jsonl"
        monkeypatch.setenv("AARON_AUDIT_PATH", str(log))
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))

        instance = Aaron(api_keys={"openai": "k"})
        instance.chat("openai/gpt-4o", "hi")
        instance.close()

        assert len(read_jsonl(log)) == 1


class TestSummariseCommand:
    """``python -m aaron.audit summarise`` reads a log without leaking content."""

    @respx.mock
    def test_it_totals_by_model_and_never_prints_content(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aaron.audit.__main__ import main

        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=load("ollama_chat")))
        log = tmp_path / "audit.jsonl"
        instance = Aaron(api_keys={"openai": "k"}, audit=log, record_content=True)
        instance.chat("openai/gpt-4o", "a confidential prompt")
        instance.chat("ollama/llama3.1", "another confidential prompt")
        instance.close()

        assert main(["summarise", str(log)]) == 0
        output = capsys.readouterr().out
        assert "openai/gpt-4o" in output
        assert "ollama/llama3.1" in output
        assert "confidential" not in output

    def test_a_missing_file_is_an_error_not_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aaron.audit.__main__ import main

        assert main(["summarise", str(tmp_path / "nope.jsonl")]) == 2
        assert "not found" in capsys.readouterr().err

    def test_a_malformed_line_is_skipped_with_a_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aaron.audit.__main__ import main

        log = tmp_path / "mixed.jsonl"
        good = AuditRecord(model_requested="openai/gpt-4o", provider="openai").to_json()
        log.write_text(f"{good}\nnot json at all\n", encoding="utf-8")

        assert main(["summarise", str(log)]) == 0
        captured = capsys.readouterr()
        assert "openai/gpt-4o" in captured.out
        assert "malformed" in (captured.out + captured.err).lower()


class TestMessageSnapshot:
    """The counts are derived without keeping the text."""

    @respx.mock
    def test_an_image_is_counted_but_not_stored(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        png = bytes.fromhex("89504e470d0a1a0a") + b"a" * 5000
        client.chat("openai/gpt-4o", Message.user("what is this", images=[png]))

        record = records[0]
        assert record.content is None
        # The base64 image must not inflate the character count of the text.
        assert record.input_chars == len("what is this")
