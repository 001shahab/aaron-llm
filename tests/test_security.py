"""Section 18 of the specification, as executable assertions.

The rule is absolute: an API key, an ``Authorization`` header and any ``x-api-key``
value must never appear in an audit record, an exception message, a repr, or a debug
log. These tests scan serialised output for the fixture key rather than trusting that
each individual code path remembered to mask.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import io
import json
import logging
import pickle
import traceback
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from aaron import Aaron, AsyncAaron, Message, Policy
from aaron.audit import AuditRecord, CallbackSink
from aaron.errors import AaronError, MissingAPIKey
from aaron.retry import RetryPolicy
from conftest import FIXTURE_KEY, load

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def scan(*values: Any) -> None:
    """Fail if the fixture key appears in any rendering of any value."""
    for value in values:
        for rendered in (str(value), repr(value)):
            assert FIXTURE_KEY not in rendered, f"the api key leaked into {type(value).__name__}"


@pytest.fixture
def records() -> list[AuditRecord]:
    """Audit records captured in memory."""
    return []


@pytest.fixture
def client(records: list[AuditRecord]) -> Aaron:
    """A client holding the fixture key, auditing into memory."""
    return Aaron(
        api_keys={"openai": FIXTURE_KEY, "anthropic": FIXTURE_KEY},
        audit=CallbackSink(records.append),
        retry=RetryPolicy(attempts=1),
    )


class TestAuditRecords:
    """The record is written to disk and read by other people. It must be clean."""

    @respx.mock
    def test_a_successful_call_leaves_no_key_in_the_record(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "hi")

        record = records[0]
        assert FIXTURE_KEY not in record.to_json()
        scan(record, record.model_dump(), record.policy_snapshot)

    @respx.mock
    def test_a_failed_call_leaves_no_key_in_the_record(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(401, json={"error": {"message": "invalid api key"}})
        )
        with pytest.raises(AaronError):
            client.chat("openai/gpt-4o", "hi")

        assert FIXTURE_KEY not in records[0].to_json()

    @respx.mock
    def test_a_record_with_content_still_has_no_key(self, tmp_path: Path) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        log = tmp_path / "audit.jsonl"
        instance = Aaron(api_keys={"openai": FIXTURE_KEY}, audit=log, record_content=True)
        instance.chat("openai/gpt-4o", f"do not echo {FIXTURE_KEY} back")
        instance.close()

        # The prompt itself mentioned the key, which is the caller's business, but the
        # base_url, headers and policy snapshot must not add one of their own.
        record = json.loads(log.read_text().splitlines()[0])
        record.pop("content")
        assert FIXTURE_KEY not in json.dumps(record)

    @respx.mock
    def test_the_record_never_carries_a_headers_field_at_all(
        self, client: Aaron, records: list[AuditRecord]
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "hi")
        fields = set(records[0].model_dump())
        assert "headers" not in fields
        assert "api_key" not in fields
        assert "auth" not in fields


class TestExceptions:
    """An exception ends up in a bug report, a log aggregator and a terminal."""

    @respx.mock
    def test_an_authentication_failure_does_not_echo_the_key(self, client: Aaron) -> None:
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(401, json={"error": {"message": "invalid key"}})
        )
        with pytest.raises(AaronError) as info:
            client.chat("openai/gpt-4o", "hi")

        error = info.value
        scan(error, error.raw, error.__dict__)
        assert FIXTURE_KEY not in "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    @respx.mock
    def test_a_transport_failure_does_not_echo_the_key(self, client: Aaron) -> None:
        respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("no route to host"))
        with pytest.raises(AaronError) as info:
            client.chat("openai/gpt-4o", "hi")
        scan(info.value, info.value.__dict__)

    def test_a_missing_key_error_does_not_quote_another_provider_key(
        self, client: Aaron
    ) -> None:
        with pytest.raises(MissingAPIKey) as info:
            client.chat("google/gemini-2.5-flash", "hi")
        scan(info.value)

    @respx.mock
    def test_a_provider_error_body_is_kept_but_carries_no_header(self, client: Aaron) -> None:
        # A provider that echoes the key in its own error body would be its own bug,
        # so this checks that we at least do not add the header to `raw` ourselves.
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(500, json={"error": {"m": "x"}}))
        with pytest.raises(AaronError) as info:
            client.chat("openai/gpt-4o", "hi")
        assert "authorization" not in json.dumps(info.value.raw or {}).lower()


class TestReprs:
    """Every object a developer might print."""

    def test_the_client_repr_and_attributes(self, client: Aaron) -> None:
        scan(client, client.__dict__, client.audit, client.registry)

    def test_the_prepared_request_from_dry_run(self, client: Aaron) -> None:
        prepared = client.dry_run("openai/gpt-4o", "hi")
        scan(prepared, prepared.headers, prepared.body, prepared.auth)

    def test_the_masked_header_still_shows_which_header_it_was(self, client: Aaron) -> None:
        prepared = client.dry_run("openai/gpt-4o", "hi")
        assert "authorization" in prepared.headers
        assert prepared.headers["authorization"] == "****"

    def test_x_api_key_is_masked_too(self, client: Aaron) -> None:
        prepared = client.dry_run("anthropic/claude-sonnet-4-5", "hi")
        assert prepared.headers["x-api-key"] == "****"
        scan(prepared)

    def test_the_internal_request_object(self, client: Aaron) -> None:
        request = client.build("openai/gpt-4o", "hi")
        scan(request)

    @respx.mock
    def test_the_response_and_its_message(self, client: Aaron) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        response = client.chat("openai/gpt-4o", "hi")
        scan(response, response.message, response.usage, response.cost, response.raw)

    @respx.mock
    def test_a_stream_and_its_events(self, client: Aaron) -> None:
        from conftest import sse

        body = sse(
            {"id": "s", "choices": [{"index": 0, "delta": {"content": "hi"}}]},
            {"id": "s", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, text=body))
        for event in client.stream("openai/gpt-4o", "hi"):
            scan(event)


class TestLogging:
    """Debug logging is the classic accidental leak."""

    @respx.mock
    def test_nothing_at_debug_level_contains_the_key(
        self, client: Aaron, caplog: pytest.LogCaptureFixture
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        with caplog.at_level(logging.DEBUG):
            client.chat("openai/gpt-4o", "hi")
        assert FIXTURE_KEY not in caplog.text

    @respx.mock
    def test_nothing_at_debug_level_leaks_on_failure(
        self, client: Aaron, caplog: pytest.LogCaptureFixture
    ) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(500, json={"error": {}}))
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(AaronError):
                client.chat("openai/gpt-4o", "hi")
        assert FIXTURE_KEY not in caplog.text

    def test_the_library_logs_only_under_its_own_logger(self) -> None:
        # A caller must be able to silence or route everything with one name.
        import aaron.audit.sinks as sinks
        import aaron.client as client_module
        import aaron.registry as registry_module

        for module in (client_module, sinks, registry_module):
            logger = getattr(module, "log", None)
            if logger is not None:
                assert logger.name == "aaron" or logger.name.startswith("aaron.")


class TestSecretValueDiscipline:
    """The wrapper is the mechanism, so its edges are tested directly."""

    def test_a_secret_survives_pickling_without_exposing_a_repr(self) -> None:
        from aaron.types import SecretValue

        secret = SecretValue(FIXTURE_KEY)
        restored = pickle.loads(pickle.dumps(secret))
        assert restored.get() == FIXTURE_KEY
        scan(restored)

    def test_writing_a_secret_to_a_stream_with_print_style_formatting(self) -> None:
        from aaron.types import SecretValue

        buffer = io.StringIO()
        print(SecretValue(FIXTURE_KEY), file=buffer)
        print(f"{SecretValue(FIXTURE_KEY)}", file=buffer)
        print("%s" % SecretValue(FIXTURE_KEY), file=buffer)  # noqa: UP031
        print(json.dumps({"k": SecretValue(FIXTURE_KEY).masked()}), file=buffer)
        assert FIXTURE_KEY not in buffer.getvalue()

    def test_a_callable_key_source_is_not_stored_as_a_string(self) -> None:
        instance = Aaron(api_keys={"openai": lambda: FIXTURE_KEY})
        scan(instance.__dict__)

    def test_a_key_from_the_environment_is_not_cached_on_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", FIXTURE_KEY)
        instance = Aaron()
        instance.dry_run("openai/gpt-4o", "hi")
        scan(instance.__dict__)

    def test_there_is_no_module_level_key_cache(self) -> None:
        import aaron
        import aaron.client
        import aaron.providers.openai

        for module in (aaron, aaron.client, aaron.providers.openai):
            for name, value in vars(module).items():
                if name.startswith("__"):
                    continue
                assert FIXTURE_KEY not in str(value)


class TestTheKeyActuallyReachesTheProvider:
    """Masking is only correct if the real key still gets sent."""

    @respx.mock
    def test_openai_receives_the_real_key(self, client: Aaron) -> None:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        client.chat("openai/gpt-4o", "hi")
        assert route.calls[0].request.headers["authorization"] == f"Bearer {FIXTURE_KEY}"

    @respx.mock
    def test_anthropic_receives_the_real_key(self, client: Aaron) -> None:
        route = respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, json=load("anthropic_chat"))
        )
        client.chat("anthropic/claude-sonnet-4-5", "hi")
        assert route.calls[0].request.headers["x-api-key"] == FIXTURE_KEY

    @respx.mock
    async def test_the_async_client_sends_it_too(self) -> None:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        async with AsyncAaron(api_keys={"openai": FIXTURE_KEY}) as instance:
            await instance.chat("openai/gpt-4o", "hi")
        assert route.calls[0].request.headers["authorization"] == f"Bearer {FIXTURE_KEY}"


class TestNoTelemetry:
    """Section 18 forbids telemetry of any kind."""

    @respx.mock
    def test_exactly_one_request_is_made_per_call(self) -> None:
        # Any extra host contacted would show up as an unmocked request and fail.
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        instance = Aaron(api_keys={"openai": FIXTURE_KEY})
        instance.chat("openai/gpt-4o", "hi")
        assert route.call_count == 1
        assert len(respx.calls) == 1

    def test_importing_the_package_opens_no_connection_and_reads_no_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # conftest turns a real socket into a failure, so a clean import proves it.
        monkeypatch.setenv("OPENAI_API_KEY", FIXTURE_KEY)
        import importlib

        import aaron

        importlib.reload(aaron)
        scan(vars(aaron))


class TestPolicySnapshotIsSafe:
    """The snapshot is embedded in every record."""

    @respx.mock
    def test_a_redactor_pattern_is_not_copied_into_the_snapshot(
        self, records: list[AuditRecord]
    ) -> None:
        from aaron.policy.redaction import RegexRedactor

        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        secret_pattern = r"ACME-INTERNAL-\d{6}"
        instance = Aaron(
            api_keys={"openai": FIXTURE_KEY},
            audit=CallbackSink(records.append),
            policy=Policy(
                redactors=[RegexRedactor(pattern=secret_pattern, replacement="[X]", name="acme")]
            ),
        )
        instance.chat("openai/gpt-4o", "hi")

        rendered = records[0].to_json()
        assert "ACME-INTERNAL" not in rendered
        assert '"acme"' in rendered


class TestRedactionActuallyPrecedesTransmission:
    """A redactor that ran after the request was built would be useless."""

    @respx.mock
    def test_the_redacted_text_is_what_goes_on_the_wire(self) -> None:
        from aaron.policy.redaction import EmailRedactor

        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        instance = Aaron(api_keys={"openai": FIXTURE_KEY}, policy=Policy(redactors=[EmailRedactor()]))
        instance.chat("openai/gpt-4o", Message.user("reach me at person@example.com"))

        assert "person@example.com" not in route.calls[0].request.content.decode()
