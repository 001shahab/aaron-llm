"""Tests for the client itself: configuration, aliases, retries, batch and extract.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import BaseModel, Field

import aaron
from aaron import Aaron, AsyncAaron, Message
from aaron.errors import (
    AaronError,
    ConfigurationError,
    InvalidRequest,
    RateLimitError,
    ServerError,
    TimeoutError,
    TransportError,
)
from aaron.retry import RetryPolicy
from conftest import load

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/chat"


@pytest.fixture
def client() -> Aaron:
    """A client with a key and no retries."""
    return Aaron(api_keys={"openai": "k"}, retry=RetryPolicy(attempts=1))


class TestModelResolution:
    """Model strings, aliases and defaults."""

    def test_a_model_must_be_provider_slash_model(self, client: Aaron) -> None:
        with pytest.raises(InvalidRequest, match="provider/model"):
            client.chat("gpt-4o", "hi")

    def test_only_the_first_slash_splits(self, client: Aaron) -> None:
        # A model id may itself contain slashes, as on OpenRouter or Hugging Face.
        request = client.build("openai_compat/meta-llama/Llama-3.1-70B", "hi", base_url="http://x")
        assert request.provider == "openai_compat"
        assert request.model_name == "meta-llama/Llama-3.1-70B"

    def test_an_unknown_provider_is_named(self, client: Aaron) -> None:
        from aaron.errors import UnknownProvider

        with pytest.raises(UnknownProvider):
            client.chat("openai2/gpt-4o", "hi")

    @respx.mock
    def test_an_alias_resolves(self, client: Aaron) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.alias("fast", "openai/gpt-4o-mini")
        assert client.resolve_model("fast") == "openai/gpt-4o-mini"

    def test_an_alias_can_point_at_another_alias(self, client: Aaron) -> None:
        client.alias("a", "b")
        client.alias("b", "openai/gpt-4o")
        assert client.resolve_model("a") == "openai/gpt-4o"

    def test_an_alias_cycle_is_reported_rather_than_hanging(self, client: Aaron) -> None:
        client.alias("a", "b")
        client.alias("b", "a")
        with pytest.raises(ConfigurationError, match="cycle"):
            client.resolve_model("a")

    def test_an_alias_may_not_contain_a_slash(self, client: Aaron) -> None:
        # Otherwise an alias could shadow a real provider/model string.
        with pytest.raises(ConfigurationError, match="slash"):
            client.alias("openai/gpt-4o", "openai/gpt-4o-mini")

    @respx.mock
    def test_a_default_model_is_used_when_none_is_given(self) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        instance = Aaron(api_keys={"openai": "k"}, default_model="openai/gpt-4o")
        assert instance.chat(messages="hi").text

    def test_no_model_and_no_default_says_what_to_do(self, client: Aaron) -> None:
        with pytest.raises(ConfigurationError, match="default_model"):
            client.chat(messages="hi")


class TestPromptForms:
    """A prompt may be a string, a message, or a sequence."""

    @respx.mock
    def test_a_string(self, client: Aaron) -> None:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        client.chat("openai/gpt-4o", "hi")
        assert json.loads(route.calls[0].request.content)["messages"][0]["role"] == "user"

    @respx.mock
    def test_a_single_message(self, client: Aaron) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        assert client.chat("openai/gpt-4o", Message.user("hi")).text

    @respx.mock
    def test_a_list_of_messages(self, client: Aaron) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        assert client.chat("openai/gpt-4o", [Message.system("be brief"), Message.user("hi")]).text

    def test_an_empty_prompt_is_refused(self, client: Aaron) -> None:
        with pytest.raises(InvalidRequest):
            client.chat("openai/gpt-4o", [])


class TestValidation:
    """Bad arguments are caught locally, before a paid call."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"temperature": 3.0}, "temperature"),
            ({"temperature": -1.0}, "temperature"),
            ({"top_p": 1.5}, "top_p"),
            ({"max_tokens": 0}, "max_tokens"),
            ({"max_tokens": -5}, "max_tokens"),
        ],
    )
    def test_out_of_range_values(self, client: Aaron, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(InvalidRequest, match=match):
            client.chat("openai/gpt-4o", "hi", **kwargs)

    def test_two_tools_with_the_same_name(self, client: Aaron) -> None:
        from aaron import Tool

        tool = Tool(name="f", description="d", parameters={"type": "object"})
        with pytest.raises(InvalidRequest, match="duplicate"):
            client.chat("openai/gpt-4o", "hi", tools=[tool, tool])

    def test_a_tool_result_without_a_preceding_call_is_refused(self, client: Aaron) -> None:
        with pytest.raises(InvalidRequest, match="tool"):
            client.chat("openai/gpt-4o", [Message.user("hi"), Message.tool("nope", "result")])

    def test_a_json_schema_cannot_be_combined_with_tools(self, client: Aaron) -> None:
        from aaron import Tool

        class Doc(BaseModel):
            """A doc."""

            a: str

        tool = Tool(name="f", description="d", parameters={"type": "object"})
        with pytest.raises(InvalidRequest):
            client.extract("openai/gpt-4o", "hi", schema=Doc, tools=[tool])


class TestRetries:
    """Only failures a later attempt could survive are retried."""

    @respx.mock
    def test_a_server_error_is_retried_and_then_succeeds(self) -> None:
        route = respx.post(OPENAI_URL).mock(
            side_effect=[
                httpx.Response(500, json={"error": {"message": "boom"}}),
                httpx.Response(200, json=load("openai_chat")),
            ]
        )
        instance = Aaron(
            api_keys={"openai": "k"}, retry=RetryPolicy(attempts=3, initial_delay=0.0)
        )
        assert instance.chat("openai/gpt-4o", "hi").text
        assert route.call_count == 2

    @respx.mock
    def test_a_rate_limit_is_retried(self) -> None:
        route = respx.post(OPENAI_URL).mock(
            side_effect=[
                httpx.Response(429, json={"error": {}}),
                httpx.Response(200, json=load("openai_chat")),
            ]
        )
        instance = Aaron(
            api_keys={"openai": "k"}, retry=RetryPolicy(attempts=2, initial_delay=0.0)
        )
        instance.chat("openai/gpt-4o", "hi")
        assert route.call_count == 2

    @respx.mock
    def test_an_authentication_error_is_not_retried(self) -> None:
        route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(401, json={"error": {}}))
        instance = Aaron(
            api_keys={"openai": "k"}, retry=RetryPolicy(attempts=5, initial_delay=0.0)
        )
        with pytest.raises(AaronError):
            instance.chat("openai/gpt-4o", "hi")
        assert route.call_count == 1, "a bad key will still be bad on the fourth attempt"

    @respx.mock
    def test_an_invalid_request_is_not_retried(self) -> None:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(400, json={"error": {"message": "bad parameter"}})
        )
        instance = Aaron(
            api_keys={"openai": "k"}, retry=RetryPolicy(attempts=5, initial_delay=0.0)
        )
        with pytest.raises(AaronError):
            instance.chat("openai/gpt-4o", "hi")
        assert route.call_count == 1

    @respx.mock
    def test_attempts_are_bounded(self) -> None:
        route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(500, json={"error": {}}))
        instance = Aaron(
            api_keys={"openai": "k"}, retry=RetryPolicy(attempts=3, initial_delay=0.0)
        )
        with pytest.raises(ServerError):
            instance.chat("openai/gpt-4o", "hi")
        assert route.call_count == 3

    @respx.mock
    def test_a_stream_that_has_yielded_text_is_not_retried(self) -> None:
        from conftest import sse

        body = sse({"id": "s", "choices": [{"index": 0, "delta": {"content": "partial"}}]})
        route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, text=body))
        instance = Aaron(
            api_keys={"openai": "k"}, retry=RetryPolicy(attempts=3, initial_delay=0.0)
        )
        list(instance.stream("openai/gpt-4o", "hi"))
        assert route.call_count == 1


class TestRetryPolicy:
    """The backoff arithmetic, without sleeping."""

    def test_one_attempt_means_no_retry(self) -> None:
        assert RetryPolicy(attempts=1).should_retry(ServerError("x"), attempt=1) is False

    def test_a_retryable_error_within_the_budget_is_retried(self) -> None:
        assert RetryPolicy(attempts=3).should_retry(ServerError("x"), attempt=1) is True

    def test_a_non_retryable_error_is_never_retried(self) -> None:
        assert RetryPolicy(attempts=3).should_retry(InvalidRequest("x"), attempt=1) is False

    def test_a_timeout_is_retryable(self) -> None:
        assert RetryPolicy(attempts=3).should_retry(TimeoutError("x"), attempt=1) is True

    def test_the_delay_grows_and_is_capped(self) -> None:
        policy = RetryPolicy(initial_delay=1.0, factor=2.0, max_delay=4.0)
        # Full jitter means a uniform draw over the window, so check the ceiling.
        assert all(0.0 <= policy.delay_for(attempt) <= 4.0 for attempt in range(1, 8))

    def test_retry_after_is_honoured_when_it_is_longer(self) -> None:
        policy = RetryPolicy(initial_delay=0.01, max_delay=60.0)
        error = RateLimitError("slow down", retry_after=5.0)
        assert policy.delay_for(1, error) >= 5.0

    def test_retry_after_is_still_capped_by_max_delay(self) -> None:
        policy = RetryPolicy(max_delay=2.0)
        error = RateLimitError("slow down", retry_after=3600.0)
        assert policy.delay_for(1, error) <= 2.0

    def test_it_can_be_disabled_entirely(self) -> None:
        policy = RetryPolicy(respect_retry_after=False, initial_delay=0.0, max_delay=0.0)
        assert policy.delay_for(1, RateLimitError("x", retry_after=99.0)) == 0.0


class TestTimeouts:
    """A timeout is a typed error, not an httpx exception."""

    @respx.mock
    def test_a_read_timeout_becomes_our_timeout_error(self, client: Aaron) -> None:
        respx.post(OPENAI_URL).mock(side_effect=httpx.ReadTimeout("too slow"))
        with pytest.raises(TimeoutError):
            client.chat("openai/gpt-4o", "hi")

    @respx.mock
    def test_a_connect_error_becomes_a_transport_error(self, client: Aaron) -> None:
        respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("no route"))
        with pytest.raises(TransportError):
            client.chat("openai/gpt-4o", "hi")

    def test_a_per_call_timeout_overrides_the_client_default(self, client: Aaron) -> None:
        request = client.build("openai/gpt-4o", "hi", timeout=2.5)
        assert request.timeout == 2.5

    def test_the_client_default_is_used_otherwise(self) -> None:
        instance = Aaron(api_keys={"openai": "k"}, timeout=12.0)
        assert instance.transport_config.timeout == 12.0


class TestBatch:
    """Concurrency with per item error isolation."""

    @respx.mock
    async def test_every_request_gets_a_result_in_order(self) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        async with AsyncAaron(api_keys={"openai": "k"}) as client:
            results = await client.batch(
                [
                    {"model": "openai/gpt-4o", "messages": "one"},
                    {"model": "openai/gpt-4o", "messages": "two"},
                    {"model": "openai/gpt-4o", "messages": "three"},
                ]
            )
        assert len(results) == 3
        assert all(not isinstance(item, BaseException) for item in results)

    @respx.mock
    async def test_one_failure_does_not_lose_the_others(self) -> None:
        respx.post(OPENAI_URL).mock(
            side_effect=[
                httpx.Response(200, json=load("openai_chat")),
                httpx.Response(401, json={"error": {"message": "bad key"}}),
                httpx.Response(200, json=load("openai_chat")),
            ]
        )
        async with AsyncAaron(
            api_keys={"openai": "k"}, retry=RetryPolicy(attempts=1)
        ) as client:
            results = await client.batch(
                [{"model": "openai/gpt-4o", "messages": str(i)} for i in range(3)]
            )

        assert not isinstance(results[0], AaronError)
        assert isinstance(results[1], AaronError)
        assert not isinstance(results[2], AaronError)

    @respx.mock
    async def test_concurrency_is_bounded(self) -> None:
        in_flight = 0
        peak = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return httpx.Response(200, json=load("openai_chat"))

        respx.post(OPENAI_URL).mock(side_effect=handler)
        async with AsyncAaron(api_keys={"openai": "k"}) as client:
            await client.batch(
                [{"model": "openai/gpt-4o", "messages": str(i)} for i in range(12)],
                concurrency=3,
            )
        assert peak <= 3

    @respx.mock
    async def test_an_empty_batch_is_an_empty_list(self) -> None:
        async with AsyncAaron(api_keys={"openai": "k"}) as client:
            assert await client.batch([]) == []


class TestExtract:
    """Structured output, validated locally whatever the provider claims."""

    class City(BaseModel):
        """A city and its country."""

        name: str = Field(description="The city name.")
        country: str
        population: int | None = None

    @respx.mock
    def test_a_valid_document_is_returned(self, client: Aaron) -> None:
        payload = load("openai_chat")
        payload["choices"][0]["message"]["content"] = '{"name": "Tallinn", "country": "Estonia"}'
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=payload))

        city = client.extract("openai/gpt-4o", "capital of Estonia?", schema=self.City)
        assert isinstance(city, self.City)
        assert city.name == "Tallinn"

    @respx.mock
    def test_the_schema_is_sent_to_the_provider(self, client: Aaron) -> None:
        payload = load("openai_chat")
        payload["choices"][0]["message"]["content"] = '{"name": "T", "country": "E"}'
        route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=payload))
        client.extract("openai/gpt-4o", "capital?", schema=self.City)

        body = json.loads(route.calls[0].request.content)
        assert body["response_format"]["type"] == "json_schema"
        assert "$ref" not in json.dumps(body["response_format"])

    @respx.mock
    def test_a_fenced_code_block_is_unwrapped(self, client: Aaron) -> None:
        # Models add ```json fences even when told not to.
        payload = load("openai_chat")
        payload["choices"][0]["message"]["content"] = (
            '```json\n{"name": "Tallinn", "country": "Estonia"}\n```'
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=payload))
        assert client.extract("openai/gpt-4o", "capital?", schema=self.City).name == "Tallinn"

    @respx.mock
    def test_an_invalid_document_is_shown_back_to_the_model(self, client: Aaron) -> None:
        bad = load("openai_chat")
        bad["choices"][0]["message"]["content"] = '{"name": "Tallinn"}'
        good = load("openai_chat")
        good["choices"][0]["message"]["content"] = '{"name": "Tallinn", "country": "Estonia"}'
        route = respx.post(OPENAI_URL).mock(
            side_effect=[httpx.Response(200, json=bad), httpx.Response(200, json=good)]
        )

        city = client.extract("openai/gpt-4o", "capital?", schema=self.City, max_repairs=1)
        assert city.country == "Estonia"
        assert route.call_count == 2
        # The repair turn must quote the validation error, or the model cannot fix it.
        repair = json.loads(route.calls[1].request.content)
        assert "country" in json.dumps(repair["messages"])

    @respx.mock
    def test_repairs_are_bounded_and_the_error_explains(self, client: Aaron) -> None:
        bad = load("openai_chat")
        bad["choices"][0]["message"]["content"] = "{}"
        route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=bad))

        with pytest.raises(InvalidRequest) as info:
            client.extract("openai/gpt-4o", "capital?", schema=self.City, max_repairs=2)
        assert route.call_count == 3
        assert "country" in str(info.value)

    @respx.mock
    def test_zero_repairs_fails_on_the_first_invalid_document(self, client: Aaron) -> None:
        bad = load("openai_chat")
        bad["choices"][0]["message"]["content"] = "not json at all"
        route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=bad))

        with pytest.raises(InvalidRequest):
            client.extract("openai/gpt-4o", "capital?", schema=self.City, max_repairs=0)
        assert route.call_count == 1

    @respx.mock
    async def test_the_async_client_extracts_too(self) -> None:
        payload = load("openai_chat")
        payload["choices"][0]["message"]["content"] = '{"name": "Tallinn", "country": "Estonia"}'
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=payload))
        async with AsyncAaron(api_keys={"openai": "k"}) as client:
            city = await client.extract("openai/gpt-4o", "capital?", schema=self.City)
        assert city.name == "Tallinn"

    @respx.mock
    def test_a_provider_without_native_schema_support_still_works(self) -> None:
        # ollama gets the schema in `format`, and the result is validated locally.
        payload = load("ollama_chat")
        payload["message"]["content"] = '{"name": "Tallinn", "country": "Estonia"}'
        respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=payload))
        assert Aaron().extract("ollama/llama3.1", "capital?", schema=self.City).name == "Tallinn"


class TestConfigurationResolution:
    """Section 14: explicit argument, then environment, then file, then default."""

    @respx.mock
    def test_an_explicit_key_beats_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        Aaron(api_keys={"openai": "explicit"}).chat("openai/gpt-4o", "hi")
        assert route.calls[0].request.headers["authorization"] == "Bearer explicit"

    @respx.mock
    def test_the_environment_is_used_when_nothing_is_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        Aaron().chat("openai/gpt-4o", "hi")
        assert route.calls[0].request.headers["authorization"] == "Bearer from-env"

    def test_a_callable_key_is_resolved_at_call_time(self) -> None:
        calls: list[int] = []

        def rotate() -> str:
            calls.append(1)
            return f"key-{len(calls)}"

        instance = Aaron(api_keys={"openai": rotate})
        assert instance.dry_run("openai/gpt-4o", "hi")
        assert instance.dry_run("openai/gpt-4o", "hi")
        assert len(calls) == 2, "a rotating credential must be re read for every call"

    def test_a_callable_that_returns_nothing_is_an_error(self) -> None:
        instance = Aaron(api_keys={"openai": lambda: ""})
        from aaron.errors import MissingAPIKey

        with pytest.raises(MissingAPIKey, match="empty"):
            instance.dry_run("openai/gpt-4o", "hi")

    def test_a_toml_config_file_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "aaron.toml"
        path.write_text(
            "\n".join(
                [
                    'default_model = "openai/gpt-4o-mini"',
                    "timeout = 42.0",
                    "[aliases]",
                    'fast = "openai/gpt-4o-mini"',
                    "[base_urls]",
                    'openai_compat = "https://gateway.example.com/v1"',
                ]
            ),
            encoding="utf-8",
        )
        instance = Aaron(config_file=path)

        assert instance.default_model == "openai/gpt-4o-mini"
        assert instance.transport_config.timeout == 42.0
        assert instance.resolve_model("fast") == "openai/gpt-4o-mini"

    def test_an_explicit_argument_beats_the_config_file(self, tmp_path: Path) -> None:
        path = tmp_path / "aaron.toml"
        path.write_text("timeout = 42.0\n", encoding="utf-8")
        assert Aaron(config_file=path, timeout=7.0).transport_config.timeout == 7.0

    def test_a_missing_config_file_is_an_error_when_named_explicitly(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigurationError):
            Aaron(config_file=tmp_path / "nope.toml")

    def test_the_default_model_can_come_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AARON_DEFAULT_MODEL", "ollama/llama3.1")
        assert Aaron().default_model == "ollama/llama3.1"

    def test_a_base_url_override_is_used(self) -> None:
        instance = Aaron(api_keys={"openai": "k"}, base_urls={"openai": "https://proxy.internal/v1"})
        assert instance.dry_run("openai/gpt-4o", "hi").url.startswith("https://proxy.internal/v1")

    def test_a_trailing_slash_on_a_base_url_does_not_double_up(self) -> None:
        instance = Aaron(api_keys={"openai": "k"}, base_urls={"openai": "https://x.test/v1/"})
        assert instance.dry_run("openai/gpt-4o", "hi").url == "https://x.test/v1/chat/completions"


class TestLifecycle:
    """Connection pools are owned and released."""

    @respx.mock
    def test_the_sync_client_works_as_a_context_manager(self) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        with Aaron(api_keys={"openai": "k"}) as client:
            assert client.chat("openai/gpt-4o", "hi").text

    @respx.mock
    async def test_the_async_client_works_as_a_context_manager(self) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        async with AsyncAaron(api_keys={"openai": "k"}) as client:
            assert (await client.chat("openai/gpt-4o", "hi")).text

    def test_close_is_idempotent(self) -> None:
        client = Aaron(api_keys={"openai": "k"})
        client.close()
        client.close()

    def test_an_httpx_client_can_be_supplied(self) -> None:
        # For a caller who already has a configured pool, proxy or transport.
        custom = httpx.Client(headers={"x-tenant": "acme"})
        instance = Aaron(api_keys={"openai": "k"}, http_client=custom)
        assert instance.http is custom
        instance.close()


class TestModuleLevelFunctions:
    """The convenience functions share one lazily built client."""

    @respx.mock
    def test_chat_works_without_constructing_anything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        aaron.set_default_client(None)
        assert aaron.chat("openai/gpt-4o", "hi").text

    def test_the_default_client_is_created_once(self) -> None:
        aaron.set_default_client(None)
        assert aaron.default_client() is aaron.default_client()

    def test_the_default_client_can_be_replaced(self) -> None:
        replacement = Aaron(api_keys={"openai": "k"})
        aaron.set_default_client(replacement)
        assert aaron.default_client() is replacement
        aaron.set_default_client(None)

    def test_nothing_is_constructed_at_import_time(self) -> None:
        # A module level client would open a pool and read the environment on import.
        aaron.set_default_client(None)
        assert aaron._default_client is None
