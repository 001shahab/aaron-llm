"""Tests for the OpenAI provider: request shape, response parsing, streaming, errors.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from aaron import Aaron, AsyncAaron, Message, Tool
from aaron.errors import (
    AuthenticationError,
    ConfigurationError,
    ContentFilterError,
    ContextLengthExceeded,
    PermissionError,
    ProviderError,
    RateLimitError,
    ServerError,
    ServiceOverloaded,
)
from aaron.stream import DoneEvent, TextEvent, ToolCallEndEvent
from conftest import load, sse

URL = "https://api.openai.com/v1/chat/completions"


@pytest.fixture
def client(keys: dict[str, str]) -> Aaron:
    """A client with retries disabled, so an error surfaces on the first attempt."""
    from aaron.retry import RetryPolicy

    return Aaron(api_keys=keys, retry=RetryPolicy(attempts=1))


class TestRequestShape:
    """What goes on the wire, checked against the documented API."""

    @respx.mock
    def test_a_plain_call(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "What is the capital of Estonia?")

        body = json.loads(route.calls[0].request.content)
        assert body["model"] == "gpt-4o"
        assert body["messages"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of Estonia?"}],
            }
        ]
        assert "stream" not in body

    @respx.mock
    def test_the_credential_goes_in_the_authorization_header(
        self, client: Aaron, api_key: str
    ) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "hi")
        assert route.calls[0].request.headers["authorization"] == f"Bearer {api_key}"

    @respx.mock
    def test_a_system_message_stays_a_message(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", [Message.system("Be brief."), Message.user("hi")])
        body = json.loads(route.calls[0].request.content)
        assert [m["role"] for m in body["messages"]] == ["system", "user"]

    @respx.mock
    def test_sampling_options_are_passed_through(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat(
            "openai/gpt-4o",
            "hi",
            temperature=0.2,
            top_p=0.9,
            max_tokens=100,
            stop=["END"],
            provider_options={"seed": 7},
        )
        body = json.loads(route.calls[0].request.content)
        assert body["temperature"] == 0.2
        assert body["top_p"] == 0.9
        assert body["max_tokens"] == 100
        assert body["stop"] == ["END"]
        assert body["seed"] == 7

    @respx.mock
    def test_a_reasoning_model_uses_max_completion_tokens(self, client: Aaron) -> None:
        # o3 rejects max_tokens and rejects temperature entirely.
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_cached")))
        client.chat("openai/o3", "2+2", max_tokens=500, temperature=0.5)
        body = json.loads(route.calls[0].request.content)
        assert body["max_completion_tokens"] == 500
        assert "max_tokens" not in body
        assert "temperature" not in body

    @respx.mock
    def test_tools_are_wrapped_in_the_function_envelope(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        tool = Tool(
            name="get_weather",
            description="Look it up.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        client.chat("openai/gpt-4o", "weather?", tools=[tool])
        body = json.loads(route.calls[0].request.content)
        assert body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look it up.",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]

    @respx.mock
    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            ("auto", "auto"),
            ("none", "none"),
            ("required", "required"),
            ("get_weather", {"type": "function", "function": {"name": "get_weather"}}),
        ],
    )
    def test_tool_choice_is_translated(self, client: Aaron, choice: str, expected: Any) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        tool = Tool(name="get_weather", description="d", parameters={"type": "object"})
        client.chat("openai/gpt-4o", "hi", tools=[tool], tool_choice=choice)
        assert json.loads(route.calls[0].request.content)["tool_choice"] == expected

    @respx.mock
    def test_an_image_becomes_a_data_url(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        png = bytes.fromhex("89504e470d0a1a0a") + b"body"
        client.chat("openai/gpt-4o", Message.user("what is this", images=[png]))
        parts = json.loads(route.calls[0].request.content)["messages"][0]["content"]
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @respx.mock
    def test_a_tool_result_becomes_a_tool_role_message(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        from aaron.types import ToolCall

        conversation = [
            Message.user("weather?"),
            Message.assistant(
                "", tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "X"})]
            ),
            Message.tool("call_1", "sunny"),
        ]
        client.chat("openai/gpt-4o", conversation)
        messages = json.loads(route.calls[0].request.content)["messages"]
        assert messages[1]["tool_calls"][0]["id"] == "call_1"
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "X"}'
        assert messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}

    @respx.mock
    def test_an_idempotency_key_is_sent(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "hi")
        assert route.calls[0].request.headers["idempotency-key"]

    @respx.mock
    def test_provider_options_reach_the_body_untouched(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "hi", provider_options={"logit_bias": {"123": -100}})
        assert json.loads(route.calls[0].request.content)["logit_bias"] == {"123": -100}

    @respx.mock
    def test_a_caller_option_wins_over_a_computed_one(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        client.chat("openai/gpt-4o", "hi", temperature=0.1, provider_options={"temperature": 0.9})
        assert json.loads(route.calls[0].request.content)["temperature"] == 0.9


class TestResponseParsing:
    """Turning a payload into a Response."""

    @respx.mock
    def test_text_usage_and_ids(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        response = client.chat("openai/gpt-4o", "hi")

        assert response.text == "Tallinn is the capital of Estonia."
        assert response.model == "openai/gpt-4o"
        assert response.resolved_model == "gpt-4o-2024-11-20"
        assert response.stop_reason == "stop"
        assert response.usage.input_tokens == 14
        assert response.usage.output_tokens == 8
        assert response.id == "chatcmpl-Bx7fake000000000000001"
        assert response.latency_ms >= 0
        assert response.cost.usd > 0
        assert response.cost.estimated is False

    @respx.mock
    def test_tool_calls_are_parsed(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_tool_call")))
        response = client.chat("openai/gpt-4o", "weather?")

        assert response.stop_reason == "tool_use"
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Tallinn", "units": "c"}
        assert call.id == "call_fake0000000000000001"

    @respx.mock
    def test_cached_and_reasoning_tokens_are_kept(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_cached")))
        response = client.chat("openai/o3", "2+2")
        assert response.usage.cached_input_tokens == 1024
        assert response.usage.reasoning_tokens == 192

    @respx.mock
    def test_the_raw_payload_is_kept(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        response = client.chat("openai/gpt-4o", "hi")
        assert response.raw["system_fingerprint"] == "fp_0000000000"

    @respx.mock
    def test_malformed_tool_arguments_raise_with_the_raw_text(self, client: Aaron) -> None:
        payload = load("openai_tool_call")
        payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
        respx.post(URL).mock(return_value=httpx.Response(200, json=payload))

        from aaron.errors import ToolArgumentError

        with pytest.raises(ToolArgumentError) as info:
            client.chat("openai/gpt-4o", "weather?")
        assert info.value.raw_arguments == "{not json"
        assert info.value.tool_name == "get_weather"

    @respx.mock
    def test_a_body_that_is_not_json_is_an_error_not_a_crash(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, text="<html>gateway</html>"))
        with pytest.raises(ProviderError, match="JSON"):
            client.chat("openai/gpt-4o", "hi")

    @respx.mock
    def test_a_json_body_with_no_choices_is_an_error(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json={"id": "x", "object": "y"}))
        with pytest.raises(ProviderError, match="no choices"):
            client.chat("openai/gpt-4o", "hi")


class TestErrorMapping:
    """Every status becomes the documented class."""

    @respx.mock
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, AuthenticationError),
            (403, PermissionError),
            (429, RateLimitError),
            (500, ServerError),
            (503, ServiceOverloaded),
        ],
    )
    def test_status_codes(self, client: Aaron, status: int, expected: type[Exception]) -> None:
        respx.post(URL).mock(
            return_value=httpx.Response(status, json={"error": {"message": "no"}})
        )
        with pytest.raises(expected):
            client.chat("openai/gpt-4o", "hi")

    @respx.mock
    def test_context_length_is_detected_from_the_message(self, client: Aaron) -> None:
        respx.post(URL).mock(
            return_value=httpx.Response(400, json=load("openai_error_context"))
        )
        with pytest.raises(ContextLengthExceeded):
            client.chat("openai/gpt-4o", "hi")

    @respx.mock
    def test_content_filter_is_detected(self, client: Aaron) -> None:
        respx.post(URL).mock(
            return_value=httpx.Response(
                400,
                json={"error": {"message": "Your request was rejected by our safety system"}},
            )
        )
        with pytest.raises(ContentFilterError):
            client.chat("openai/gpt-4o", "hi")

    @respx.mock
    def test_a_content_filter_finish_reason_becomes_a_stop_reason(self, client: Aaron) -> None:
        # There is text here, so returning it with the reason beats throwing it away.
        payload = load("openai_chat")
        payload["choices"][0]["finish_reason"] = "content_filter"
        respx.post(URL).mock(return_value=httpx.Response(200, json=payload))
        response = client.chat("openai/gpt-4o", "hi")
        assert response.stop_reason == "content_filter"
        assert response.text

    @respx.mock
    def test_retry_after_is_carried_on_the_error(self, client: Aaron) -> None:
        respx.post(URL).mock(
            return_value=httpx.Response(
                429, headers={"retry-after": "12"}, json=load("openai_error_rate_limit")
            )
        )
        with pytest.raises(RateLimitError) as info:
            client.chat("openai/gpt-4o", "hi")
        assert info.value.retry_after == 12.0

    @respx.mock
    def test_the_error_carries_provider_model_and_status(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(500, json={"error": {"message": "x"}}))
        with pytest.raises(ServerError) as info:
            client.chat("openai/gpt-4o", "hi")
        assert info.value.provider == "openai"
        assert info.value.model == "openai/gpt-4o"
        assert info.value.status_code == 500

    @respx.mock
    def test_a_request_id_is_captured_for_a_support_ticket(self, client: Aaron) -> None:
        respx.post(URL).mock(
            return_value=httpx.Response(
                500, headers={"x-request-id": "req_abc123"}, json={"error": {"message": "x"}}
            )
        )
        with pytest.raises(ServerError) as info:
            client.chat("openai/gpt-4o", "hi")
        assert info.value.request_id == "req_abc123"


class TestStreaming:
    """SSE, sync and async, from the same provider parser."""

    CHUNKS = (
        {
            "id": "chatcmpl-stream1",
            "model": "gpt-4o-2024-11-20",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Tal"}}],
        },
        {
            "id": "chatcmpl-stream1",
            "choices": [{"index": 0, "delta": {"content": "linn"}}],
        },
        {
            "id": "chatcmpl-stream1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
    )

    @respx.mock
    def test_text_events_then_exactly_one_done(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse(*self.CHUNKS)))
        events = list(client.stream("openai/gpt-4o", "capital?"))

        text = [e.text for e in events if isinstance(e, TextEvent)]
        assert text == ["Tal", "linn"]
        done = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done) == 1
        assert isinstance(events[-1], DoneEvent)
        assert done[0].response.text == "Tallinn"
        assert done[0].response.usage.output_tokens == 2
        assert done[0].response.stop_reason == "stop"

    @respx.mock
    def test_stream_options_ask_for_usage(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, text=sse(*self.CHUNKS)))
        list(client.stream("openai/gpt-4o", "hi"))
        body = json.loads(route.calls[0].request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}

    @respx.mock
    def test_streamed_tool_calls_are_assembled_from_index_deltas(self, client: Aaron) -> None:
        chunks = (
            {
                "id": "s2",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                    }
                ],
            },
            {
                "id": "s2",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"city"'}}]
                        },
                    }
                ],
            },
            {
                "id": "s2",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": ':"Tallinn"}'}}]
                        },
                    }
                ],
            },
            {"id": "s2", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        )
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse(*chunks)))
        events = list(client.stream("openai/gpt-4o", "weather?"))

        ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(ends) == 1
        assert ends[0].call.name == "get_weather"
        assert ends[0].call.arguments == {"city": "Tallinn"}
        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.response.tool_calls[0].arguments == {"city": "Tallinn"}

    @respx.mock
    def test_an_error_status_on_a_stream_raises_before_any_event(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(429, json={"error": {"message": "slow"}}))
        with pytest.raises(RateLimitError):
            list(client.stream("openai/gpt-4o", "hi"))

    @respx.mock
    def test_a_stream_can_be_closed_early(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse(*self.CHUNKS)))
        with client.stream("openai/gpt-4o", "hi") as stream:
            first = next(iter(stream))
        assert isinstance(first, TextEvent)

    @respx.mock
    def test_a_malformed_sse_frame_is_skipped(self, client: Aaron) -> None:
        body = "data: {not json\n\n" + sse(*self.CHUNKS)
        respx.post(URL).mock(return_value=httpx.Response(200, text=body))
        events = list(client.stream("openai/gpt-4o", "hi"))
        assert [e.text for e in events if isinstance(e, TextEvent)] == ["Tal", "linn"]

    @respx.mock
    async def test_async_streaming_yields_the_same_events(self, keys: dict[str, str]) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse(*self.CHUNKS)))
        async with AsyncAaron(api_keys=keys) as client:
            events = [event async for event in client.stream("openai/gpt-4o", "capital?")]

        assert [e.text for e in events if isinstance(e, TextEvent)] == ["Tal", "linn"]
        assert isinstance(events[-1], DoneEvent)


class TestAsync:
    """The async client is the same behaviour on the same wire format."""

    @respx.mock
    async def test_chat(self, keys: dict[str, str]) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        async with AsyncAaron(api_keys=keys) as client:
            response = await client.chat("openai/gpt-4o", "hi")
        assert response.text == "Tallinn is the capital of Estonia."

    @respx.mock
    async def test_errors_map_the_same_way(self, keys: dict[str, str]) -> None:
        from aaron.retry import RetryPolicy

        respx.post(URL).mock(return_value=httpx.Response(401, json={"error": {"message": "no"}}))
        async with AsyncAaron(api_keys=keys, retry=RetryPolicy(attempts=1)) as client:
            with pytest.raises(AuthenticationError):
                await client.chat("openai/gpt-4o", "hi")


class TestOpenAICompat:
    """The compatible provider is the same code against someone else's base URL."""

    @respx.mock
    def test_it_needs_an_explicit_base_url(self, keys: dict[str, str]) -> None:
        client = Aaron(api_keys=keys)
        with pytest.raises(ConfigurationError, match="no default endpoint"):
            client.chat("openai_compat/llama-3.1-70b", "hi")

    @respx.mock
    def test_it_posts_to_the_given_endpoint(self, keys: dict[str, str]) -> None:
        route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        client = Aaron(api_keys=keys, base_urls={"openai_compat": "https://api.groq.com/openai/v1"})
        response = client.chat("openai_compat/llama-3.1-70b", "hi")
        assert response.text
        assert json.loads(route.calls[0].request.content)["model"] == "llama-3.1-70b"

    @respx.mock
    def test_it_works_without_any_credential(self) -> None:
        # A self hosted vLLM or llama.cpp server usually has no authentication.
        route = respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        client = Aaron(base_urls={"openai_compat": "http://localhost:8000/v1"})
        client.chat("openai_compat/local-model", "hi")
        assert "authorization" not in route.calls[0].request.headers
