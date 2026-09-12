"""Tests for the Anthropic provider.

Anthropic differs from OpenAI in four ways that matter, and each has a test here:
system is a top level field, consecutive same role turns must be merged, tool results
are content blocks inside a user turn, and structured output goes through a forced
tool call because there is no response_format.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import BaseModel

from aaron import Aaron, AsyncAaron, Message, Tool
from aaron.errors import AuthenticationError, ProviderError, ServiceOverloaded
from aaron.retry import RetryPolicy
from aaron.stream import DoneEvent, TextEvent, ThinkingEvent, ToolCallEndEvent
from aaron.types import ToolCall
from conftest import load, sse_typed

URL = "https://api.anthropic.com/v1/messages"


@pytest.fixture
def client(keys: dict[str, str]) -> Aaron:
    """A client with retries off, so one mocked failure produces one error."""
    return Aaron(api_keys=keys, retry=RetryPolicy(attempts=1))


class TestRequestShape:
    """The Messages API wire format."""

    @respx.mock
    def test_the_credential_uses_x_api_key_not_bearer(self, client: Aaron, api_key: str) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_chat")))
        client.chat("anthropic/claude-sonnet-4-5", "hi")
        request = route.calls[0].request
        assert request.headers["x-api-key"] == api_key
        assert "authorization" not in request.headers
        assert request.headers["anthropic-version"] == "2023-06-01"

    @respx.mock
    def test_system_becomes_a_top_level_field(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_chat")))
        client.chat(
            "anthropic/claude-sonnet-4-5", [Message.system("Be brief."), Message.user("hi")]
        )
        body = json.loads(route.calls[0].request.content)
        assert body["system"] == "Be brief."
        assert [m["role"] for m in body["messages"]] == ["user"]

    @respx.mock
    def test_several_system_messages_are_joined(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_chat")))
        client.chat(
            "anthropic/claude-sonnet-4-5",
            [Message.system("Be brief."), Message.system("Be polite."), Message.user("hi")],
        )
        assert json.loads(route.calls[0].request.content)["system"] == "Be brief.\n\nBe polite."

    @respx.mock
    def test_max_tokens_is_always_sent_because_the_api_requires_it(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_chat")))
        client.chat("anthropic/claude-sonnet-4-5", "hi")
        assert json.loads(route.calls[0].request.content)["max_tokens"] == 4096

    @respx.mock
    def test_a_caller_max_tokens_wins(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_chat")))
        client.chat("anthropic/claude-sonnet-4-5", "hi", max_tokens=100)
        assert json.loads(route.calls[0].request.content)["max_tokens"] == 100

    @respx.mock
    def test_consecutive_user_turns_are_merged(self, client: Aaron) -> None:
        # The API rejects two user turns in a row, so the provider merges them.
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_chat")))
        client.chat("anthropic/claude-sonnet-4-5", [Message.user("first"), Message.user("second")])
        messages = json.loads(route.calls[0].request.content)["messages"]
        assert len(messages) == 1
        assert [part["text"] for part in messages[0]["content"]] == ["first", "second"]

    @respx.mock
    def test_a_tool_result_is_a_block_in_a_user_turn(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_chat")))
        conversation = [
            Message.user("weather?"),
            Message.assistant(
                "", tool_calls=[ToolCall(id="toolu_1", name="get_weather", arguments={"city": "X"})]
            ),
            Message.tool("toolu_1", "sunny"),
        ]
        client.chat("anthropic/claude-sonnet-4-5", conversation)
        messages = json.loads(route.calls[0].request.content)["messages"]

        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"][0] == {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "X"},
        }
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "sunny",
        }

    @respx.mock
    def test_tools_use_input_schema_not_parameters(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_chat")))
        tool = Tool(
            name="get_weather",
            description="Look it up.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        client.chat("anthropic/claude-sonnet-4-5", "weather?", tools=[tool])
        assert json.loads(route.calls[0].request.content)["tools"] == [
            {
                "name": "get_weather",
                "description": "Look it up.",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]

    @respx.mock
    def test_an_image_is_a_base64_source_block(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_chat")))
        png = bytes.fromhex("89504e470d0a1a0a") + b"body"
        client.chat("anthropic/claude-sonnet-4-5", Message.user("what is this", images=[png]))
        part = json.loads(route.calls[0].request.content)["messages"][0]["content"][1]
        assert part["type"] == "image"
        assert part["source"]["type"] == "base64"
        assert part["source"]["media_type"] == "image/png"

    @respx.mock
    def test_the_pdf_beta_header_is_only_sent_when_a_pdf_is_attached(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_chat")))
        client.chat("anthropic/claude-sonnet-4-5", "no attachment here")
        assert "anthropic-beta" not in route.calls[0].request.headers

        client.chat(
            "anthropic/claude-sonnet-4-5", Message.user("summarise", documents=[b"%PDF-1.7 body"])
        )
        assert route.calls[1].request.headers["anthropic-beta"] == "pdfs-2024-09-25"


class TestResponseParsing:
    """Content blocks become one normalised message."""

    @respx.mock
    def test_text_and_usage(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_chat")))
        response = client.chat("anthropic/claude-sonnet-4-5", "hi")
        assert response.text == "Tallinn is the capital of Estonia."
        assert response.stop_reason == "stop"
        assert response.usage.input_tokens == 15
        assert response.usage.output_tokens == 9
        assert response.resolved_model == "claude-sonnet-4-5-20250929"

    @respx.mock
    def test_a_tool_use_block_becomes_a_tool_call(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_tool_call")))
        response = client.chat("anthropic/claude-sonnet-4-5", "weather?")
        assert response.stop_reason == "tool_use"
        assert response.text == "Let me look that up."
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"city": "Tallinn", "units": "c"}

    @respx.mock
    def test_a_thinking_block_is_kept_out_of_the_text_but_on_the_raw(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("anthropic_thinking")))
        response = client.chat("anthropic/claude-opus-4-1", "capital?")
        assert response.text == "Tallinn."
        assert "The user asked for a capital city." in json.dumps(response.raw)

    @respx.mock
    def test_a_payload_with_no_content_array_is_a_provider_error(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json={"id": "m", "type": "message"}))
        with pytest.raises(ProviderError, match="content array"):
            client.chat("anthropic/claude-sonnet-4-5", "hi")

    @respx.mock
    def test_overloaded_maps_to_its_own_class(self, client: Aaron) -> None:
        respx.post(URL).mock(
            return_value=httpx.Response(529, json=load("anthropic_error_overloaded"))
        )
        with pytest.raises(ServiceOverloaded):
            client.chat("anthropic/claude-sonnet-4-5", "hi")

    @respx.mock
    def test_a_bad_key_maps_to_authentication(self, client: Aaron) -> None:
        respx.post(URL).mock(
            return_value=httpx.Response(
                401,
                json={
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "bad"},
                },
            )
        )
        with pytest.raises(AuthenticationError):
            client.chat("anthropic/claude-sonnet-4-5", "hi")


class TestStreaming:
    """Anthropic streams named events, not anonymous chunks."""

    EVENTS = (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stream1",
                    "model": "claude-sonnet-4-5-20250929",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Tal"},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "linn"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )

    @respx.mock
    def test_text_deltas_and_one_done(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse_typed(*self.EVENTS)))
        events = list(client.stream("anthropic/claude-sonnet-4-5", "capital?"))

        assert [e.text for e in events if isinstance(e, TextEvent)] == ["Tal", "linn"]
        done = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done) == 1
        assert done[0].response.text == "Tallinn"
        assert done[0].response.usage.input_tokens == 12
        assert done[0].response.usage.output_tokens == 3
        assert done[0].response.stop_reason == "stop"

    @respx.mock
    def test_a_thinking_delta_is_its_own_event(self, client: Aaron) -> None:
        events = (
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {"id": "m", "model": "claude-opus-4-1", "usage": {}},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "hmm"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "Tallinn"},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        )
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse_typed(*events)))
        received = list(client.stream("anthropic/claude-opus-4-1", "capital?"))

        assert [e.text for e in received if isinstance(e, ThinkingEvent)] == ["hmm"]
        assert [e.text for e in received if isinstance(e, TextEvent)] == ["Tallinn"]

    @respx.mock
    def test_streamed_tool_json_is_accumulated(self, client: Aaron) -> None:
        events = (
            (
                "message_start",
                {"type": "message_start", "message": {"id": "m", "model": "c", "usage": {}}},
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city"'},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": ':"Tallinn"}'},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_stop", {"type": "message_stop"}),
        )
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse_typed(*events)))
        received = list(client.stream("anthropic/claude-sonnet-4-5", "weather?"))

        ends = [e for e in received if isinstance(e, ToolCallEndEvent)]
        assert ends[0].call.arguments == {"city": "Tallinn"}
        assert ends[0].call.id == "toolu_1"

    @respx.mock
    async def test_async_gets_the_same_events(self, keys: dict[str, str]) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse_typed(*self.EVENTS)))
        async with AsyncAaron(api_keys=keys) as client:
            events = [e async for e in client.stream("anthropic/claude-sonnet-4-5", "hi")]
        assert [e.text for e in events if isinstance(e, TextEvent)] == ["Tal", "linn"]
        assert isinstance(events[-1], DoneEvent)


class TestStructuredOutput:
    """There is no response_format, so extract forces a single tool call."""

    class City(BaseModel):
        """A city."""

        name: str
        country: str

    @respx.mock
    def test_extract_forces_a_tool_and_reads_the_arguments(self, client: Aaron) -> None:
        payload = load("anthropic_tool_call")
        payload["content"] = [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "respond_with_json",
                "input": {"name": "Tallinn", "country": "Estonia"},
            }
        ]
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=payload))
        city = client.extract(
            "anthropic/claude-sonnet-4-5", "capital of Estonia?", schema=self.City
        )

        assert city.name == "Tallinn"
        assert city.country == "Estonia"
        body = json.loads(route.calls[0].request.content)
        assert body["tool_choice"]["type"] == "tool"
        schema = body["tools"][0]["input_schema"]
        assert schema["properties"]["country"]["type"] == "string"
        assert "$ref" not in json.dumps(schema)
        assert "response_format" not in body
