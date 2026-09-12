"""Tests for the Google Gemini provider.

Gemini renames almost everything: contents instead of messages, parts instead of
content, model instead of assistant, and the model goes in the URL path. The key goes
in a header and never in the query string, because query strings end up in access logs.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import BaseModel

from aaron import Aaron, AsyncAaron, Message, Tool
from aaron.errors import ContentFilterError, ProviderError
from aaron.retry import RetryPolicy
from aaron.stream import DoneEvent, TextEvent
from aaron.types import ToolCall
from conftest import load, sse

BASE = "https://generativelanguage.googleapis.com/v1beta"
URL = f"{BASE}/models/gemini-2.5-flash:generateContent"
STREAM_URL = f"{BASE}/models/gemini-2.5-flash:streamGenerateContent"


@pytest.fixture
def client(keys: dict[str, str]) -> Aaron:
    """A client with retries off."""
    return Aaron(api_keys=keys, retry=RetryPolicy(attempts=1))


class TestRequestShape:
    """The generateContent wire format."""

    @respx.mock
    def test_the_model_is_in_the_path_and_the_key_is_in_a_header(
        self, client: Aaron, api_key: str
    ) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_chat")))
        client.chat("google/gemini-2.5-flash", "hi")
        request = route.calls[0].request

        assert request.headers["x-goog-api-key"] == api_key
        assert api_key not in str(request.url)
        assert "key=" not in str(request.url)

    @respx.mock
    def test_messages_become_contents_with_parts(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_chat")))
        client.chat("google/gemini-2.5-flash", "hi")
        body = json.loads(route.calls[0].request.content)
        assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]

    @respx.mock
    def test_the_assistant_role_is_called_model(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_chat")))
        client.chat(
            "google/gemini-2.5-flash", [Message.user("hi"), Message.assistant("hello"), Message.user("again")]
        )
        roles = [entry["role"] for entry in json.loads(route.calls[0].request.content)["contents"]]
        assert roles == ["user", "model", "user"]

    @respx.mock
    def test_system_becomes_system_instruction(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_chat")))
        client.chat("google/gemini-2.5-flash", [Message.system("Be brief."), Message.user("hi")])
        body = json.loads(route.calls[0].request.content)
        assert body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
        assert all(entry["role"] != "system" for entry in body["contents"])

    @respx.mock
    def test_sampling_options_go_into_generation_config(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_chat")))
        client.chat(
            "google/gemini-2.5-flash",
            "hi",
            temperature=0.3,
            top_p=0.8,
            max_tokens=256,
            stop=["END"],
        )
        config = json.loads(route.calls[0].request.content)["generationConfig"]
        assert config["temperature"] == 0.3
        assert config["topP"] == 0.8
        assert config["maxOutputTokens"] == 256
        assert config["stopSequences"] == ["END"]

    @respx.mock
    def test_tools_become_function_declarations(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_chat")))
        tool = Tool(
            name="get_weather",
            description="Look it up.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        client.chat("google/gemini-2.5-flash", "weather?", tools=[tool])
        body = json.loads(route.calls[0].request.content)
        declarations = body["tools"][0]["functionDeclarations"]
        assert declarations[0]["name"] == "get_weather"
        assert declarations[0]["parameters"]["properties"]["city"] == {"type": "string"}

    @respx.mock
    def test_schema_keywords_gemini_rejects_are_stripped(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_chat")))
        tool = Tool(
            name="f",
            description="d",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "title": "F",
                "properties": {"a": {"type": "string", "title": "A", "default": "x"}},
            },
        )
        client.chat("google/gemini-2.5-flash", "hi", tools=[tool])
        rendered = json.dumps(json.loads(route.calls[0].request.content)["tools"])
        assert "additionalProperties" not in rendered
        assert "title" not in rendered

    @respx.mock
    def test_a_tool_result_becomes_a_function_response(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_chat")))
        conversation = [
            Message.user("weather?"),
            Message.assistant(
                "", tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "X"})]
            ),
            Message.tool("c1", "sunny"),
        ]
        client.chat("google/gemini-2.5-flash", conversation)
        contents = json.loads(route.calls[0].request.content)["contents"]
        assert contents[1]["parts"][0]["functionCall"] == {
            "name": "get_weather",
            "args": {"city": "X"},
        }
        assert contents[2]["parts"][0]["functionResponse"]["name"] == "get_weather"

    @respx.mock
    def test_tool_choice_becomes_a_tool_config_mode(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_chat")))
        tool = Tool(name="f", description="d", parameters={"type": "object"})
        client.chat("google/gemini-2.5-flash", "hi", tools=[tool], tool_choice="required")
        config = json.loads(route.calls[0].request.content)["toolConfig"]
        assert config["functionCallingConfig"]["mode"] == "ANY"

    @respx.mock
    def test_an_image_becomes_inline_data(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_chat")))
        png = bytes.fromhex("89504e470d0a1a0a") + b"body"
        client.chat("google/gemini-2.5-flash", Message.user("what is this", images=[png]))
        parts = json.loads(route.calls[0].request.content)["contents"][0]["parts"]
        assert parts[1]["inlineData"]["mimeType"] == "image/png"
        assert parts[1]["inlineData"]["data"]


class TestResponseParsing:
    """Candidates become one normalised response."""

    @respx.mock
    def test_text_and_usage(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_chat")))
        response = client.chat("google/gemini-2.5-flash", "hi")
        assert response.text == "Tallinn is the capital of Estonia."
        assert response.stop_reason == "stop"
        assert response.usage.input_tokens == 12
        assert response.usage.output_tokens == 8
        assert response.resolved_model == "gemini-2.5-flash"

    @respx.mock
    def test_a_function_call_becomes_a_tool_call_with_a_synthetic_id(self, client: Aaron) -> None:
        # Gemini does not send a call id, so one is generated to keep the pairing.
        respx.post(f"{BASE}/models/gemini-2.5-pro:generateContent").mock(
            return_value=httpx.Response(200, json=load("google_tool_call"))
        )
        response = client.chat("google/gemini-2.5-pro", "weather?")
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"city": "Tallinn", "units": "c"}
        assert response.tool_calls[0].id

    @respx.mock
    def test_thought_parts_are_not_mixed_into_the_answer(self, client: Aaron) -> None:
        respx.post(f"{BASE}/models/gemini-2.5-pro:generateContent").mock(
            return_value=httpx.Response(200, json=load("google_thinking"))
        )
        response = client.chat("google/gemini-2.5-pro", "capital?")
        assert response.text == "Tallinn."
        assert response.usage.reasoning_tokens == 22

    @respx.mock
    def test_a_blocked_prompt_is_a_content_filter_error(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("google_blocked")))
        with pytest.raises(ContentFilterError, match="SAFETY"):
            client.chat("google/gemini-2.5-flash", "something disallowed")

    @respx.mock
    def test_no_candidates_and_no_block_reason_is_a_provider_error(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json={"usageMetadata": {}}))
        with pytest.raises(ProviderError, match="no candidates"):
            client.chat("google/gemini-2.5-flash", "hi")


class TestStreaming:
    """Gemini streams SSE when asked with alt=sse."""

    CHUNKS = (
        {
            "candidates": [{"content": {"parts": [{"text": "Tal"}], "role": "model"}, "index": 0}],
            "modelVersion": "gemini-2.5-flash",
        },
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "linn"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
        },
    )

    @respx.mock
    def test_streaming_asks_for_sse_and_yields_one_done(self, client: Aaron) -> None:
        route = respx.post(url__startswith=STREAM_URL).mock(
            return_value=httpx.Response(200, text=sse(*self.CHUNKS))
        )
        events = list(client.stream("google/gemini-2.5-flash", "capital?"))

        assert "alt=sse" in str(route.calls[0].request.url)
        assert [e.text for e in events if isinstance(e, TextEvent)] == ["Tal", "linn"]
        done = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done) == 1
        assert done[0].response.text == "Tallinn"
        assert done[0].response.usage.output_tokens == 2

    @respx.mock
    async def test_async_streaming(self, keys: dict[str, str]) -> None:
        respx.post(url__startswith=STREAM_URL).mock(
            return_value=httpx.Response(200, text=sse(*self.CHUNKS))
        )
        async with AsyncAaron(api_keys=keys) as client:
            events = [e async for e in client.stream("google/gemini-2.5-flash", "hi")]
        assert [e.text for e in events if isinstance(e, TextEvent)] == ["Tal", "linn"]


class TestStructuredOutput:
    """Gemini takes a response schema natively."""

    class City(BaseModel):
        """A city."""

        name: str
        country: str

    @respx.mock
    def test_extract_sends_a_response_schema(self, client: Aaron) -> None:
        payload = load("google_chat")
        payload["candidates"][0]["content"]["parts"] = [
            {"text": '{"name": "Tallinn", "country": "Estonia"}'}
        ]
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=payload))
        city = client.extract("google/gemini-2.5-flash", "capital of Estonia?", schema=self.City)

        assert city.name == "Tallinn"
        config = json.loads(route.calls[0].request.content)["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseSchema"]["properties"]["name"]["type"] == "string"
