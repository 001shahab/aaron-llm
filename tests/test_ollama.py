"""Tests for the Ollama provider.

Ollama is the provider a reader can run for free, so it is also the one the README
opens with. It needs no credential, it streams NDJSON rather than SSE, and its errors
should tell a first time user to pull the model.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import BaseModel

from aaron import Aaron, AsyncAaron, Message, Tool
from aaron.errors import LocalProviderUnavailable, MissingAPIKey
from aaron.retry import RetryPolicy
from aaron.stream import DoneEvent, TextEvent
from conftest import load, ndjson

URL = "http://localhost:11434/api/chat"


@pytest.fixture
def client() -> Aaron:
    """A client with no credentials at all, which is the point of this provider."""
    return Aaron(retry=RetryPolicy(attempts=1))


class TestNoCredential:
    """Nothing about Ollama should ask for a key."""

    @respx.mock
    def test_a_call_works_with_no_keys_configured(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("ollama_chat")))
        response = client.chat("ollama/llama3.1", "hi")
        assert response.text == "Tallinn is the capital of Estonia."
        assert "authorization" not in route.calls[0].request.headers

    def test_a_remote_provider_still_demands_one(self, client: Aaron) -> None:
        with pytest.raises(MissingAPIKey):
            client.chat("openai/gpt-4o", "hi")


class TestRequestShape:
    """The /api/chat wire format."""

    @respx.mock
    def test_messages_are_flat_strings_and_stream_is_explicit(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("ollama_chat")))
        client.chat("ollama/llama3.1", "hi")
        body = json.loads(route.calls[0].request.content)
        assert body["model"] == "llama3.1"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        # Ollama streams unless told not to, so the flag is always sent.
        assert body["stream"] is False

    @respx.mock
    def test_sampling_options_go_into_the_options_object(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("ollama_chat")))
        client.chat(
            "ollama/llama3.1", "hi", temperature=0.4, top_p=0.7, max_tokens=64, stop=["END"]
        )
        options = json.loads(route.calls[0].request.content)["options"]
        assert options["temperature"] == 0.4
        assert options["top_p"] == 0.7
        assert options["num_predict"] == 64
        assert options["stop"] == ["END"]

    @respx.mock
    def test_a_system_message_stays_a_message(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("ollama_chat")))
        client.chat("ollama/llama3.1", [Message.system("Be brief."), Message.user("hi")])
        roles = [m["role"] for m in json.loads(route.calls[0].request.content)["messages"]]
        assert roles == ["system", "user"]

    @respx.mock
    def test_images_sit_beside_the_text_as_base64(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("ollama_chat")))
        png = bytes.fromhex("89504e470d0a1a0a") + b"body"
        client.chat("ollama/llava", Message.user("what is this", images=[png]))
        message = json.loads(route.calls[0].request.content)["messages"][0]
        assert message["content"] == "what is this"
        assert len(message["images"]) == 1

    @respx.mock
    def test_tools_use_the_openai_function_envelope(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=load("ollama_chat")))
        tool = Tool(name="get_weather", description="d", parameters={"type": "object"})
        client.chat("ollama/llama3.1", "weather?", tools=[tool])
        tools = json.loads(route.calls[0].request.content)["tools"]
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_weather"

    @respx.mock
    def test_the_host_can_be_moved_with_an_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
        route = respx.post("http://gpu-box:11434/api/chat").mock(
            return_value=httpx.Response(200, json=load("ollama_chat"))
        )
        Aaron().chat("ollama/llama3.1", "hi")
        assert route.called


class TestResponseParsing:
    """Ollama reports token counts under its own names."""

    @respx.mock
    def test_text_and_usage(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("ollama_chat")))
        response = client.chat("ollama/llama3.1", "hi")
        assert response.text == "Tallinn is the capital of Estonia."
        assert response.usage.input_tokens == 18
        assert response.usage.output_tokens == 9
        assert response.stop_reason == "stop"

    @respx.mock
    def test_a_local_model_costs_nothing_and_says_so_exactly(self, client: Aaron) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("ollama_chat")))
        response = client.chat("ollama/llama3.1", "hi")
        assert response.cost.usd == 0.0
        assert response.cost.estimated is False

    @respx.mock
    def test_tool_calls_arrive_already_parsed(self, client: Aaron) -> None:
        # Ollama sends arguments as an object, not as a JSON string.
        respx.post(URL).mock(return_value=httpx.Response(200, json=load("ollama_tool_call")))
        response = client.chat("ollama/llama3.1", "weather?")
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"city": "Tallinn", "units": "c"}
        assert response.tool_calls[0].id


class TestErrors:
    """A first time user meets these, so the message has to be useful."""

    @respx.mock
    def test_a_missing_model_says_how_to_pull_it(self, client: Aaron) -> None:
        respx.post(URL).mock(
            return_value=httpx.Response(404, json=load("ollama_error_missing_model"))
        )
        with pytest.raises(Exception, match="ollama pull") as info:
            client.chat("ollama/llama3.1", "hi")
        assert "llama3.1" in str(info.value)

    @respx.mock
    def test_a_refused_connection_says_the_server_is_not_running(self, client: Aaron) -> None:
        respx.post(URL).mock(side_effect=httpx.ConnectError("connection refused"))
        with pytest.raises(LocalProviderUnavailable) as info:
            client.chat("ollama/llama3.1", "hi")
        assert "ollama" in str(info.value).lower()


class TestStreaming:
    """NDJSON, one JSON object per line, no sentinel."""

    CHUNKS = (
        {"model": "llama3.1", "message": {"role": "assistant", "content": "Tal"}, "done": False},
        {"model": "llama3.1", "message": {"role": "assistant", "content": "linn"}, "done": False},
        {
            "model": "llama3.1",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 6,
            "eval_count": 2,
        },
    )

    @respx.mock
    def test_ndjson_becomes_text_events_and_one_done(self, client: Aaron) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, text=ndjson(*self.CHUNKS)))
        events = list(client.stream("ollama/llama3.1", "capital?"))

        assert json.loads(route.calls[0].request.content)["stream"] is True
        assert [e.text for e in events if isinstance(e, TextEvent)] == ["Tal", "linn"]
        done = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done) == 1
        assert done[0].response.text == "Tallinn"
        assert done[0].response.usage.output_tokens == 2

    @respx.mock
    def test_a_blank_line_in_the_stream_is_ignored(self, client: Aaron) -> None:
        body = ndjson(self.CHUNKS[0]) + "\n" + ndjson(*self.CHUNKS[1:])
        respx.post(URL).mock(return_value=httpx.Response(200, text=body))
        events = list(client.stream("ollama/llama3.1", "hi"))
        assert [e.text for e in events if isinstance(e, TextEvent)] == ["Tal", "linn"]

    @respx.mock
    async def test_async_streaming(self) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, text=ndjson(*self.CHUNKS)))
        async with AsyncAaron() as client:
            events = [e async for e in client.stream("ollama/llama3.1", "hi")]
        assert [e.text for e in events if isinstance(e, TextEvent)] == ["Tal", "linn"]
        assert isinstance(events[-1], DoneEvent)


class TestStructuredOutput:
    """Ollama takes a JSON schema in its format field."""

    class City(BaseModel):
        """A city."""

        name: str
        country: str

    @respx.mock
    def test_extract_sends_the_schema_as_format(self, client: Aaron) -> None:
        payload = load("ollama_chat")
        payload["message"]["content"] = '{"name": "Tallinn", "country": "Estonia"}'
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=payload))
        city = client.extract("ollama/llama3.1", "capital of Estonia?", schema=self.City)

        assert city.name == "Tallinn"
        body = json.loads(route.calls[0].request.content)
        assert body["format"]["properties"]["name"]["type"] == "string"
