"""The contract every provider must satisfy, run against all five.

A new provider is only correct if it passes this file unchanged. That is the whole
point of the abstraction: a caller who switches provider should not have to learn a
second set of behaviours.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import typing
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import respx

from aaron import Aaron, AsyncAaron, Message, Tool
from aaron.providers import available_providers, get_provider
from aaron.stream import DoneEvent, StreamEvent, TextEvent
from aaron.types import Response
from conftest import FIXTURE_KEY, load, ndjson, sse, sse_typed

# The union is an annotated discriminated union, so isinstance needs the members.
EVENT_CLASSES = tuple(typing.get_args(typing.get_args(StreamEvent)[0]))

TOOL = Tool(
    name="get_weather",
    description="Look up the weather.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


@dataclass(frozen=True)
class Case:
    """Everything needed to drive one provider through the contract."""

    provider: str
    model: str
    payload: dict[str, Any]
    stream_body: str
    tool_payload: dict[str, Any]
    base_urls: dict[str, str] = ()  # type: ignore[assignment]

    @property
    def name(self) -> str:
        """The provider name, used as the test id."""
        return self.provider


OPENAI_STREAM = sse(
    {"id": "s", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "Tal"}}]},
    {"id": "s", "choices": [{"index": 0, "delta": {"content": "linn"}}], "usage": {}},
    {"id": "s", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
)

ANTHROPIC_STREAM = sse_typed(
    (
        "message_start",
        {"type": "message_start", "message": {"id": "m", "model": "claude", "usage": {}}},
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Tal"}},
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "linn"},
        },
    ),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
    ("message_stop", {"type": "message_stop"}),
)

GOOGLE_STREAM = sse(
    {"candidates": [{"content": {"parts": [{"text": "Tal"}], "role": "model"}, "index": 0}]},
    {
        "candidates": [
            {
                "content": {"parts": [{"text": "linn"}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
    },
)

OLLAMA_STREAM = ndjson(
    {"model": "llama3.1", "message": {"content": "Tal"}, "done": False},
    {"model": "llama3.1", "message": {"content": "linn"}, "done": False},
    {"model": "llama3.1", "message": {"content": ""}, "done": True, "done_reason": "stop"},
)

CASES = [
    Case("openai", "openai/gpt-4o", load("openai_chat"), OPENAI_STREAM, load("openai_tool_call")),
    Case(
        "anthropic",
        "anthropic/claude-sonnet-4-5",
        load("anthropic_chat"),
        ANTHROPIC_STREAM,
        load("anthropic_tool_call"),
    ),
    Case(
        "google",
        "google/gemini-2.5-flash",
        load("google_chat"),
        GOOGLE_STREAM,
        load("google_tool_call"),
    ),
    Case("ollama", "ollama/llama3.1", load("ollama_chat"), OLLAMA_STREAM, load("ollama_tool_call")),
    Case(
        "openai_compat",
        "openai_compat/some-model",
        load("openai_chat"),
        OPENAI_STREAM,
        load("openai_tool_call"),
        {"openai_compat": "https://gateway.example.com/v1"},
    ),
]

IDS = [case.provider for case in CASES]


@pytest.fixture
def keys_for_all() -> dict[str, str]:
    """One credential for every provider, including the ones that ignore it."""
    return dict.fromkeys(available_providers(), FIXTURE_KEY)


def build(case: Case, keys: dict[str, str], **kwargs: Any) -> Aaron:
    """A client configured for one case."""
    from aaron.retry import RetryPolicy

    return Aaron(
        api_keys=keys,
        base_urls=dict(case.base_urls or {}),
        retry=RetryPolicy(attempts=1),
        **kwargs,
    )


class TestEveryProviderIsRegistered:
    """The registry of providers itself."""

    def test_the_five_documented_providers_are_present(self) -> None:
        assert set(available_providers()) == {
            "openai",
            "anthropic",
            "google",
            "ollama",
            "openai_compat",
        }

    @pytest.mark.parametrize("name", ["openai", "anthropic", "google", "ollama", "openai_compat"])
    def test_a_provider_declares_the_whole_protocol(self, name: str) -> None:
        provider = get_provider(name)
        assert provider.name == name
        assert isinstance(provider.local, bool)
        assert isinstance(provider.requires_key, bool)
        assert isinstance(provider.env_key, str)
        for method in ("build_request", "parse_response", "iter_events", "map_error"):
            assert callable(getattr(provider, method))

    def test_an_unknown_provider_names_the_ones_that_exist(self) -> None:
        from aaron.errors import UnknownProvider

        with pytest.raises(UnknownProvider) as info:
            get_provider("openai-typo")
        assert "openai" in str(info.value)


@pytest.mark.parametrize("case", CASES, ids=IDS)
class TestContract:
    """Behaviour that must not vary between providers."""

    @respx.mock
    def test_chat_returns_a_valid_response(self, case: Case, keys_for_all: dict[str, str]) -> None:
        respx.route(method="POST").mock(return_value=httpx.Response(200, json=case.payload))
        response = build(case, keys_for_all).chat(case.model, "What is the capital of Estonia?")

        assert isinstance(response, Response)
        assert response.text == "Tallinn is the capital of Estonia."
        assert response.model == case.model
        assert response.resolved_model
        assert response.id
        assert response.stop_reason == "stop"
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.latency_ms >= 0
        assert response.cost.usd >= 0.0

    @respx.mock
    def test_a_stream_ends_with_exactly_one_done_event(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        respx.route(method="POST").mock(return_value=httpx.Response(200, text=case.stream_body))
        events = list(build(case, keys_for_all).stream(case.model, "capital?"))

        done = [event for event in events if isinstance(event, DoneEvent)]
        assert len(done) == 1, "a stream must carry exactly one done event"
        assert isinstance(events[-1], DoneEvent), "the done event must be last"
        assert [e.text for e in events if isinstance(e, TextEvent)] == ["Tal", "linn"]

    @respx.mock
    def test_streamed_and_whole_text_agree(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        # The assembled response must equal the concatenation of the text events.
        respx.route(method="POST").mock(return_value=httpx.Response(200, text=case.stream_body))
        events = list(build(case, keys_for_all).stream(case.model, "capital?"))
        streamed = "".join(e.text for e in events if isinstance(e, TextEvent))
        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.response.text == streamed

    @respx.mock
    def test_every_event_is_a_member_of_the_documented_union(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        respx.route(method="POST").mock(return_value=httpx.Response(200, text=case.stream_body))
        for event in build(case, keys_for_all).stream(case.model, "hi"):
            assert isinstance(event, EVENT_CLASSES)

    @respx.mock
    def test_a_tool_call_is_parsed_into_arguments(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        respx.route(method="POST").mock(return_value=httpx.Response(200, json=case.tool_payload))
        response = build(case, keys_for_all).chat(case.model, "weather?", tools=[TOOL])

        assert response.stop_reason == "tool_use"
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call.id, "every tool call needs an id, synthesised if the provider omits one"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Tallinn", "units": "c"}

    @respx.mock
    def test_a_tool_result_can_be_sent_back(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        # Round tripping a tool call must produce a body the provider would accept,
        # which here means it must at least build and send without raising.
        route = respx.route(method="POST").mock(
            return_value=httpx.Response(200, json=case.tool_payload)
        )
        client = build(case, keys_for_all)
        first = client.chat(case.model, "weather?", tools=[TOOL])
        conversation = [
            Message.user("weather?"),
            first.message,
            Message.tool(first.tool_calls[0].id, "sunny, 7C"),
        ]
        respx.route(method="POST").mock(return_value=httpx.Response(200, json=case.payload))
        client.chat(case.model, conversation, tools=[TOOL])
        assert route.called

    @respx.mock
    def test_the_system_prompt_is_accepted_in_some_form(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        route = respx.route(method="POST").mock(
            return_value=httpx.Response(200, json=case.payload)
        )
        build(case, keys_for_all).chat(
            case.model, [Message.system("Be brief."), Message.user("hi")]
        )
        # Wherever it goes, it must reach the wire.
        assert "Be brief." in route.calls[0].request.content.decode()

    @respx.mock
    def test_dry_run_builds_a_request_without_sending_anything(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        route = respx.route(method="POST").mock(
            return_value=httpx.Response(200, json=case.payload)
        )
        prepared = build(case, keys_for_all).dry_run(case.model, "hi")

        assert not route.called, "dry_run must never send a request"
        assert prepared.method == "POST"
        assert prepared.url.startswith("http")
        assert isinstance(prepared.body, dict)

    @respx.mock
    def test_an_error_status_maps_to_an_aaron_error(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        from aaron.errors import AaronError

        respx.route(method="POST").mock(
            return_value=httpx.Response(500, json={"error": {"message": "boom"}})
        )
        with pytest.raises(AaronError) as info:
            build(case, keys_for_all).chat(case.model, "hi")
        assert info.value.provider == case.provider
        assert info.value.status_code == 500

    @respx.mock
    async def test_the_async_client_produces_the_same_text(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        respx.route(method="POST").mock(return_value=httpx.Response(200, json=case.payload))
        async with AsyncAaron(
            api_keys=keys_for_all, base_urls=dict(case.base_urls or {})
        ) as client:
            response = await client.chat(case.model, "hi")
        assert response.text == "Tallinn is the capital of Estonia."

    @respx.mock
    async def test_async_streaming_matches_sync_streaming(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        respx.route(method="POST").mock(return_value=httpx.Response(200, text=case.stream_body))
        sync_events = [
            e.text for e in build(case, keys_for_all).stream(case.model, "hi")
            if isinstance(e, TextEvent)
        ]

        respx.route(method="POST").mock(return_value=httpx.Response(200, text=case.stream_body))
        async with AsyncAaron(
            api_keys=keys_for_all, base_urls=dict(case.base_urls or {})
        ) as client:
            async_events = [
                e.text async for e in client.stream(case.model, "hi") if isinstance(e, TextEvent)
            ]

        assert sync_events == async_events


@pytest.mark.parametrize("case", CASES, ids=IDS)
class TestNoCredentialLeaks:
    """Section 18 of the specification, enforced for every provider."""

    @respx.mock
    def test_the_key_is_absent_from_every_representation_of_a_response(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        respx.route(method="POST").mock(return_value=httpx.Response(200, json=case.payload))
        response = build(case, keys_for_all).chat(case.model, "hi")

        for rendered in (repr(response), str(response), response.model_dump_json()):
            assert FIXTURE_KEY not in rendered

    @respx.mock
    def test_the_key_is_absent_from_a_dry_run(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        prepared = build(case, keys_for_all).dry_run(case.model, "hi")
        rendered = repr(prepared) + json.dumps(prepared.headers) + json.dumps(prepared.body)
        assert FIXTURE_KEY not in rendered

    @respx.mock
    def test_the_key_is_absent_from_an_error(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        from aaron.errors import AaronError

        respx.route(method="POST").mock(
            return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
        )
        with pytest.raises(AaronError) as info:
            build(case, keys_for_all).chat(case.model, "hi")

        error = info.value
        for rendered in (str(error), repr(error), str(error.raw), str(error.__dict__)):
            assert FIXTURE_KEY not in rendered

    @respx.mock
    def test_the_key_is_absent_from_the_client_repr(
        self, case: Case, keys_for_all: dict[str, str]
    ) -> None:
        client = build(case, keys_for_all)
        assert FIXTURE_KEY not in repr(client)
        assert FIXTURE_KEY not in repr(client.__dict__)
