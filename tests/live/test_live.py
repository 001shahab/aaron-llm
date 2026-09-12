"""Live tests against real endpoints. Skipped unless ``AARON_LIVE_TESTS=1``.

Every other test in this repository is offline and mocked. These are the only ones
that open a socket, and they exist for one reason: a mocked test proves we send what
we think we send, not that the provider still accepts it. Run them before a release.

They cost money on the paid providers. The Ollama tests cost nothing, so they are
worth running often; run only those with:

    AARON_LIVE_TESTS=1 pytest tests/live -k ollama

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from aaron import Aaron, AsyncAaron, Message, Tool
from aaron.stream import DoneEvent, TextEvent

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("AARON_LIVE_TESTS") != "1",
        reason="live tests need AARON_LIVE_TESTS=1 and real credentials",
    ),
]

PROMPT = "Reply with exactly one word: Tallinn"


@pytest.fixture(autouse=True)
def real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the socket block from the top level conftest, for these tests only."""
    import socket

    monkeypatch.undo()
    assert socket.socket  # the real one, restored by monkeypatch.undo()


def needs(variable: str) -> None:
    """Skip when a credential is not in the environment."""
    if not os.environ.get(variable):
        pytest.skip(f"{variable} is not set")


class TestOllama:
    """Free, local, and the quickstart in the README. Needs `ollama serve`."""

    MODEL = "ollama/llama3.1"

    def test_chat(self) -> None:
        response = Aaron().chat(self.MODEL, PROMPT)
        assert response.text.strip()
        assert response.usage.output_tokens > 0
        assert response.cost.usd == 0.0

    def test_stream(self) -> None:
        events = list(Aaron().stream(self.MODEL, PROMPT))
        assert isinstance(events[-1], DoneEvent)
        assert any(isinstance(event, TextEvent) for event in events)

    def test_extract(self) -> None:
        class City(BaseModel):
            """A city."""

            name: str
            country: str

        city = Aaron().extract(self.MODEL, "The capital of Estonia, as JSON.", schema=City)
        assert city.name

    def test_tools(self) -> None:
        def get_weather(city: str) -> str:
            """Look up the weather in a city.

            Args:
                city: The city name.
            """
            return "sunny"

        response = Aaron().chat(
            self.MODEL,
            "What is the weather in Tallinn? Use the tool.",
            tools=[Tool.from_function(get_weather)],
        )
        assert response.tool_calls or response.text


class TestOpenAI:
    """Costs money. One short call each."""

    MODEL = "openai/gpt-4o-mini"

    def test_chat(self) -> None:
        needs("OPENAI_API_KEY")
        response = Aaron().chat(self.MODEL, PROMPT, max_tokens=10)
        assert "tallinn" in response.text.lower()
        assert response.cost.usd > 0
        assert response.cost.estimated is False

    def test_stream(self) -> None:
        needs("OPENAI_API_KEY")
        events = list(Aaron().stream(self.MODEL, PROMPT, max_tokens=10))
        assert isinstance(events[-1], DoneEvent)
        assert events[-1].response.usage.output_tokens > 0

    def test_vision(self) -> None:
        needs("OPENAI_API_KEY")
        # A 1x1 red PNG, the smallest honest test of the image path.
        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000d4944415478da63f8cfc0000003010100b6c8b4b40000000049454e44ae426082"
        )
        response = Aaron().chat(
            self.MODEL, Message.user("What colour is this? One word.", images=[png]), max_tokens=10
        )
        assert response.text.strip()

    async def test_async_chat(self) -> None:
        needs("OPENAI_API_KEY")
        async with AsyncAaron() as client:
            response = await client.chat(self.MODEL, PROMPT, max_tokens=10)
        assert response.text.strip()


class TestAnthropic:
    """Costs money."""

    MODEL = "anthropic/claude-haiku-4-5"

    def test_chat(self) -> None:
        needs("ANTHROPIC_API_KEY")
        response = Aaron().chat(self.MODEL, PROMPT, max_tokens=10)
        assert "tallinn" in response.text.lower()

    def test_stream(self) -> None:
        needs("ANTHROPIC_API_KEY")
        events = list(Aaron().stream(self.MODEL, PROMPT, max_tokens=10))
        assert isinstance(events[-1], DoneEvent)

    def test_extract_uses_a_forced_tool_call(self) -> None:
        needs("ANTHROPIC_API_KEY")

        class City(BaseModel):
            """A city."""

            name: str
            country: str

        city = Aaron().extract(self.MODEL, "The capital of Estonia.", schema=City)
        assert city.country.lower().startswith("est")


class TestGoogle:
    """Costs money, and has a free tier."""

    MODEL = "google/gemini-2.5-flash"

    def test_chat(self) -> None:
        needs("GOOGLE_API_KEY")
        response = Aaron().chat(self.MODEL, PROMPT, max_tokens=10)
        assert response.text.strip()

    def test_stream(self) -> None:
        needs("GOOGLE_API_KEY")
        events = list(Aaron().stream(self.MODEL, PROMPT, max_tokens=64))
        assert isinstance(events[-1], DoneEvent)


class TestRegistryIsStillAccurate:
    """The registry claims a context window and a price. Check the cheap half."""

    def test_the_resolved_model_names_look_current(self) -> None:
        needs("OPENAI_API_KEY")
        response = Aaron().chat("openai/gpt-4o-mini", "hi", max_tokens=5)
        # A dated snapshot id means the alias still resolves to something real.
        assert response.resolved_model.startswith("gpt-4o-mini")
