"""Shared fixtures.

No test in this suite touches the network. Every HTTP interaction goes through respx,
and `no_network` fails loudly if any code path tries to open a real socket anyway.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest

# The single fixture credential. Every leak test scans serialised output for this
# exact string, so it must be distinctive and must never be changed to something a
# provider payload could plausibly contain.
FIXTURE_KEY = "sk-aaron-test-DO-NOT-LEAK-8f3a1c9e"

FIXTURES = Path(__file__).parent / "fixtures"

AARON_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_COMPAT_API_KEY",
    "OLLAMA_HOST",
    "AARON_DEFAULT_MODEL",
    "AARON_POLICY_FILE",
    "AARON_AUDIT_PATH",
    "AARON_TIMEOUT",
    "AARON_CONFIG",
    "AARON_LIVE_TESTS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Remove every variable Aaron reads, so a developer's shell cannot change a result."""
    for name in AARON_ENV:
        monkeypatch.delenv(name, raising=False)
    # A policy file in the working directory is picked up implicitly, so run from a
    # directory that is guaranteed not to have one.
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn any real connection attempt into an immediate, obvious failure."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "a test tried to open a real socket. Every HTTP call must be mocked with respx."
        )

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def api_key() -> str:
    """The fixture credential, as a plain string."""
    return FIXTURE_KEY


@pytest.fixture
def keys() -> dict[str, str]:
    """A credential for every provider that wants one."""
    return {
        "openai": FIXTURE_KEY,
        "anthropic": FIXTURE_KEY,
        "google": FIXTURE_KEY,
        "openai_compat": FIXTURE_KEY,
    }


def load(name: str) -> dict[str, Any]:
    """Read a recorded provider payload from ``tests/fixtures``.

    Args:
        name: The file name, without the ``.json`` suffix.

    Returns:
        The parsed payload. These are real response shapes with the text shortened and
        every identifier replaced, so nothing here came from a real account.
    """
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def sse(*chunks: Any) -> str:
    """Build an SSE body from payloads, ending with the OpenAI style sentinel."""
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return "".join(lines) + "data: [DONE]\n\n"


def sse_typed(*events: tuple[str, Any]) -> str:
    """Build an SSE body whose frames carry an explicit event name, as Anthropic does."""
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def ndjson(*chunks: Any) -> str:
    """Build a newline delimited JSON body, as Ollama streams."""
    return "".join(f"{json.dumps(chunk)}\n" for chunk in chunks)


@pytest.fixture
def audit_path(tmp_path: Path) -> Path:
    """A path for a JSONL audit log that does not exist yet."""
    return tmp_path / "audit" / "aaron.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse every line of a JSONL file."""
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def live_tests_enabled() -> bool:
    """Whether the opt in live tests should run."""
    return os.environ.get("AARON_LIVE_TESTS") == "1"
