"""Tests for messages, content parts, secrets and tool schemas.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from aaron import Message, Tool
from aaron.errors import InvalidRequest
from aaron.types import DocumentPart, ImagePart, SecretValue, TextPart, ToolCall, Usage

PNG = bytes.fromhex("89504e470d0a1a0a") + b"rest of a png"
JPEG = bytes.fromhex("ffd8ff") + b"rest of a jpeg"
PDF = b"%PDF-1.7\nrest of a pdf"


class TestSecretValue:
    """A credential must never be readable by accident."""

    def test_repr_is_masked(self) -> None:
        secret = SecretValue("sk-super-secret-value")
        assert "sk-super-secret-value" not in repr(secret)
        assert "sk-super-secret-value" not in str(secret)
        assert "sk-super-secret-value" not in f"{secret}"
        assert "sk-super-secret-value" not in "{}".format(secret)

    def test_get_returns_the_real_value(self) -> None:
        assert SecretValue("sk-abc").get() == "sk-abc"

    def test_masked_keeps_a_recognisable_tail(self) -> None:
        assert SecretValue("sk-0123456789abcdef").masked().endswith("cdef")
        assert "0123456789" not in SecretValue("sk-0123456789abcdef").masked()

    def test_short_secret_is_fully_masked(self) -> None:
        assert SecretValue("abc").masked() == "****"

    def test_secret_is_not_in_a_container_repr(self) -> None:
        # A dict or list repr calls repr on its members, which is the usual leak.
        assert "sk-abc" not in repr({"authorization": SecretValue("sk-abc")})
        assert "sk-abc" not in repr([SecretValue("sk-abc")])

    def test_the_value_survives_a_round_trip_through_a_container(self) -> None:
        holder = {"key": SecretValue("sk-abc")}
        assert holder["key"].get() == "sk-abc"


class TestMessageConstructors:
    """The four constructors are the only supported way to build a message."""

    def test_system_and_user_and_assistant(self) -> None:
        assert Message.system("be brief").role == "system"
        assert Message.user("hello").text == "hello"
        assert Message.assistant("hi").role == "assistant"

    def test_user_rejects_an_empty_message(self) -> None:
        with pytest.raises(InvalidRequest, match="needs text, an image or a document"):
            Message.user("")

    def test_tool_message_needs_its_call_id(self) -> None:
        message = Message.tool("call_1", "sunny")
        assert message.role == "tool"
        assert message.tool_call_id == "call_1"
        assert message.text == "sunny"

    def test_tool_message_serialises_a_non_string_result(self) -> None:
        assert Message.tool("call_1", {"temp": 7}).text == '{"temp": 7}'

    def test_messages_are_frozen(self) -> None:
        message = Message.user("hello")
        with pytest.raises(Exception, match="frozen|immutable"):
            message.role = "system"  # type: ignore[misc]


class TestImages:
    """Images arrive as bytes, a path, base64 or a data URL. Never as a remote URL."""

    def test_bytes_are_sniffed_and_encoded(self) -> None:
        message = Message.user("what is this", images=[PNG])
        part = message.content[1]
        assert isinstance(part, ImagePart)
        assert part.media_type == "image/png"
        assert base64.b64decode(part.data) == PNG

    def test_jpeg_is_sniffed(self) -> None:
        part = Message.user("x", images=[JPEG]).content[1]
        assert isinstance(part, ImagePart)
        assert part.media_type == "image/jpeg"

    def test_a_path_is_read_from_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "shot.png"
        path.write_bytes(PNG)
        part = Message.user("x", images=[path]).content[1]
        assert isinstance(part, ImagePart)
        assert part.media_type == "image/png"

    def test_a_path_string_is_read_from_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "shot.png"
        path.write_bytes(PNG)
        part = Message.user("x", images=[str(path)]).content[1]
        assert isinstance(part, ImagePart)
        assert base64.b64decode(part.data) == PNG

    def test_base64_string_is_accepted(self) -> None:
        part = Message.user("x", images=[base64.b64encode(PNG).decode()]).content[1]
        assert isinstance(part, ImagePart)
        assert part.media_type == "image/png"

    def test_data_url_is_accepted(self) -> None:
        encoded = base64.b64encode(PNG).decode()
        part = Message.user("x", images=[f"data:image/png;base64,{encoded}"]).content[1]
        assert isinstance(part, ImagePart)
        assert part.media_type == "image/png"
        assert part.data == encoded

    def test_a_remote_url_is_refused(self) -> None:
        # v1 does not fetch anything on the caller's behalf.
        with pytest.raises(InvalidRequest):
            Message.user("x", images=["https://example.com/cat.png"])

    def test_unrecognised_bytes_are_refused_rather_than_guessed(self) -> None:
        with pytest.raises(InvalidRequest, match="media type"):
            Message.user("x", images=[b"\x00\x01\x02not a known format"])


class TestDocuments:
    """Documents are PDFs in v1."""

    def test_pdf_bytes(self) -> None:
        part = Message.user("summarise", documents=[PDF]).content[1]
        assert isinstance(part, DocumentPart)
        assert part.media_type == "application/pdf"

    def test_pdf_from_disk_keeps_its_name(self, tmp_path: Path) -> None:
        path = tmp_path / "contract.pdf"
        path.write_bytes(PDF)
        part = Message.user("summarise", documents=[path]).content[1]
        assert isinstance(part, DocumentPart)
        assert part.name == "contract.pdf"

    def test_text_comes_first_so_the_prompt_is_read_before_the_file(self) -> None:
        message = Message.user("summarise", documents=[PDF])
        assert isinstance(message.content[0], TextPart)
        assert isinstance(message.content[1], DocumentPart)


class TestToolFromFunction:
    """A tool schema is derived from the signature and the docstring, with no decorator."""

    def test_name_description_and_parameters(self) -> None:
        def get_weather(city: str, units: str = "c") -> str:
            """Look up the weather for a city.

            Args:
                city: The city name.
                units: Either c or f.

            Returns:
                A short description.
            """
            return "sunny"

        tool = Tool.from_function(get_weather)
        assert tool.name == "get_weather"
        assert tool.description == "Look up the weather for a city."
        assert tool.parameters["properties"]["city"] == {
            "type": "string",
            "description": "The city name.",
        }
        assert tool.parameters["required"] == ["city"]

    def test_a_multi_line_argument_description_is_joined(self) -> None:
        def f(a: str) -> None:
            """Do a thing.

            Args:
                a: The first part
                    and the continuation.
            """

        assert Tool.from_function(f).parameters["properties"]["a"]["description"] == (
            "The first part and the continuation."
        )

    def test_types_are_mapped(self) -> None:
        def f(a: int, b: float, c: bool, d: list[str], e: dict[str, int]) -> None:
            """Take many types."""

        properties = Tool.from_function(f).parameters["properties"]
        assert properties["a"]["type"] == "integer"
        assert properties["b"]["type"] == "number"
        assert properties["c"]["type"] == "boolean"
        assert properties["d"] == {"type": "array", "items": {"type": "string"}}
        assert properties["e"]["type"] == "object"

    def test_optional_is_not_required(self) -> None:
        def f(a: str, b: str | None = None) -> None:
            """Take an optional."""

        assert Tool.from_function(f).parameters["required"] == ["a"]

    def test_a_literal_becomes_an_enum(self) -> None:
        def f(unit: Literal["c", "f"]) -> None:
            """Take a literal."""

        assert Tool.from_function(f).parameters["properties"]["unit"]["enum"] == ["c", "f"]

    def test_an_undocumented_function_is_refused(self) -> None:
        def f(a: str) -> None:
            return None

        with pytest.raises(InvalidRequest, match="docstring"):
            Tool.from_function(f)

    def test_an_unannotated_parameter_is_refused(self) -> None:
        def f(a) -> None:  # type: ignore[no-untyped-def]
            """Take anything."""

        with pytest.raises(InvalidRequest, match="annotation"):
            Tool.from_function(f)

    def test_self_and_cls_are_skipped(self) -> None:
        class Handler:
            def run(self, value: str) -> None:
                """Run it.

                Args:
                    value: The value.
                """

        tool = Tool.from_function(Handler().run)
        assert "self" not in tool.parameters["properties"]


class TestToolFromModel:
    """A pydantic model is the other way to declare a tool."""

    def test_schema_is_taken_from_the_model(self) -> None:
        class Search(BaseModel):
            """Search the catalogue."""

            query: str = Field(description="What to look for.")
            limit: int = 10

        tool = Tool.from_model(Search)
        assert tool.name == "Search"
        assert tool.description == "Search the catalogue."
        assert tool.parameters["properties"]["query"]["description"] == "What to look for."

    def test_nested_models_are_inlined_so_no_provider_sees_a_ref(self) -> None:
        class Address(BaseModel):
            city: str

        class Person(BaseModel):
            """A person."""

            name: str
            address: Address

        rendered = str(Tool.from_model(Person).parameters)
        assert "$ref" not in rendered
        assert "$defs" not in rendered
        assert "city" in rendered

    def test_a_custom_name_and_description_win(self) -> None:
        class Search(BaseModel):
            """Original."""

            query: str

        tool = Tool.from_model(Search, name="find", description="Replaced.")
        assert (tool.name, tool.description) == ("find", "Replaced.")


class TestToolValidation:
    """Provider name rules are enforced once, here, rather than five times."""

    @pytest.mark.parametrize("name", ["has space", "has/slash", "", "a" * 65, "has.dot"])
    def test_bad_names_are_refused(self, name: str) -> None:
        with pytest.raises(InvalidRequest):
            Tool(name=name, description="d", parameters={"type": "object", "properties": {}})

    @pytest.mark.parametrize("name", ["get_weather", "getWeather", "a", "_private", "a-b", "f1"])
    def test_good_names_are_accepted(self, name: str) -> None:
        assert Tool(name=name, description="d", parameters={"type": "object"}).name == name

    def test_parameters_must_be_an_object_schema(self) -> None:
        with pytest.raises(InvalidRequest, match="object"):
            Tool(name="f", description="d", parameters={"type": "string"})


class TestUsageAndToolCall:
    """Small value objects."""

    def test_total_tokens_adds_input_and_output(self) -> None:
        assert Usage(input_tokens=10, output_tokens=5).total_tokens == 15

    def test_cached_tokens_are_already_inside_the_input_count(self) -> None:
        usage = Usage(input_tokens=100, cached_input_tokens=80, output_tokens=10)
        assert usage.total_tokens == 110

    def test_tool_call_holds_parsed_arguments(self) -> None:
        call = ToolCall(id="c1", name="f", arguments={"a": 1})
        assert call.arguments["a"] == 1

    def test_message_tool_calls_are_exposed_on_the_message(self) -> None:
        message = Message.assistant("", tool_calls=[ToolCall(id="c1", name="f", arguments={})])
        assert [c.name for c in message.tool_calls] == ["f"]
