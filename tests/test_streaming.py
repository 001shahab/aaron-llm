"""Tests for the decoders and the stream assembler, below the provider layer.

The assembler is the one piece every provider shares, so its invariants are tested
directly rather than only through five providers. The interesting cases are the ugly
ones: a frame split across chunks, a tool call that never closes, a stream that stops
mid sentence.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import random

import pytest

from aaron.errors import ToolArgumentError
from aaron.providers.base import NDJSONDecoder, SSEDecoder, SSEMessage, merge_options
from aaron.request import ChatRequest
from aaron.stream import (
    DoneEvent,
    StreamAssembler,
    TextEvent,
    ThinkingEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    parse_tool_arguments,
)
from aaron.types import Message, Usage


def request(model: str = "openai/gpt-4o") -> ChatRequest:
    """A minimal request for the assembler to describe."""
    provider, _, model_name = model.partition("/")
    return ChatRequest(
        model=model,
        provider=provider,
        model_name=model_name,
        messages=[Message.user("hi")],
        base_url="https://example.invalid",
    )


def lines(body: str) -> list[bytes]:
    """Split a body the way httpx ``iter_lines`` hands it to the decoder."""
    return [line.encode("utf-8") for line in body.split("\n")]


def decode(body: str) -> list[SSEMessage]:
    """Feed a whole SSE body through the decoder, line by line."""
    decoder = SSEDecoder()
    messages = [message for line in lines(body) for message in decoder.feed(line)]
    return messages + decoder.finish()


class TestSSEDecoder:
    """A push decoder fed one line at a time, since httpx already splits lines."""

    def test_a_simple_frame(self) -> None:
        assert [m.data for m in decode('data: {"a": 1}\n\n')] == ['{"a": 1}']

    def test_two_frames(self) -> None:
        body = 'data: {"a": 1}\n\ndata: {"b": 2}\n\n'
        assert [m.data for m in decode(body)] == ['{"a": 1}', '{"b": 2}']

    def test_an_event_name_is_captured(self) -> None:
        messages = decode("event: message_stop\ndata: {}\n\n")
        assert messages[0].event == "message_stop"
        assert messages[0].data == "{}"

    def test_an_event_name_does_not_leak_into_the_next_frame(self) -> None:
        body = "event: first\ndata: {}\n\ndata: {}\n\n"
        assert [m.event for m in decode(body)] == ["first", None]

    def test_several_data_lines_are_joined_with_newlines(self) -> None:
        assert decode("data: line one\ndata: line two\n\n")[0].data == "line one\nline two"

    def test_a_comment_line_is_ignored(self) -> None:
        # Providers send ": keep-alive" comments to hold the connection open.
        assert [m.data for m in decode(': keep-alive\n\ndata: {"a": 1}\n\n')] == ['{"a": 1}']

    def test_carriage_returns_are_stripped(self) -> None:
        assert [m.data for m in decode('data: {"a": 1}\r\n\r\n')] == ['{"a": 1}']

    def test_the_done_sentinel_is_consumed_not_surfaced(self) -> None:
        # [DONE] is not a chunk, and a provider must never try to parse it as JSON.
        assert decode('data: {"a": 1}\n\ndata: [DONE]\n\n') == [
            SSEMessage(event=None, data='{"a": 1}')
        ]

    def test_only_one_space_after_the_colon_is_removed(self) -> None:
        assert decode("data:  leading space kept\n\n")[0].data == " leading space kept"

    def test_a_frame_with_no_trailing_blank_line_is_still_emitted_at_the_end(self) -> None:
        # A provider that closes the connection without a final blank line must not
        # cost the caller the last chunk.
        assert [m.data for m in decode('data: {"a": 1}')] == ['{"a": 1}']

    def test_an_unknown_field_is_ignored(self) -> None:
        assert [m.data for m in decode('id: 7\nretry: 100\ndata: {"a": 1}\n\n')] == ['{"a": 1}']


class TestNDJSONDecoder:
    """One JSON object per line, no framing at all."""

    def test_two_lines(self) -> None:
        decoder = NDJSONDecoder()
        received = [m.data for line in lines('{"a": 1}\n{"b": 2}\n') for m in decoder.feed(line)]
        assert received == ['{"a": 1}', '{"b": 2}']

    def test_blank_lines_are_skipped(self) -> None:
        decoder = NDJSONDecoder()
        received = [m.data for line in lines('\n\n{"a": 1}\n\n') for m in decoder.feed(line)]
        assert received == ['{"a": 1}']

    def test_nothing_is_buffered(self) -> None:
        assert NDJSONDecoder().finish() == []


class TestAssemblerText:
    """Text accumulates in order."""

    def test_text_events_concatenate(self) -> None:
        assembler = StreamAssembler(request())
        for chunk in ("Hello", ", ", "world"):
            assembler.add(TextEvent(text=chunk))
        assert assembler.done().response.text == "Hello, world"

    def test_thinking_is_not_part_of_the_answer(self) -> None:
        assembler = StreamAssembler(request())
        assembler.add(ThinkingEvent(text="let me think"))
        assembler.add(TextEvent(text="42"))
        response = assembler.done().response
        assert response.text == "42"
        assert "let me think" not in response.text

    def test_an_empty_stream_still_produces_a_response(self) -> None:
        # A model that returns nothing is not an error, it is an empty answer.
        response = StreamAssembler(request()).done().response
        assert response.text == ""
        assert response.stop_reason in ("stop", "other")

    def test_the_response_carries_the_requested_model(self) -> None:
        response = StreamAssembler(request("anthropic/claude-sonnet-4-5")).done().response
        assert response.model == "anthropic/claude-sonnet-4-5"


class TestAssemblerToolCalls:
    """Tool calls arrive in pieces and must end up parsed exactly once."""

    def test_start_delta_end_produces_one_call(self) -> None:
        assembler = StreamAssembler(request())
        assembler.add(ToolCallStartEvent(id="c1", name="get_weather"))
        assembler.add(ToolCallDeltaEvent(id="c1", arguments_json='{"city"'))
        assembler.add(ToolCallDeltaEvent(id="c1", arguments_json=':"Tallinn"}'))
        for event in assembler.close_tool_calls():
            assembler.add(event)

        calls = assembler.done().response.tool_calls
        assert len(calls) == 1
        assert calls[0].arguments == {"city": "Tallinn"}

    def test_an_explicit_end_event_is_not_duplicated_by_close(self) -> None:
        from aaron.types import ToolCall

        assembler = StreamAssembler(request())
        assembler.add(ToolCallStartEvent(id="c1", name="f"))
        assembler.add(ToolCallEndEvent(call=ToolCall(id="c1", name="f", arguments={"a": 1})))
        assert assembler.close_tool_calls() == []
        assert len(assembler.done().response.tool_calls) == 1

    def test_two_concurrent_tool_calls_stay_separate(self) -> None:
        assembler = StreamAssembler(request())
        assembler.add(ToolCallStartEvent(id="c1", name="first"))
        assembler.add(ToolCallStartEvent(id="c2", name="second"))
        assembler.add(ToolCallDeltaEvent(id="c1", arguments_json='{"a": 1}'))
        assembler.add(ToolCallDeltaEvent(id="c2", arguments_json='{"b": 2}'))
        for event in assembler.close_tool_calls():
            assembler.add(event)

        calls = {call.name: call.arguments for call in assembler.done().response.tool_calls}
        assert calls == {"first": {"a": 1}, "second": {"b": 2}}

    def test_a_tool_call_makes_the_stop_reason_tool_use(self) -> None:
        assembler = StreamAssembler(request())
        assembler.add(ToolCallStartEvent(id="c1", name="f"))
        assembler.add(ToolCallDeltaEvent(id="c1", arguments_json="{}"))
        for event in assembler.close_tool_calls():
            assembler.add(event)
        assert assembler.done().response.stop_reason == "tool_use"

    def test_a_delta_for_an_unannounced_call_is_tolerated(self) -> None:
        # Forgiving on input: a provider that skips the start event is not fatal.
        assembler = StreamAssembler(request())
        assembler.add(ToolCallDeltaEvent(id="c9", arguments_json='{"a": 1}'))
        for event in assembler.close_tool_calls():
            assembler.add(event)
        assert assembler.done().response.tool_calls[0].arguments == {"a": 1}

    def test_empty_arguments_become_an_empty_object(self) -> None:
        assembler = StreamAssembler(request())
        assembler.add(ToolCallStartEvent(id="c1", name="f"))
        for event in assembler.close_tool_calls():
            assembler.add(event)
        assert assembler.done().response.tool_calls[0].arguments == {}

    def test_unparseable_arguments_raise_with_the_raw_text(self) -> None:
        assembler = StreamAssembler(request())
        assembler.add(ToolCallStartEvent(id="c1", name="f"))
        assembler.add(ToolCallDeltaEvent(id="c1", arguments_json="{not json"))
        with pytest.raises(ToolArgumentError) as info:
            assembler.close_tool_calls()
        assert info.value.raw_arguments == "{not json"


class TestParseToolArguments:
    """The shared parser used by every provider."""

    def test_an_object(self) -> None:
        assert parse_tool_arguments('{"a": 1}', tool_name="f") == {"a": 1}

    def test_empty_text_is_an_empty_object(self) -> None:
        assert parse_tool_arguments("", tool_name="f") == {}
        assert parse_tool_arguments("   ", tool_name="f") == {}

    def test_a_json_array_is_refused_because_arguments_are_named(self) -> None:
        with pytest.raises(ToolArgumentError):
            parse_tool_arguments("[1, 2]", tool_name="f")

    def test_a_bare_string_is_refused(self) -> None:
        with pytest.raises(ToolArgumentError):
            parse_tool_arguments('"just a string"', tool_name="f")

    def test_the_error_names_the_tool(self) -> None:
        with pytest.raises(ToolArgumentError) as info:
            parse_tool_arguments("{oops", tool_name="get_weather")
        assert info.value.tool_name == "get_weather"


class TestAssemblerUsageAndCost:
    """Usage may be reported, partly reported, or not reported at all."""

    def test_reported_usage_is_used_as_given(self) -> None:
        assembler = StreamAssembler(request())
        assembler.add(TextEvent(text="hi"))
        assembler.note(usage=Usage(input_tokens=11, output_tokens=3))
        response = assembler.done().response
        assert response.usage.input_tokens == 11
        assert response.cost.estimated is False

    def test_missing_usage_is_estimated_and_flagged(self) -> None:
        assembler = StreamAssembler(request())
        assembler.add(TextEvent(text="a fairly long answer " * 5))
        response = assembler.done().response
        assert response.usage.output_tokens > 0
        assert response.cost.estimated is True

    def test_an_unpriced_model_is_flagged_even_with_real_usage(self) -> None:
        assembler = StreamAssembler(request("openai_compat/mystery"))
        assembler.note(usage=Usage(input_tokens=10, output_tokens=10))
        assert assembler.done().response.cost.estimated is True

    def test_a_note_can_set_the_resolved_model_and_id(self) -> None:
        assembler = StreamAssembler(request())
        assembler.note(id="chatcmpl-1", resolved_model="gpt-4o-2024-11-20")
        response = assembler.done().response
        assert response.id == "chatcmpl-1"
        assert response.resolved_model == "gpt-4o-2024-11-20"

    def test_a_later_note_wins_over_an_earlier_one(self) -> None:
        assembler = StreamAssembler(request())
        assembler.note(usage=Usage(output_tokens=1))
        assembler.note(usage=Usage(output_tokens=9))
        assert assembler.done().response.usage.output_tokens == 9

    def test_the_id_falls_back_to_something_non_empty(self) -> None:
        assert StreamAssembler(request()).done().response.id


class TestDoneEvent:
    """The terminal event carries the whole response."""

    def test_done_is_a_done_event_holding_a_response(self) -> None:
        event = StreamAssembler(request()).done()
        assert isinstance(event, DoneEvent)
        assert event.response.latency_ms >= 0

    def test_calling_done_twice_gives_equal_text(self) -> None:
        assembler = StreamAssembler(request())
        assembler.add(TextEvent(text="hello"))
        assert assembler.done().response.text == assembler.done().response.text


class TestMergeOptions:
    """Provider options are merged deeply, and the caller always wins."""

    def test_a_caller_key_overrides_a_computed_one(self) -> None:
        assert merge_options({"temperature": 0.1}, {"temperature": 0.9})["temperature"] == 0.9

    def test_nested_maps_are_merged_not_replaced(self) -> None:
        merged = merge_options(
            {"options": {"num_predict": 10, "temperature": 0.1}}, {"options": {"temperature": 0.9}}
        )
        assert merged["options"] == {"num_predict": 10, "temperature": 0.9}

    def test_a_list_is_replaced_wholesale(self) -> None:
        # Merging lists element by element would be surprising.
        assert merge_options({"stop": ["a", "b"]}, {"stop": ["c"]})["stop"] == ["c"]

    def test_an_empty_options_map_leaves_the_body_alone(self) -> None:
        assert merge_options({"a": 1}, {}) == {"a": 1}

    def test_the_original_body_is_not_mutated(self) -> None:
        body = {"options": {"a": 1}}
        merge_options(body, {"options": {"b": 2}})
        assert body == {"options": {"a": 1}}


class TestAssemblerUnderRandomChunking:
    """A property style check: chunk boundaries must never change the result."""

    @pytest.mark.parametrize("seed", range(12))
    def test_arbitrary_splits_of_the_same_text_give_the_same_answer(self, seed: int) -> None:
        rng = random.Random(seed)
        text = "The capital of Estonia is Tallinn, on the Gulf of Finland."

        pieces: list[str] = []
        rest = text
        while rest:
            cut = rng.randint(1, max(1, min(7, len(rest))))
            pieces.append(rest[:cut])
            rest = rest[cut:]

        assembler = StreamAssembler(request())
        for piece in pieces:
            assembler.add(TextEvent(text=piece))
        assert assembler.done().response.text == text

    @pytest.mark.parametrize("seed", range(12))
    def test_arbitrary_splits_of_tool_json_give_the_same_arguments(self, seed: int) -> None:
        rng = random.Random(seed)
        arguments = {"city": "Tallinn", "units": "c", "days": 3, "detail": {"hourly": True}}
        encoded = json.dumps(arguments)

        assembler = StreamAssembler(request())
        assembler.add(ToolCallStartEvent(id="c1", name="get_weather"))
        rest = encoded
        while rest:
            cut = rng.randint(1, max(1, min(5, len(rest))))
            assembler.add(ToolCallDeltaEvent(id="c1", arguments_json=rest[:cut]))
            rest = rest[cut:]
        for event in assembler.close_tool_calls():
            assembler.add(event)

        assert assembler.done().response.tool_calls[0].arguments == arguments

    @pytest.mark.parametrize("seed", range(8))
    def test_extra_keep_alive_comments_anywhere_change_nothing(self, seed: int) -> None:
        rng = random.Random(seed)
        frames = [f"data: {json.dumps({'i': i})}\n\n" for i in range(6)]
        expected = [m.data for m in decode("".join(frames))]

        noisy: list[str] = []
        for frame in frames:
            if rng.random() < 0.5:
                noisy.append(": keep-alive\n\n")
            noisy.append(frame)
        assert [m.data for m in decode("".join(noisy))] == expected
