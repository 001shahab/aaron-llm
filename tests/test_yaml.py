"""Tests for the bundled YAML subset reader.

This reader is what runs on a default install, because PyYAML is only the optional
``aaron-llm[yaml]`` extra. PyYAML is present in the development environment, so every
test here calls the subset parser directly, and a second class checks that both
backends agree on the file the package actually ships.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

from aaron._yaml import YamlSubsetError, _parse, _significant_lines, load_mapping

SHIPPED = Path(__file__).resolve().parents[1] / "src" / "aaron" / "registry" / "models.yaml"


def subset(text: str) -> dict[str, Any]:
    """Parse with the bundled reader, whether or not PyYAML is installed."""
    return _parse(_significant_lines(text))


@pytest.fixture
def without_pyyaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import yaml`` fail, so ``load_mapping`` takes the bundled path."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ImportError("no yaml for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


class TestScalars:
    """One value at a time."""

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("a: hello", "hello"),
            ("a: 42", 42),
            ("a: -7", -7),
            ("a: 2.5", 2.5),
            ("a: 0.0", 0.0),
            ("a: true", True),
            ("a: false", False),
            ("a: True", True),
            ("a: null", None),
            ("a: ~", None),
            ("a:", None),
            ('a: "quoted"', "quoted"),
            ("a: 'single'", "single"),
            ('a: "2026-09-01"', "2026-09-01"),
            ("a: gpt-4o", "gpt-4o"),
        ],
    )
    def test_a_value(self, source: str, expected: Any) -> None:
        assert subset(source)["a"] == expected

    def test_a_quoted_number_stays_a_string(self) -> None:
        assert subset('a: "42"')["a"] == "42"

    def test_a_colon_inside_a_model_id_is_not_a_separator(self) -> None:
        # "ollama/llama3.1:8b" is a real model name.
        parsed = subset("ollama/llama3.1:8b:\n  context_window: 8192")
        assert parsed["ollama/llama3.1:8b"] == {"context_window": 8192}

    def test_an_escaped_double_quoted_string(self) -> None:
        assert subset('a: "with \\"quotes\\" inside"')["a"] == 'with "quotes" inside'

    def test_a_doubled_single_quote(self) -> None:
        assert subset("a: 'it''s here'")["a"] == "it's here"


class TestStructure:
    """Mappings, nesting and sequences."""

    def test_a_flat_mapping(self) -> None:
        assert subset("a: 1\nb: 2") == {"a": 1, "b": 2}

    def test_a_nested_mapping(self) -> None:
        parsed = subset("openai/gpt-4o:\n  context_window: 128000\n  provider_region: us")
        assert parsed == {"openai/gpt-4o": {"context_window": 128000, "provider_region": "us"}}

    def test_two_nested_mappings(self) -> None:
        parsed = subset("a:\n  x: 1\nb:\n  y: 2")
        assert parsed == {"a": {"x": 1}, "b": {"y": 2}}

    def test_a_block_sequence(self) -> None:
        assert subset("deny:\n  - openai/*\n  - google/*") == {"deny": ["openai/*", "google/*"]}

    def test_a_block_sequence_at_the_same_indentation_as_its_key(self) -> None:
        # Valid YAML, and how most people write a top level list.
        assert subset("deny:\n- openai/*\n- google/*") == {"deny": ["openai/*", "google/*"]}

    def test_an_inline_sequence(self) -> None:
        assert subset("capabilities: [tools, vision, streaming]") == {
            "capabilities": ["tools", "vision", "streaming"]
        }

    def test_an_inline_sequence_of_quoted_strings(self) -> None:
        assert subset('a: ["one", "two"]')["a"] == ["one", "two"]

    def test_an_empty_inline_sequence(self) -> None:
        assert subset("a: []")["a"] == []

    def test_an_inline_mapping(self) -> None:
        assert subset('a: {"x": 1, "y": 2}')["a"] == {"x": 1, "y": 2}

    def test_a_key_with_an_empty_block_is_none(self) -> None:
        assert subset("a:\nb: 1") == {"a": None, "b": 1}


class TestNoise:
    """Comments, blank lines and document markers."""

    def test_comments_are_dropped(self) -> None:
        assert subset("# a comment\na: 1  # trailing\n") == {"a": 1}

    def test_a_hash_inside_a_quoted_string_is_kept(self) -> None:
        assert subset('a: "value # not a comment"')["a"] == "value # not a comment"

    def test_a_hash_with_no_preceding_space_is_kept(self) -> None:
        assert subset("a: colour#123")["a"] == "colour#123"

    def test_blank_lines_are_ignored(self) -> None:
        assert subset("\n\na: 1\n\n\nb: 2\n\n") == {"a": 1, "b": 2}

    def test_document_markers_are_ignored(self) -> None:
        assert subset("---\na: 1\n...\n") == {"a": 1}

    def test_an_empty_document_is_an_empty_mapping(self) -> None:
        assert subset("") == {}
        assert subset("# only a comment\n") == {}


class TestRefusals:
    """What it will not do, it says so about, with a line number."""

    def test_tab_indentation(self) -> None:
        with pytest.raises(YamlSubsetError, match="tab") as info:
            subset("a:\n\tb: 1")
        assert info.value.line_number == 2

    def test_deeper_than_one_level_of_nesting(self) -> None:
        with pytest.raises(YamlSubsetError, match="one level"):
            subset("a:\n  b:\n    c: 1")

    def test_a_block_scalar(self) -> None:
        with pytest.raises(YamlSubsetError, match="block scalars"):
            subset("a: |\n  some text")

    def test_an_anchor(self) -> None:
        with pytest.raises(YamlSubsetError, match="anchors"):
            subset("a: &anchor value")

    def test_a_line_that_is_not_key_value(self) -> None:
        with pytest.raises(YamlSubsetError, match="key: value"):
            subset("just a bare line")

    def test_the_error_names_the_extra_that_fixes_it(self) -> None:
        with pytest.raises(YamlSubsetError, match=r"aaron-llm\[yaml\]"):
            subset("a: |\n  text")


class TestLoadMappingWithoutPyyaml:
    """``load_mapping`` must work when the optional extra is absent."""

    def test_it_falls_back_to_the_bundled_reader(self, without_pyyaml: None) -> None:
        assert load_mapping("a: 1\nb:\n  c: 2") == {"a": 1, "b": {"c": 2}}

    def test_the_shipped_registry_file_loads(self, without_pyyaml: None) -> None:
        parsed = load_mapping(SHIPPED.read_text(encoding="utf-8"))
        assert "openai/gpt-4o" in parsed
        assert parsed["openai/gpt-4o"]["input_usd_per_mtok"] == 2.5

    def test_a_policy_file_loads(self, without_pyyaml: None) -> None:
        source = "\n".join(
            [
                "allow:",
                "  - anthropic/*",
                "deny:",
                "  - openai/*",
                "residency: eu",
                "max_usd_per_call: 0.5",
                "on_violation: raise",
            ]
        )
        assert load_mapping(source) == {
            "allow": ["anthropic/*"],
            "deny": ["openai/*"],
            "residency": "eu",
            "max_usd_per_call": 0.5,
            "on_violation": "raise",
        }

    def test_the_registry_still_builds_with_no_yaml_installed(self, without_pyyaml: None) -> None:
        from aaron.registry import Registry

        registry = Registry.from_sources(SHIPPED)
        assert registry.lookup("openai/gpt-4o").input_usd_per_mtok == 2.5
        assert registry.region("ollama/llama3.1") == "local"


class TestBothBackendsAgree:
    """The shipped file must parse identically either way, field for field."""

    def test_pyyaml_is_available_in_the_development_environment(self) -> None:
        # If this ever fails, the comparison below is vacuous and must be fixed.
        pytest.importorskip("yaml")

    def test_every_entry_matches(self) -> None:
        pytest.importorskip("yaml")
        import yaml

        source = SHIPPED.read_text(encoding="utf-8")
        theirs = yaml.safe_load(source)
        ours = subset(source)

        assert set(ours) == set(theirs)
        for key in theirs:
            assert ours[key] == theirs[key], key

    def test_dates_are_strings_in_both(self) -> None:
        pytest.importorskip("yaml")
        import yaml

        source = SHIPPED.read_text(encoding="utf-8")
        for key, entry in yaml.safe_load(source).items():
            if "last_verified" in entry:
                assert isinstance(entry["last_verified"], str), key
                assert isinstance(subset(source)[key]["last_verified"], str), key

    def test_a_mapping_document_is_required(self) -> None:
        with pytest.raises(YamlSubsetError, match="expected a mapping"):
            load_mapping("- just\n- a list\n")

    def test_a_blank_document_is_empty_with_either_backend(self, without_pyyaml: None) -> None:
        assert load_mapping("") == {}
