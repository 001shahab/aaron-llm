"""Tests for the model registry and the cost calculation.

The registry is the only place a price appears, and a wrong price is a wrong invoice,
so the arithmetic is checked against hand computed numbers rather than against itself.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aaron.errors import UnknownModel
from aaron.registry import (
    Capabilities,
    ModelInfo,
    Registry,
    default_registry,
    glob_match,
)
from aaron.types import Usage


class TestGlobMatch:
    """Only ``*`` is special, deliberately not fnmatch."""

    @pytest.mark.parametrize(
        ("pattern", "value", "expected"),
        [
            ("openai/gpt-4o", "openai/gpt-4o", True),
            ("openai/*", "openai/gpt-4o", True),
            ("openai/*", "anthropic/claude", False),
            ("*", "anything/at-all", True),
            ("openai/gpt-4*", "openai/gpt-4o-mini", True),
            ("openai/gpt-4*", "openai/o3", False),
            ("*/gpt-4o", "openai/gpt-4o", True),
            ("openai/gpt-4.", "openai/gpt-4x", False),
            ("openai/gpt-4?", "openai/gpt-4o", False),
            ("openai/gpt[34]", "openai/gpt3", False),
        ],
    )
    def test_matching(self, pattern: str, value: str, expected: bool) -> None:
        assert glob_match(value, pattern) is expected

    def test_a_question_mark_and_brackets_are_literal(self) -> None:
        # fnmatch would treat these as wildcards. A model name is not a filename.
        assert glob_match("openai/gpt-4?", "openai/gpt-4?") is True
        assert glob_match("openai/gpt[34]", "openai/gpt[34]") is True


class TestLookup:
    """Exact match first, then the longest matching glob."""

    def test_an_exact_entry_wins_over_a_glob(self) -> None:
        registry = Registry.from_sources(
            {
                "openai/*": {"input_usd_per_mtok": 100.0, "output_usd_per_mtok": 100.0},
                "openai/gpt-4o": {"input_usd_per_mtok": 2.5, "output_usd_per_mtok": 10.0},
            }
        )
        assert registry.lookup("openai/gpt-4o").input_usd_per_mtok == 2.5

    def test_the_longest_glob_wins(self) -> None:
        registry = Registry.from_sources(
            {
                "openai/*": {"input_usd_per_mtok": 1.0},
                "openai/gpt-4*": {"input_usd_per_mtok": 2.0},
            }
        )
        assert registry.lookup("openai/gpt-4o").input_usd_per_mtok == 2.0

    def test_an_unknown_model_falls_back_permissively(self) -> None:
        # A model we have never heard of must still be callable.
        info = Registry.from_sources({}).lookup("mystery/model-9")
        assert isinstance(info, ModelInfo)
        assert info.input_usd_per_mtok is None

    def test_the_fallback_warns_once_per_model(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = Registry.from_sources({})
        with caplog.at_level("WARNING", logger="aaron"):
            for _ in range(3):
                registry.lookup("mystery/model-9")
        assert caplog.text.count("mystery/model-9") == 1

    def test_the_shipped_registry_knows_the_documented_models(self) -> None:
        registry = default_registry()
        for model in (
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/o3",
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-4-5",
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            "ollama/llama3.1",
        ):
            assert registry.lookup(model).input_usd_per_mtok is not None or model.startswith(
                "ollama/"
            )


class TestCost:
    """Arithmetic against hand computed values."""

    def test_a_priced_model_is_exact(self) -> None:
        registry = Registry.from_sources(
            {"x/y": {"input_usd_per_mtok": 2.5, "output_usd_per_mtok": 10.0}}
        )
        # 1000 in at $2.50/M = $0.0025, 500 out at $10/M = $0.005.
        cost = registry.cost("x/y", Usage(input_tokens=1000, output_tokens=500))
        assert cost.usd == pytest.approx(0.0075)
        assert cost.estimated is False

    def test_a_cached_input_token_is_charged_at_its_own_rate(self) -> None:
        registry = Registry.from_sources(
            {
                "x/y": {
                    "input_usd_per_mtok": 10.0,
                    "cached_input_usd_per_mtok": 1.0,
                    "output_usd_per_mtok": 0.0,
                }
            }
        )
        # 1000 input of which 800 cached: 200 at $10/M plus 800 at $1/M.
        cost = registry.cost(
            "x/y", Usage(input_tokens=1000, cached_input_tokens=800, output_tokens=0)
        )
        assert cost.usd == pytest.approx(0.002 + 0.0008)

    def test_an_unpriced_model_is_zero_and_says_it_is_an_estimate(self) -> None:
        cost = Registry.from_sources({"x/y": {}}).cost("x/y", Usage(input_tokens=10))
        assert cost.usd == 0.0
        assert cost.estimated is True

    def test_a_local_model_is_free_and_not_an_estimate(self) -> None:
        # Zero is the true price of running on your own hardware, not a guess.
        cost = default_registry().cost(
            "ollama/llama3.1", Usage(input_tokens=1000, output_tokens=50)
        )
        assert cost.usd == 0.0
        assert cost.estimated is False

    def test_the_worst_case_estimate_uses_max_tokens(self) -> None:
        registry = Registry.from_sources(
            {"x/y": {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 1000.0}}
        )
        estimate = registry.estimate_cost("x/y", input_tokens=1000, max_output_tokens=1000)
        assert estimate == pytest.approx(0.001 + 1.0)


class TestCapabilities:
    """Capability flags default to conservative values."""

    def test_defaults_are_permissive_so_an_undescribed_model_is_never_blocked(self) -> None:
        # A registry entry with no capability list must not accidentally refuse a
        # call. Thinking is the exception: claiming it would change request shape.
        capabilities = Capabilities()
        assert capabilities.streaming is True
        assert capabilities.tools is True
        assert capabilities.vision is True
        assert capabilities.thinking is False

    def test_supports_reads_by_name(self) -> None:
        capabilities = Capabilities(tools=True, vision=False)
        assert capabilities.supports("tools") is True
        assert capabilities.supports("vision") is False

    def test_an_unknown_name_is_not_supported(self) -> None:
        assert Capabilities(tools=True).supports("telepathy") is False

    def test_names_lists_what_is_on(self) -> None:
        off = dict.fromkeys(("documents", "json_schema", "json_mode", "streaming"), False)
        assert Capabilities(tools=True, vision=True, **off).names() == ["tools", "vision"]


class TestRegion:
    """Residency depends on this, so an absent region must stay absent."""

    def test_the_shipped_entries_declare_a_region(self) -> None:
        registry = default_registry()
        assert registry.region("openai/gpt-4o") == "us"
        assert registry.region("ollama/llama3.1") == "local"

    def test_openai_compat_has_no_region_because_it_could_be_anywhere(self) -> None:
        assert default_registry().region("openai_compat/whatever") is None


class TestOverrides:
    """A caller can correct or extend the registry without editing the package."""

    def test_a_mapping_override_wins(self) -> None:
        registry = default_registry().merge({"openai/gpt-4o": {"input_usd_per_mtok": 99.0}})
        assert registry.lookup("openai/gpt-4o").input_usd_per_mtok == 99.0

    def test_an_override_only_changes_the_fields_it_names(self) -> None:
        original = default_registry().lookup("openai/gpt-4o")
        merged = default_registry().merge({"openai/gpt-4o": {"input_usd_per_mtok": 99.0}})
        assert merged.lookup("openai/gpt-4o").output_usd_per_mtok == original.output_usd_per_mtok

    def test_a_new_model_can_be_added(self) -> None:
        registry = default_registry().merge(
            {
                "openai_compat/my-model": {
                    "input_usd_per_mtok": 0.2,
                    "provider_region": "eu",
                    "capabilities": ["tools", "streaming"],
                }
            }
        )
        info = registry.lookup("openai_compat/my-model")
        assert info.provider_region == "eu"
        assert info.to_capabilities().tools is True
        assert info.to_capabilities().vision is False

    def test_a_yaml_file_can_be_used(self, tmp_path: Path) -> None:
        path = tmp_path / "prices.yaml"
        path.write_text(
            "\n".join(
                [
                    "openai_compat/local-llama:",
                    "  input_usd_per_mtok: 0.0",
                    "  output_usd_per_mtok: 0.0",
                    "  provider_region: eu",
                    "  capabilities:",
                    "    - tools",
                    "    - streaming",
                ]
            ),
            encoding="utf-8",
        )
        registry = default_registry().merge(path)
        assert registry.region("openai_compat/local-llama") == "eu"
        assert registry.capabilities("openai_compat/local-llama").tools is True

    def test_the_default_registry_is_not_mutated_by_a_merge(self) -> None:
        default_registry().merge({"openai/gpt-4o": {"input_usd_per_mtok": 99.0}})
        assert default_registry().lookup("openai/gpt-4o").input_usd_per_mtok != 99.0


class TestShippedFile:
    """models.yaml is data a human maintains, so its shape is checked."""

    def test_every_entry_parses_with_the_bundled_reader(self) -> None:
        assert len(default_registry()) >= 20

    def test_every_priced_entry_has_a_verification_date_as_a_string(self) -> None:
        # pyyaml would turn an unquoted date into a date object, so they are quoted.
        # The catch all globs carry no price and so need no date.
        for name, info in dict(default_registry()).items():
            if not info.priced:
                continue
            assert isinstance(info.last_verified, str), name
            assert len(info.last_verified) == 10, name

    def test_every_priced_entry_has_both_directions(self) -> None:
        for name, info in dict(default_registry()).items():
            if info.input_usd_per_mtok:
                assert info.output_usd_per_mtok is not None, name

    def test_every_provider_has_a_catch_all_entry(self) -> None:
        models = dict(default_registry())
        for provider in ("openai", "anthropic", "google", "ollama", "openai_compat"):
            assert f"{provider}/*" in models

    def test_every_ollama_entry_is_local_and_free(self) -> None:
        for name, info in dict(default_registry()).items():
            if not name.startswith("ollama/"):
                continue
            assert info.provider_region == "local", name
            assert info.input_usd_per_mtok in (0.0, None), name

    def test_vision_is_claimed_only_where_it_exists(self) -> None:
        registry = default_registry()
        assert registry.capabilities("ollama/llava").vision is True
        assert registry.capabilities("ollama/llama3.1").vision is False


class TestCheckCommand:
    """``python -m aaron.registry --check`` prints prices for review."""

    def test_it_prints_every_price(self, capsys: pytest.CaptureFixture[str]) -> None:
        from aaron.registry import main

        main(["--check"])
        output = capsys.readouterr().out
        assert "openai/gpt-4o" in output
        assert "verified" in output

    def test_it_can_print_one_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        from aaron.registry import main

        assert main(["--check", "openai/gpt-4o"]) == 0
        output = capsys.readouterr().out
        assert "openai/gpt-4o" in output
        assert "anthropic" not in output

    def test_an_unknown_model_is_reported_by_name(self, capsys: pytest.CaptureFixture[str]) -> None:
        from aaron.registry import main

        assert main(["--check", "nope/nothing"]) == 2
        assert "nope/nothing" in capsys.readouterr().err

    def test_require_refuses_to_fall_back(self) -> None:
        registry = Registry.from_sources({"x/y": {}})
        with pytest.raises(UnknownModel, match="nope/nothing"):
            registry.require("nope/nothing")
        assert registry.require("x/y").known is True
