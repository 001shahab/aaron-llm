"""Tests for the error hierarchy, the transport layer and provider registration.

The hierarchy is the part of the API callers write `except` clauses against, so the
inheritance relationships are pinned here: catching a base class must keep catching
everything it caught when it was written.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import httpx
import pytest

from aaron.errors import (
    AaronError,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ContentFilterError,
    ContextLengthExceeded,
    InvalidRequest,
    LocalProviderUnavailable,
    MissingAPIKey,
    PermissionError,
    PolicyViolation,
    ProviderError,
    RateLimitError,
    RequestError,
    ServerError,
    ServiceOverloaded,
    TimeoutError,
    ToolArgumentError,
    TransportError,
    UnknownModel,
    UnknownProvider,
)
from aaron.providers import (
    Provider,
    available_providers,
    get_provider,
    register_provider,
)
from aaron.transport import TransportConfig, translate_httpx_error


class TestHierarchy:
    """Section 12, as inheritance assertions."""

    @pytest.mark.parametrize(
        ("child", "parent"),
        [
            (ConfigurationError, AaronError),
            (MissingAPIKey, ConfigurationError),
            (UnknownProvider, ConfigurationError),
            (UnknownModel, ConfigurationError),
            (PolicyViolation, AaronError),
            (RequestError, AaronError),
            (InvalidRequest, RequestError),
            (ContextLengthExceeded, RequestError),
            (ToolArgumentError, RequestError),
            (ProviderError, AaronError),
            (AuthenticationError, ProviderError),
            (PermissionError, ProviderError),
            (RateLimitError, ProviderError),
            (ServerError, ProviderError),
            (ServiceOverloaded, ProviderError),
            (ContentFilterError, ProviderError),
            (TransportError, AaronError),
            (TimeoutError, TransportError),
            (ConnectionError, TransportError),
            (LocalProviderUnavailable, ConnectionError),
        ],
    )
    def test_inheritance(self, child: type[Exception], parent: type[Exception]) -> None:
        assert issubclass(child, parent)

    def test_everything_is_catchable_as_one_class(self) -> None:
        for cls in (InvalidRequest, RateLimitError, TimeoutError, PolicyViolation):
            with pytest.raises(AaronError):
                raise cls("x")

    def test_the_three_shadowed_names_shadow_by_name_only(self) -> None:
        # TimeoutError, ConnectionError and PermissionError reuse familiar names, but
        # they are not the builtins and do not inherit from them. Catching the builtin
        # will not catch ours, which is why they all descend from AaronError instead.
        import builtins

        for ours, builtin in (
            (TimeoutError, builtins.TimeoutError),
            (ConnectionError, builtins.ConnectionError),
            (PermissionError, builtins.PermissionError),
        ):
            assert ours is not builtin
            assert not issubclass(ours, builtin)
            assert issubclass(ours, AaronError)


class TestContext:
    """Every error carries the context needed to act on it."""

    def test_the_fields_default_to_none(self) -> None:
        error = AaronError("something went wrong")
        assert error.provider is None
        assert error.model is None
        assert error.status_code is None
        assert error.request_id is None

    def test_the_message_includes_the_context_that_is_known(self) -> None:
        error = ProviderError("boom", provider="openai", model="gpt-4o", status_code=500)
        text = str(error)
        assert "boom" in text
        assert "openai" in text
        assert "gpt-4o" in text
        assert "500" in text

    def test_a_message_with_no_context_is_left_alone(self) -> None:
        assert str(AaronError("just the message")) == "just the message"

    def test_the_raw_payload_is_kept_for_inspection(self) -> None:
        error = ProviderError("boom", raw={"error": {"code": "overloaded"}})
        assert error.raw == {"error": {"code": "overloaded"}}

    def test_a_rate_limit_carries_retry_after(self) -> None:
        assert RateLimitError("slow", retry_after=30.0).retry_after == 30.0

    def test_a_tool_argument_error_carries_the_raw_text(self) -> None:
        error = ToolArgumentError("bad json", tool_name="get_weather", raw_arguments="{oops")
        assert error.tool_name == "get_weather"
        assert error.raw_arguments == "{oops"

    def test_a_policy_violation_carries_structured_detail(self) -> None:
        error = PolicyViolation(
            "refused",
            rule="residency",
            detail="us does not satisfy eu",
            candidates=[("openai/gpt-4o", "residency", "us does not satisfy eu")],
        )
        assert error.rule == "residency"
        assert error.detail == "us does not satisfy eu"
        assert error.candidates[0][0] == "openai/gpt-4o"

    def test_the_repr_is_safe_to_log(self) -> None:
        error = ProviderError("boom", provider="openai", model="gpt-4o")
        assert "ProviderError" in repr(error)
        assert "openai" in repr(error)


class TestTranslateHttpxError:
    """httpx exceptions become our classes, so callers never import httpx."""

    @pytest.mark.parametrize(
        ("exception", "expected"),
        [
            (httpx.ReadTimeout("slow"), TimeoutError),
            (httpx.ConnectTimeout("slow"), TimeoutError),
            (httpx.WriteTimeout("slow"), TimeoutError),
            (httpx.PoolTimeout("slow"), TimeoutError),
            (httpx.ConnectError("refused"), ConnectionError),
            (httpx.ReadError("dropped"), TransportError),
            (httpx.RemoteProtocolError("bad frame"), TransportError),
        ],
    )
    def test_mapping(self, exception: Exception, expected: type[Exception]) -> None:
        error = translate_httpx_error(
            exception, provider="openai", model="openai/gpt-4o", base_url="https://x.test"
        )
        assert isinstance(error, expected)

    def test_a_local_provider_gets_a_more_useful_connection_error(self) -> None:
        error = translate_httpx_error(
            httpx.ConnectError("refused"),
            provider="ollama",
            model="ollama/llama3.1",
            base_url="http://localhost:11434",
            local=True,
        )
        assert isinstance(error, LocalProviderUnavailable)
        assert "ollama" in str(error).lower()

    def test_a_remote_connection_failure_is_not_reported_as_a_local_one(self) -> None:
        error = translate_httpx_error(
            httpx.ConnectError("refused"),
            provider="openai",
            model="openai/gpt-4o",
            base_url="https://api.openai.com/v1",
        )
        assert not isinstance(error, LocalProviderUnavailable)

    def test_the_provider_name_is_attached(self) -> None:
        error = translate_httpx_error(
            httpx.ReadTimeout("slow"),
            provider="google",
            model="google/gemini-2.5-flash",
            base_url="https://x.test",
        )
        assert error.provider == "google"


class TestTransportConfig:
    """Timeouts and client construction."""

    def test_the_documented_defaults(self) -> None:
        config = TransportConfig()
        assert config.timeout == 60.0
        assert config.connect_timeout == 10.0

    def test_an_override_produces_a_new_config(self) -> None:
        config = TransportConfig(timeout=60.0)
        overridden = config.with_overrides(timeout=5.0)
        assert overridden.timeout == 5.0
        assert config.timeout == 60.0, "the original must not be mutated"

    def test_an_override_of_none_keeps_the_original(self) -> None:
        assert TransportConfig(timeout=30.0).with_overrides(timeout=None).timeout == 30.0

    def test_redirects_are_not_followed(self) -> None:
        # A redirect could send a credential to a host the caller never named.
        assert TransportConfig().client_kwargs()["follow_redirects"] is False

    def test_headers_are_passed_through(self) -> None:
        kwargs = TransportConfig(headers={"x-tenant": "acme"}).client_kwargs()
        assert kwargs["headers"]["x-tenant"] == "acme"


class TestProviderRegistration:
    """Providers load lazily and can be extended."""

    def test_the_builtins_are_listed(self) -> None:
        assert "openai" in available_providers()

    def test_get_provider_is_cached(self) -> None:
        assert get_provider("openai") is get_provider("openai")

    def test_an_unknown_name_lists_the_alternatives(self) -> None:
        with pytest.raises(UnknownProvider) as info:
            get_provider("not-a-provider")
        message = str(info.value)
        assert "not-a-provider" in message
        for name in ("openai", "anthropic", "google", "ollama"):
            assert name in message

    def test_a_third_party_provider_can_be_registered(self) -> None:
        from aaron.providers.openai import OpenAIProvider

        class MyProvider(OpenAIProvider):
            """A provider added from outside the package."""

            name = "my_gateway"
            default_base_url = "https://gateway.test/v1"
            env_key = "MY_GATEWAY_API_KEY"

        register_provider(MyProvider())
        try:
            assert get_provider("my_gateway").default_base_url == "https://gateway.test/v1"
            assert "my_gateway" in available_providers()
        finally:
            from aaron.providers import _instances

            _instances.pop("my_gateway", None)

    def test_every_builtin_satisfies_the_protocol_at_runtime(self) -> None:
        for name in available_providers():
            assert isinstance(get_provider(name), Provider)

    def test_a_local_provider_says_so(self) -> None:
        assert get_provider("ollama").local is True
        assert get_provider("openai").local is False

    def test_only_the_providers_that_need_a_key_ask_for_one(self) -> None:
        assert get_provider("openai").requires_key is True
        assert get_provider("anthropic").requires_key is True
        assert get_provider("google").requires_key is True
        assert get_provider("ollama").requires_key is False
        assert get_provider("openai_compat").requires_key is False


class TestRedactionEdges:
    """Patterns a compliance officer will try."""

    def test_an_email_with_a_plus_and_a_subdomain(self) -> None:
        from aaron.policy.redaction import EmailRedactor

        redactor = EmailRedactor()
        text, count = redactor.redact("write to first.last+tag@mail.example.co.uk now")
        assert count == 1
        assert "@" not in text

    def test_several_matches_are_all_replaced_and_counted(self) -> None:
        from aaron.policy.redaction import EmailRedactor

        text, count = EmailRedactor().redact("a@b.com and c@d.org and e@f.net")
        assert count == 3
        assert "[EMAIL]" in text

    def test_text_with_nothing_to_redact_is_returned_unchanged(self) -> None:
        from aaron.policy.redaction import EmailRedactor

        text, count = EmailRedactor().redact("nothing personal here")
        assert (text, count) == ("nothing personal here", 0)

    def test_ipv4_and_ipv6(self) -> None:
        from aaron.policy.redaction import IpAddressRedactor

        redactor = IpAddressRedactor()
        assert redactor.redact("from 10.0.0.1")[1] == 1
        assert redactor.redact("from 2001:0db8:85a3:0000:0000:8a2e:0370:7334")[1] == 1

    def test_a_named_redactor_can_be_built_from_a_string(self) -> None:
        from aaron.policy.redaction import build_redactor

        assert build_redactor("email").name == "email"
        assert build_redactor("ip").name == "ip"

    def test_a_mapping_builds_a_custom_pattern(self) -> None:
        from aaron.policy.redaction import build_redactor

        redactor = build_redactor(
            {"name": "case_id", "pattern": r"CASE-\d+", "replacement": "[CASE]"}
        )
        assert redactor.redact("see CASE-99")[0] == "see [CASE]"

    def test_an_unknown_name_is_a_configuration_error(self) -> None:
        from aaron.policy.redaction import build_redactor

        with pytest.raises(ConfigurationError):
            build_redactor("telephone-numbers-maybe")

    def test_an_invalid_regex_is_reported_as_configuration(self) -> None:
        from aaron.policy.redaction import build_redactor

        with pytest.raises(ConfigurationError):
            build_redactor({"name": "bad", "pattern": "([unclosed"})
