"""Tests for the policy layer.

The evaluation order in section 10 is fixed, and these tests pin it: deny, allow,
residency, capabilities, max_input_tokens, max_usd_per_call. The order is what makes a
violation message predictable, and a compliance reviewer reads those messages.

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from aaron import Aaron, Message, Policy
from aaron.errors import ConfigurationError, PolicyViolation
from aaron.policy.redaction import EmailRedactor, IpAddressRedactor, RegexRedactor
from aaron.policy.residency import describe, satisfies
from conftest import load

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/chat"


def client(policy: Policy, **kwargs: object) -> Aaron:
    """A client under one policy, with a key for every remote provider."""
    return Aaron(api_keys={"openai": "k", "anthropic": "k", "google": "k"}, policy=policy, **kwargs)


class TestAllowAndDeny:
    """Glob matching, with only ``*`` special."""

    def test_a_denied_model_is_refused(self) -> None:
        with pytest.raises(PolicyViolation, match="deny"):
            client(Policy(deny=["openai/*"])).chat("openai/gpt-4o", "hi")

    @respx.mock
    def test_a_model_outside_the_allow_list_is_refused(self) -> None:
        with pytest.raises(PolicyViolation, match="allow"):
            client(Policy(allow=["anthropic/*"])).chat("openai/gpt-4o", "hi")

    @respx.mock
    def test_an_allowed_model_goes_through(self) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        assert client(Policy(allow=["openai/gpt-4o"])).chat("openai/gpt-4o", "hi").text

    def test_deny_beats_allow(self) -> None:
        policy = Policy(allow=["openai/*"], deny=["openai/gpt-4o"])
        with pytest.raises(PolicyViolation, match="deny"):
            client(policy).chat("openai/gpt-4o", "hi")

    @respx.mock
    def test_a_star_matches_within_a_segment_only_at_the_pattern_position(self) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        # gpt-4* matches gpt-4o but not o3.
        allowed = client(Policy(allow=["openai/gpt-4*"]))
        assert allowed.chat("openai/gpt-4o", "hi").text
        with pytest.raises(PolicyViolation):
            allowed.chat("openai/o3", "hi")

    def test_a_literal_dot_is_not_a_wildcard(self) -> None:
        # Deliberately not fnmatch: only * is special, so a dot is a dot.
        policy = Policy(deny=["openai/gpt-4."])
        with pytest.raises(PolicyViolation):
            client(policy).chat("openai/gpt-4.", "hi")

    def test_the_violation_names_the_rule_and_the_model(self) -> None:
        with pytest.raises(PolicyViolation) as info:
            client(Policy(deny=["openai/*"])).chat("openai/gpt-4o", "hi")
        assert info.value.rule == "deny"
        assert "openai/gpt-4o" in str(info.value)


class TestResidency:
    """An unknown region is never assumed compliant."""

    def test_a_us_model_fails_an_eu_requirement(self) -> None:
        with pytest.raises(PolicyViolation, match="residency"):
            client(Policy(residency="eu")).chat("openai/gpt-4o", "hi")

    @respx.mock
    def test_a_local_model_satisfies_any_residency(self) -> None:
        # Nothing leaves the machine, so any jurisdiction is satisfied.
        respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=load("ollama_chat")))
        assert client(Policy(residency="eu")).chat("ollama/llama3.1", "hi").text

    def test_an_unknown_region_does_not_satisfy_anything(self) -> None:
        assert satisfies("eu", None) is False
        assert satisfies("eu", "mars") is False

    def test_a_matching_region_satisfies(self) -> None:
        assert satisfies("eu", "eu") is True
        assert satisfies("eu", "local") is True
        assert satisfies("us", "us") is True

    def test_a_us_region_does_not_satisfy_an_eu_requirement(self) -> None:
        assert satisfies("eu", "us") is False

    def test_the_description_is_written_for_a_compliance_reader(self) -> None:
        text = describe("eu", "us", "openai/gpt-4o")
        assert "openai/gpt-4o" in text
        assert "eu" in text and "us" in text

    def test_the_description_says_what_to_do_when_the_region_is_unknown(self) -> None:
        text = describe("eu", None, "openai_compat/x")
        assert "model_registry" in text

    def test_a_model_with_no_region_in_the_registry_is_refused(self) -> None:
        # openai_compat has no declared region because it could be anywhere.
        policy = Policy(residency="eu")
        instance = Aaron(policy=policy, base_urls={"openai_compat": "https://x.example.com/v1"})
        with pytest.raises(PolicyViolation, match="residency"):
            instance.chat("openai_compat/whatever", "hi")


class TestCapabilities:
    """A capability the model does not have is a refusal, not a silent downgrade."""

    def test_a_missing_capability_is_refused(self) -> None:
        with pytest.raises(PolicyViolation, match="vision"):
            client(Policy(require_capabilities=["vision"])).chat("ollama/llama3.1", "hi")

    @respx.mock
    def test_a_present_capability_passes(self) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        assert (
            client(Policy(require_capabilities=["tools", "vision"]))
            .chat("openai/gpt-4o", "hi")
            .text
        )

    def test_an_unknown_capability_name_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError):
            Policy(require_capabilities=["telepathy"])


class TestBudgets:
    """Ceilings are checked before the call, on the worst case."""

    def test_a_call_over_the_cost_ceiling_is_refused(self) -> None:
        with pytest.raises(PolicyViolation, match="max_usd_per_call"):
            client(Policy(max_usd_per_call=0.0000001)).chat("openai/gpt-4o", "hi")

    @respx.mock
    def test_a_call_under_the_ceiling_goes_through(self) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        assert client(Policy(max_usd_per_call=10.0)).chat("openai/gpt-4o", "hi").text

    def test_too_many_input_tokens_is_refused(self) -> None:
        with pytest.raises(PolicyViolation, match="max_input_tokens"):
            client(Policy(max_input_tokens=5)).chat("openai/gpt-4o", "a very long prompt " * 100)

    def test_the_message_says_what_the_estimate_was(self) -> None:
        with pytest.raises(PolicyViolation) as info:
            client(Policy(max_input_tokens=5)).chat("openai/gpt-4o", "x " * 500)
        assert "5" in str(info.value)


class TestEvaluationOrder:
    """Section 10 fixes the order, so a call breaking two rules reports the first."""

    def test_deny_is_reported_before_residency(self) -> None:
        policy = Policy(deny=["openai/*"], residency="eu")
        with pytest.raises(PolicyViolation) as info:
            client(policy).chat("openai/gpt-4o", "hi")
        assert info.value.rule == "deny"

    def test_allow_is_reported_before_residency(self) -> None:
        policy = Policy(allow=["anthropic/*"], residency="eu")
        with pytest.raises(PolicyViolation) as info:
            client(policy).chat("openai/gpt-4o", "hi")
        assert info.value.rule == "allow"

    def test_residency_is_reported_before_capabilities(self) -> None:
        policy = Policy(residency="eu", require_capabilities=["thinking"])
        with pytest.raises(PolicyViolation) as info:
            client(policy).chat("openai/gpt-4o", "hi")
        assert info.value.rule == "residency"

    def test_capabilities_are_reported_before_the_token_ceiling(self) -> None:
        policy = Policy(require_capabilities=["vision"], max_input_tokens=1)
        with pytest.raises(PolicyViolation) as info:
            client(policy).chat("ollama/llama3.1", "x " * 500)
        assert info.value.rule == "capabilities"

    def test_the_token_ceiling_is_reported_before_the_cost_ceiling(self) -> None:
        policy = Policy(max_input_tokens=1, max_usd_per_call=0.0)
        with pytest.raises(PolicyViolation) as info:
            client(policy).chat("openai/gpt-4o", "x " * 500)
        assert info.value.rule == "max_input_tokens"


class TestFallback:
    """Fallback re evaluates every rule for every candidate."""

    @respx.mock
    def test_a_refused_call_falls_back_to_a_compliant_model(self) -> None:
        route = respx.post(OLLAMA_URL).mock(
            return_value=httpx.Response(200, json=load("ollama_chat"))
        )
        policy = Policy(residency="eu", fallback=["ollama/llama3.1"], on_violation="fallback")
        response = client(policy).chat("openai/gpt-4o", "hi")

        assert route.called
        assert response.model == "ollama/llama3.1"

    @respx.mock
    def test_the_fallback_endpoint_is_resolved_for_the_new_provider(self) -> None:
        # A base URL belonging to the refused provider must not follow the fallback.
        route = respx.post(OLLAMA_URL).mock(
            return_value=httpx.Response(200, json=load("ollama_chat"))
        )
        policy = Policy(deny=["openai/*"], fallback=["ollama/llama3.1"], on_violation="fallback")
        client(policy).chat("openai/gpt-4o", "hi")
        assert str(route.calls[0].request.url) == OLLAMA_URL

    def test_a_fallback_that_also_violates_is_refused_with_every_candidate_listed(self) -> None:
        policy = Policy(
            residency="eu", fallback=["anthropic/claude-sonnet-4-5"], on_violation="fallback"
        )
        with pytest.raises(PolicyViolation) as info:
            client(policy).chat("openai/gpt-4o", "hi")

        candidates = [entry[0] for entry in info.value.candidates]
        assert "openai/gpt-4o" in candidates
        assert "anthropic/claude-sonnet-4-5" in candidates

    def test_fallback_without_a_list_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="fallback"):
            Policy(on_violation="fallback")

    def test_an_invalid_on_violation_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="on_violation"):
            Policy(on_violation="explode")  # type: ignore[arg-type]

    @respx.mock
    def test_a_fallback_credential_is_only_needed_for_the_model_actually_used(self) -> None:
        # No OPENAI_API_KEY anywhere, and none is needed, because policy refuses it.
        respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=load("ollama_chat")))
        policy = Policy(residency="eu", fallback=["ollama/llama3.1"], on_violation="fallback")
        assert Aaron(policy=policy).chat("openai/gpt-4o", "hi").text


class TestRedaction:
    """Redactors run on outbound text, before anything is sent."""

    @respx.mock
    def test_an_email_is_replaced_before_the_request_leaves(self) -> None:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        policy = Policy(redactors=[EmailRedactor()])
        client(policy).chat("openai/gpt-4o", "write to shb@3sholding.com about it")

        sent = route.calls[0].request.content.decode()
        assert "shb@3sholding.com" not in sent
        assert "[EMAIL]" in sent

    @respx.mock
    def test_an_ip_address_is_replaced(self) -> None:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        client(Policy(redactors=[IpAddressRedactor()])).chat("openai/gpt-4o", "from 192.168.1.44")
        assert "192.168.1.44" not in route.calls[0].request.content.decode()

    @respx.mock
    def test_a_custom_pattern_works(self) -> None:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        redactor = RegexRedactor(pattern=r"EMP-\d{4}", replacement="[STAFF]", name="employee")
        client(Policy(redactors=[redactor])).chat("openai/gpt-4o", "ticket for EMP-4471")
        assert "EMP-4471" not in route.calls[0].request.content.decode()

    @respx.mock
    def test_redactors_apply_to_every_message_not_just_the_last(self) -> None:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=load("openai_chat"))
        )
        conversation = [
            Message.system("The user is a@b.com"),
            Message.user("and my colleague is c@d.com"),
        ]
        client(Policy(redactors=[EmailRedactor()])).chat("openai/gpt-4o", conversation)
        sent = route.calls[0].request.content.decode()
        assert "a@b.com" not in sent
        assert "c@d.com" not in sent

    @respx.mock
    def test_the_replacement_count_reaches_the_audit_record(self, tmp_path: Path) -> None:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=load("openai_chat")))
        log = tmp_path / "audit.jsonl"
        instance = Aaron(
            api_keys={"openai": "k"}, policy=Policy(redactors=[EmailRedactor()]), audit=log
        )
        instance.chat("openai/gpt-4o", "a@b.com and c@d.com")
        instance.close()

        record = json.loads(log.read_text().splitlines()[0])
        assert record["redactions"] == 2


class TestPolicyFromFile:
    """A policy in version control is the point of the feature."""

    def test_it_loads_from_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "aaron.policy.yaml"
        path.write_text(
            "\n".join(
                [
                    "allow:",
                    "  - anthropic/*",
                    "  - ollama/*",
                    "deny:",
                    "  - openai/*",
                    "residency: eu",
                    "max_usd_per_call: 0.5",
                    "max_input_tokens: 20000",
                    "on_violation: raise",
                    "redactors:",
                    "  - email",
                ]
            ),
            encoding="utf-8",
        )
        policy = Policy.from_file(path)

        assert list(policy.allow) == ["anthropic/*", "ollama/*"]
        assert list(policy.deny) == ["openai/*"]
        assert policy.residency == "eu"
        assert policy.max_usd_per_call == 0.5
        assert policy.max_input_tokens == 20000
        assert [r.name for r in policy.redactors] == ["email"]

    def test_a_missing_file_says_so(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            Policy.from_file(tmp_path / "nope.yaml")

    def test_an_unknown_key_is_refused_rather_than_ignored(self, tmp_path: Path) -> None:
        # A typo in a compliance file must not silently disable a rule.
        path = tmp_path / "p.yaml"
        path.write_text("residencyy: eu\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="residencyy"):
            Policy.from_file(path)

    @respx.mock
    def test_a_policy_file_in_the_working_directory_is_picked_up(self, tmp_path: Path) -> None:
        (tmp_path / "aaron.policy.yaml").write_text("deny:\n  - openai/*\n", encoding="utf-8")
        with pytest.raises(PolicyViolation):
            Aaron(api_keys={"openai": "k"}).chat("openai/gpt-4o", "hi")

    def test_the_environment_variable_wins_over_the_default_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "aaron.policy.yaml").write_text("deny:\n  - anthropic/*\n", encoding="utf-8")
        explicit = tmp_path / "strict.yaml"
        explicit.write_text("deny:\n  - openai/*\n", encoding="utf-8")
        monkeypatch.setenv("AARON_POLICY_FILE", str(explicit))

        instance = Aaron(api_keys={"openai": "k"})
        assert instance.policy is not None
        assert list(instance.policy.deny) == ["openai/*"]


class TestSnapshot:
    """The snapshot goes into every audit record, so it must not carry a pattern."""

    def test_it_lists_redactors_by_name_only(self) -> None:
        redactor = RegexRedactor(pattern=r"SECRET-\d+", replacement="[X]", name="internal")
        snapshot = Policy(redactors=[redactor], deny=["openai/*"]).snapshot()

        assert snapshot["redactors"] == ["internal"]
        assert "SECRET" not in json.dumps(snapshot)

    def test_it_records_the_rules_that_were_in_force(self) -> None:
        snapshot = Policy(residency="eu", max_usd_per_call=1.5, deny=["x/*"]).snapshot()
        assert snapshot["residency"] == "eu"
        assert snapshot["max_usd_per_call"] == 1.5
        assert snapshot["deny"] == ["x/*"]

    def test_it_is_json_serialisable(self) -> None:
        json.dumps(Policy(allow=["a/*"], redactors=[EmailRedactor()]).snapshot())
