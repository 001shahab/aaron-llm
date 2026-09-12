# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Where a request is processed, and whether that satisfies the caller's rule.

The mapping is coarse on purpose. A model's registry ``provider_region`` records
the jurisdiction of the provider's default endpoint, which is the only thing a
client library can honestly know. It is evidence for a compliance review, not a
legal conclusion, and `docs/policy.md` says so.

The rule that matters: an unknown region is rejected, never allowed. A residency
requirement that quietly passed because we had no metadata would be worse than no
requirement at all.
"""

from __future__ import annotations

# Residency values a caller may ask for, and the regions each accepts.
JURISDICTIONS: dict[str, frozenset[str]] = {
    "eu": frozenset({"eu", "eea", "local"}),
    "eea": frozenset({"eu", "eea", "local"}),
    "us": frozenset({"us", "local"}),
    "uk": frozenset({"uk", "local"}),
    "ch": frozenset({"ch", "local"}),
    "local": frozenset({"local"}),
}

# 'local' means the model runs on the caller's own machine, so it satisfies every
# jurisdiction: no personal data crosses any border at all.
ALWAYS_ACCEPTED = frozenset({"local"})


def satisfies(required: str, region: str | None) -> bool:
    """Whether a model's region meets a residency requirement.

    Args:
        required: The residency the policy asks for, for example ``"eu"``.
        region: The model's declared region, or None when the registry has none.

    Returns:
        True when the requirement is met. An unknown region is always False.
    """
    if region is None:
        return False
    accepted = JURISDICTIONS.get(required.strip().lower())
    if accepted is None:
        # An unrecognised requirement is treated as an exact region match, so a
        # caller can use a value we have never heard of by putting the same value
        # in their registry override.
        return region.strip().lower() in {required.strip().lower(), *ALWAYS_ACCEPTED}
    return region.strip().lower() in accepted


def describe(required: str, region: str | None, model: str) -> str:
    """Explain a residency rejection in terms a compliance reader can act on.

    Args:
        required: The residency the policy asks for.
        region: The model's declared region, or None.
        model: The canonical model string.

    Returns:
        A sentence naming the model, the requirement and what to do about it.
    """
    accepted = ", ".join(sorted(JURISDICTIONS.get(required.lower(), frozenset({required}))))
    if region is None:
        return (
            f"{model} has no declared processing region in the model registry, and an "
            f"unknown region is rejected rather than assumed compliant. Residency "
            f"{required!r} accepts: {accepted}. If you know where this endpoint runs, "
            f"declare it with Aaron(model_registry=...) and a provider_region field."
        )
    return (
        f"{model} is processed in region {region!r}, which does not satisfy residency "
        f"{required!r}. Accepted regions: {accepted}."
    )
