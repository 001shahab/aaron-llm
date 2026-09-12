# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Model metadata: context windows, prices, capabilities and data residency.

An unknown model never raises. It resolves to permissive capabilities, zero cost
marked as estimated, and a single warning per model per process. That keeps a new
model release from breaking a caller who upgraded their provider but not Aaron.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ._yaml import load_mapping
from .errors import UnknownModel
from .types import Cost, Usage

log = logging.getLogger("aaron")

_SHIPPED = Path(__file__).with_name("models.yaml")

CAPABILITY_NAMES = frozenset(
    {"tools", "vision", "documents", "json_schema", "json_mode", "streaming", "thinking"}
)


class Capabilities(BaseModel):
    """What a model can do, as far as the registry knows."""

    model_config = ConfigDict(frozen=True)

    tools: bool = True
    vision: bool = True
    documents: bool = True
    json_schema: bool = True
    json_mode: bool = True
    streaming: bool = True
    thinking: bool = False

    def supports(self, name: str) -> bool:
        """Return whether a named capability is available.

        Args:
            name: A capability name such as ``"tools"``.

        Returns:
            True when supported. An unrecognised name is reported as unsupported.
        """
        value = getattr(self, name, None)
        return value is True

    def names(self) -> list[str]:
        """The supported capability names, sorted."""
        return sorted(n for n in CAPABILITY_NAMES if self.supports(n))


class ModelInfo(BaseModel):
    """One registry entry.

    ``known`` is False for the permissive fallback used for models we have no
    metadata for. Prices are per million tokens, in US dollars.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    context_window: int | None = None
    max_output_tokens: int | None = None
    input_usd_per_mtok: float | None = None
    output_usd_per_mtok: float | None = None
    cached_input_usd_per_mtok: float | None = None
    capabilities: tuple[str, ...] = ()
    provider_region: str | None = None
    last_verified: str | None = None
    known: bool = True
    pattern: str | None = Field(default=None, description="registry key this entry was matched by")

    @property
    def priced(self) -> bool:
        """Whether the entry carries enough pricing to compute a real cost."""
        return self.input_usd_per_mtok is not None and self.output_usd_per_mtok is not None

    def to_capabilities(self) -> Capabilities:
        """Turn the capability name list into a :class:`Capabilities` object.

        An entry with no capability list is treated as permissive, so that an
        unknown or partially described model is never blocked by accident.
        """
        if not self.capabilities:
            return Capabilities()
        listed = {name.strip().lower() for name in self.capabilities}
        unknown = listed - CAPABILITY_NAMES
        if unknown:
            log.warning("ignoring unrecognised capabilities in registry: %s", sorted(unknown))
        return Capabilities(**{name: name in listed for name in CAPABILITY_NAMES})


FALLBACK = ModelInfo(known=False, pattern=None)


class Registry:
    """A model metadata lookup table, built from one or more YAML sources."""

    def __init__(self, entries: Mapping[str, ModelInfo]) -> None:
        self._entries: dict[str, ModelInfo] = dict(entries)
        self._warned: set[str] = set()

    @classmethod
    def from_sources(cls, *sources: str | Path | Mapping[str, Any] | None) -> Registry:
        """Build a registry by merging sources left to right, later winning.

        Args:
            sources: Paths to YAML files, or already parsed mappings. ``None`` is
                skipped so callers can pass an optional override directly.

        Returns:
            A new registry.

        Raises:
            FileNotFoundError: A path was given that does not exist.
        """
        registry = cls({})
        for source in sources:
            registry = registry.merge(source)
        return registry

    def merge(self, source: str | Path | Mapping[str, Any] | None) -> Registry:
        """Return a new registry with ``source`` layered over this one."""
        if source is None:
            return self
        merged = dict(self._entries)
        for key, value in _read_source(source).items():
            merged[key] = _build_entry(key, value, existing=merged.get(key))
        return Registry(merged)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[tuple[str, ModelInfo]]:
        yield from sorted(self._entries.items())

    def lookup(self, model: str) -> ModelInfo:
        """Find metadata for a canonical ``provider/model`` string.

        Matching is exact first, then by the longest glob key that matches, so
        ``openai/gpt-4o-2024-11-20`` can be covered by ``openai/gpt-4o*``.

        Args:
            model: The canonical model string.

        Returns:
            The entry, or a permissive fallback whose ``known`` is False.
        """
        exact = self._entries.get(model)
        if exact is not None:
            return exact

        best: tuple[int, str, ModelInfo] | None = None
        for key, entry in self._entries.items():
            if "*" not in key or not glob_match(model, key):
                continue
            specificity = len(key.replace("*", ""))
            if best is None or specificity > best[0]:
                best = (specificity, key, entry)
        if best is not None:
            return best[2]

        if model not in self._warned:
            self._warned.add(model)
            log.warning(
                "model %r is not in the registry: capabilities are permissive and "
                "cost is reported as estimated. Add it with Aaron(model_registry=...).",
                model,
            )
        return FALLBACK

    def capabilities(self, model: str) -> Capabilities:
        """Capabilities for a model, permissive when it is unknown."""
        return self.lookup(model).to_capabilities()

    def region(self, model: str) -> str | None:
        """The declared processing region for a model, or None when unknown."""
        return self.lookup(model).provider_region

    def require(self, model: str) -> ModelInfo:
        """Look a model up, refusing to fall back.

        Args:
            model: A canonical ``provider/model`` string.

        Returns:
            The entry.

        Raises:
            UnknownModel: The registry has no entry, exact or glob, for this model.
        """
        info = self.lookup(model)
        if not info.known:
            raise UnknownModel(
                f"{model} is not in the model registry. Add it with "
                f"Aaron(model_registry=...) or check the spelling.",
                model=model,
            )
        return info

    def cost(self, model: str, usage: Usage, *, estimated: bool = False) -> Cost:
        """Price a call from its token counts.

        Args:
            model: Canonical model string.
            usage: Token counts for the call.
            estimated: Set by the caller when the token counts themselves were a
                guess, for example a stream that never reported usage.

        Returns:
            The cost. ``estimated`` is True when the model has no price in the
            registry or when the caller flagged the usage as a guess.
        """
        entry = self.lookup(model)
        if not entry.priced:
            return Cost(usd=0.0, estimated=True)

        input_rate = entry.input_usd_per_mtok or 0.0
        output_rate = entry.output_usd_per_mtok or 0.0
        cached_rate = entry.cached_input_usd_per_mtok
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        fresh = usage.input_tokens - cached

        usd = (fresh * input_rate + usage.output_tokens * output_rate) / 1_000_000
        usd += cached * (cached_rate if cached_rate is not None else input_rate) / 1_000_000
        return Cost(usd=round(usd, 8), estimated=estimated)

    def estimate_cost(self, model: str, input_tokens: int, max_output_tokens: int | None) -> float:
        """Worst case dollar cost of a call, used by the policy budget check.

        Args:
            model: Canonical model string.
            input_tokens: Estimated input tokens.
            max_output_tokens: The output ceiling, or None when the caller set no
                ceiling, in which case the model's own maximum is assumed.

        Returns:
            The upper bound in US dollars. Zero when the model has no price, since
            an unpriced model must not be blocked by a cost budget.
        """
        entry = self.lookup(model)
        if not entry.priced:
            return 0.0
        ceiling = max_output_tokens or entry.max_output_tokens or 4096
        return self.cost(model, Usage(input_tokens=input_tokens, output_tokens=ceiling)).usd


def _read_source(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return {str(key): value for key, value in source.items()}
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"model registry file not found: {path}")
    return load_mapping(path.read_text(encoding="utf-8"))


def _build_entry(key: str, value: Any, *, existing: ModelInfo | None) -> ModelInfo:
    """Validate one registry entry, layering it over an entry with the same key."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"registry entry {key!r} must be a mapping, got {type(value).__name__}")
    fields: dict[str, Any] = dict(existing.model_dump()) if existing else {}
    fields.update({str(k): v for k, v in value.items()})
    verified = fields.get("last_verified")
    if verified is not None and not isinstance(verified, str):
        fields["last_verified"] = str(verified)  # PyYAML parses a bare date into date
    capabilities = fields.get("capabilities")
    if isinstance(capabilities, str):
        fields["capabilities"] = tuple(part.strip() for part in capabilities.split(",") if part)
    elif isinstance(capabilities, list):
        fields["capabilities"] = tuple(str(part) for part in capabilities)
    fields["pattern"] = key
    fields["known"] = True
    return ModelInfo(**fields)


def glob_match(value: str, pattern: str) -> bool:
    """Match a glob with ``*`` only, so a model id's own characters stay literal."""
    if "*" not in pattern:
        return value == pattern
    segments = pattern.split("*")
    if not value.startswith(segments[0]):
        return False
    position = len(segments[0])
    for segment in segments[1:-1]:
        found = value.find(segment, position)
        if found == -1:
            return False
        position = found + len(segment)
    tail = segments[-1]
    return value.endswith(tail) and len(value) - len(tail) >= position


@lru_cache(maxsize=1)
def default_registry() -> Registry:
    """The shipped registry, parsed once per process."""
    return Registry.from_sources(_SHIPPED)


def main(argv: list[str] | None = None) -> int:
    """Print registry entries with their prices and last verified dates.

    Args:
        argv: Command line arguments, or None to read ``sys.argv``.

    Returns:
        0 when every listed entry carries a ``last_verified`` date, 1 otherwise, so
        that a release script can gate on stale pricing.
    """
    parser = argparse.ArgumentParser(
        prog="python -m aaron.registry",
        description="Inspect the shipped model registry so pricing can be reviewed at release.",
    )
    parser.add_argument("--check", action="store_true", help="list every model and exit")
    parser.add_argument("--registry", help="extra registry file to merge over the shipped one")
    parser.add_argument("model", nargs="?", help="show one model instead of the whole table")
    args = parser.parse_args(argv)

    registry = default_registry().merge(args.registry)
    if args.model:
        try:
            entries = [(args.model, registry.require(args.model))]
        except UnknownModel as error:
            print(error.message, file=sys.stderr)
            return 2
    else:
        entries = list(registry)
    header = f"{'model':44} {'in $/Mtok':>10} {'out $/Mtok':>11} {'region':>8}  verified"
    print(header)
    print("-" * len(header))
    stale = 0
    for name, entry in entries:
        verified = entry.last_verified or "never"
        if entry.last_verified is None:
            stale += 1
        price_in, price_out = _money(entry.input_usd_per_mtok), _money(entry.output_usd_per_mtok)
        region = entry.provider_region or "?"
        print(f"{name:44} {price_in:>10} {price_out:>11} {region:>8}  {verified}")
    print(f"\n{len(entries)} entries, {stale} without a last_verified date.")
    print("Prices go stale. Verify them against provider pricing pages before a release.")
    return 1 if stale else 0


def _money(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}".rstrip("0").rstrip(".")


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    sys.exit(main())
