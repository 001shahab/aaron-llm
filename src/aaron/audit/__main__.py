# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""``python -m aaron.audit summarise a.jsonl``.

Prints calls, tokens, cost and error rate grouped by model and by tag. Standard
library only, and it never prints record content even when the file has it, so the
summary of a content recording log is still safe to paste into a ticket.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Bucket:
    """Running totals for one group of records."""

    calls: int = 0
    errors: int = 0
    violations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    estimated: bool = False

    def add(self, record: dict[str, object]) -> None:
        """Fold one record into the totals."""
        self.calls += 1
        outcome = record.get("outcome")
        if outcome == "policy_violation":
            self.violations += 1
        elif outcome != "ok":
            self.errors += 1

        usage = record.get("usage")
        if isinstance(usage, dict):
            self.input_tokens += int(usage.get("input_tokens") or 0)
            self.output_tokens += int(usage.get("output_tokens") or 0)
        cost = record.get("cost")
        if isinstance(cost, dict):
            self.usd += float(cost.get("usd") or 0.0)
            self.estimated = self.estimated or bool(cost.get("estimated"))

    @property
    def error_rate(self) -> float:
        """Failed calls, including policy refusals, as a fraction of all calls."""
        return (self.errors + self.violations) / self.calls if self.calls else 0.0


def read_records(paths: Sequence[Path]) -> Iterator[dict[str, object]]:
    """Yield each record from one or more JSONL files, skipping unreadable lines."""
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    print(f"{path}:{number}: skipping malformed line", file=sys.stderr)
                    continue
                if isinstance(record, dict):
                    yield record


def summarise(paths: Sequence[Path]) -> int:
    """Print the summary tables.

    Args:
        paths: Audit files to read.

    Returns:
        A process exit code: 0 always, since a summary is not a test.
    """
    by_model: dict[str, Bucket] = defaultdict(Bucket)
    by_tag: dict[str, Bucket] = defaultdict(Bucket)
    overall = Bucket()

    for record in read_records(paths):
        model = str(record.get("model_requested") or "unknown")
        by_model[model].add(record)
        overall.add(record)
        tags = record.get("tags")
        if isinstance(tags, dict):
            for key, value in tags.items():
                by_tag[f"{key}={value}"].add(record)

    if not overall.calls:
        print("no records found")
        return 0

    _table("by model", by_model)
    if by_tag:
        _table("by tag", by_tag)

    print(
        f"\ntotal: {overall.calls} calls, "
        f"{overall.input_tokens + overall.output_tokens} tokens, "
        f"${overall.usd:.4f}{' (contains estimates)' if overall.estimated else ''}, "
        f"{overall.error_rate:.1%} error rate"
    )
    return 0


def _table(title: str, buckets: dict[str, Bucket]) -> None:
    print(f"\n{title}")
    header = f"{'key':38} {'calls':>6} {'in tok':>9} {'out tok':>9} {'usd':>9} {'errors':>7}"
    print(header)
    print("-" * len(header))
    for key, bucket in sorted(buckets.items(), key=lambda item: -item[1].calls):
        marker = "~" if bucket.estimated else " "
        print(
            f"{key[:38]:38} {bucket.calls:>6} {bucket.input_tokens:>9} "
            f"{bucket.output_tokens:>9} {marker}{bucket.usd:>8.4f} {bucket.error_rate:>6.1%}"
        )
    print("  ~ means the cost includes at least one estimate")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m aaron.audit``.

    Args:
        argv: Command line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m aaron.audit",
        description="Summarise Aaron audit logs. Never prints recorded content.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("summarise", help="group calls, tokens, cost and errors")
    summary.add_argument("paths", nargs="+", type=Path, help="one or more .jsonl audit files")
    args = parser.parse_args(argv)

    missing = [str(path) for path in args.paths if not path.is_file()]
    if missing:
        print(f"file not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    return summarise(args.paths)


if __name__ == "__main__":
    sys.exit(main())
