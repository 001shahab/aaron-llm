# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""``python -m aaron.registry --check``.

Prints every registry entry with its prices, region and the date the price was last
verified, and exits non zero when any entry has never been verified, so a release
script can refuse to ship a stale price. Standard library only.
"""

from __future__ import annotations

import argparse
import sys

from ..errors import UnknownModel
from . import default_registry


def main(argv: list[str] | None = None) -> int:
    """Print registry entries with their prices and last verified dates.

    Args:
        argv: Command line arguments, or None to read ``sys.argv``.

    Returns:
        0 when every listed entry carries a ``last_verified`` date, 1 when any is
        missing one, and 2 when a named model is not in the registry at all.
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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
