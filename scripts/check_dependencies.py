# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Enforce the dependency posture promised in SECURITY.md.

Run as ``python scripts/check_dependencies.py``. Exits non zero with an
explanation when the promise is broken. Uses only the standard library so it can
run before anything is installed.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "aaron"

ALLOWED_RUNTIME = {"httpx", "pydantic"}

# Vendor SDKs that must never be imported anywhere in the package.
BANNED_IMPORTS = {
    "openai",
    "anthropic",
    "google",
    "google_generativeai",
    "google_genai",
    "ollama",
    "cohere",
    "mistralai",
    "litellm",
    "langchain",
    "boto3",
    "requests",
    "aiohttp",
}

# Imported lazily inside functions only, never at module import time.
LAZY_ONLY = {"yaml", "opentelemetry"}

IMPORT_RE = re.compile(r"^(?:from|import)\s+([A-Za-z_][\w.]*)", re.MULTILINE)

_IGNORED_TOKENS = frozenset(
    {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.COMMENT,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
)


def _requirement_name(spec: str) -> str:
    return re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].strip().lower()


def check_runtime_dependencies(errors: list[str]) -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {_requirement_name(d) for d in data["project"]["dependencies"]}
    if declared != ALLOWED_RUNTIME:
        errors.append(
            f"runtime dependencies must be exactly {sorted(ALLOWED_RUNTIME)}, "
            f"found {sorted(declared)}"
        )


def check_no_vendor_sdks(errors: list[str]) -> None:
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        for match in IMPORT_RE.finditer(source):
            root = match.group(1).split(".")[0]
            if root in BANNED_IMPORTS:
                line = source[: match.start()].count("\n") + 1
                errors.append(f"{rel}:{line} imports banned vendor package {root!r}")


def check_optional_imports_are_lazy(errors: list[str]) -> None:
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        for raw in source.splitlines():
            match = IMPORT_RE.match(raw)
            if match is None:
                continue
            root = match.group(1).split(".")[0]
            # A module level import has no leading indentation.
            if root in LAZY_ONLY and not raw.startswith((" ", "\t")):
                errors.append(f"{rel} imports optional dependency {root!r} at module level")


def code_lines(source: str) -> int:
    """Count lines that carry code, ignoring blanks, comments and docstrings.

    The 4000 line budget is about how much implementation a reviewer has to read.
    Docstrings are mandatory on every public function, and they help that reviewer
    rather than burden them, so they are not counted. See docs/decisions.md.
    """
    tree = ast.parse(source)
    docstrings: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
        ):
            docstrings.append((first.lineno, first.end_lineno))

    counted: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in _IGNORED_TOKENS or not token.line.strip():
            continue
        line = token.start[0]
        if any(start <= line <= end for start, end in docstrings):
            continue
        counted.add(line)
    return len(counted)


def check_provider_file_size(errors: list[str]) -> None:
    # base.py is where shared logic is meant to accumulate, so the per provider
    # limit does not apply to it.
    for path in sorted((SRC / "providers").glob("*.py")):
        if path.name in ("base.py", "__init__.py"):
            continue
        lines = len(path.read_text(encoding="utf-8").splitlines())
        if lines > 400:
            errors.append(f"{path.relative_to(ROOT)} is {lines} lines, the limit is 400")


def check_source_size(errors: list[str]) -> None:
    """Report the three line counts, and enforce the budget on implementation lines.

    The 4000 line budget is a limit on how much implementation a reviewer has to
    read. Docstrings are mandatory on every public function and they help that
    reviewer; import blocks and ``__all__`` lists are boilerplate the formatter
    expands one name per line. Neither is implementation, so neither is counted. See
    docs/decisions.md.
    """
    physical = code = logic = 0
    for path in SRC.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        physical += len(source.splitlines())
        code += code_lines(source)
        logic += code_lines(source) - _boilerplate_lines(source)

    print(f"src/aaron: {physical} physical lines, {code} code lines, {logic} implementation lines")
    if logic > 4000:
        errors.append(f"src/aaron is {logic} implementation lines, the limit is 4000")


def _boilerplate_lines(source: str) -> int:
    """Count lines taken by import statements and ``__all__`` declarations."""
    total = 0
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import | ast.ImportFrom) and node.end_lineno:
            total += node.end_lineno - node.lineno + 1
        elif isinstance(node, ast.Assign) and node.end_lineno:
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "__all__" in names:
                total += node.end_lineno - node.lineno + 1
    return total


def main() -> int:
    errors: list[str] = []
    check_runtime_dependencies(errors)
    check_no_vendor_sdks(errors)
    check_optional_imports_are_lazy(errors)
    check_provider_file_size(errors)
    check_source_size(errors)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("dependency policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
