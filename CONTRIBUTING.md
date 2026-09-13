# Contributing

Aaron's selling point is that a reviewer can read the whole library in an afternoon.
That constrains what a contribution can look like, so this page is mostly about the
constraints. They are not personal preferences; they are the product.

## Setting up

```sh
git clone https://github.com/001shahab/aaron-llm
cd aaron-llm
python3 -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements-dev.lock
pip install --no-deps -e .
```

Point git at the versioned hooks while you are here. The `commit-msg` hook refuses a
commit with the wrong identity or an attribution trailer, which is the rule this
repository is strictest about:

```sh
git config core.hooksPath .githooks
```

The lock is hash pinned, so this installs exactly what CI installs. Regenerate it after
changing `requirements-dev.in`:

```sh
pip install pip-tools
pip-compile --generate-hashes --strip-extras --output-file=requirements-dev.lock requirements-dev.in
```

## The four gates

Run all of them before opening a pull request. CI runs the same commands, so a green
local run means a green build.

```sh
ruff format . && ruff check .
mypy --strict src/aaron
pytest --cov=aaron --cov-report=term-missing --cov-fail-under=90
python3 scripts/check_dependencies.py
python3 scripts/check_authorship.py
```

The fourth one is the unusual one. It fails the build if a runtime dependency other than
`httpx` and `pydantic` appears, if any vendor SDK or analytics package is imported
anywhere, if a provider file exceeds 400 lines, or if the library exceeds 4000
implementation lines. `docs/decisions.md` explains how that last number is counted. The fifth asserts that
every commit and tag in the history has one author and one committer, both
Prof. Shahab Anbarjafari, and carries no attribution trailer.

## Rules that will not bend

- **Two runtime dependencies**, `httpx` and `pydantic`. An optional extra is allowed if
  it is imported inside a function, never at module import time, and the library works
  without it.
- **No vendor SDK.** Every provider is written against its documented HTTP API. This is
  the whole point of the project.
- **No network in a test, ever.** `respx` mocks HTTP; a real request is a test bug. The
  socket is patched in `conftest.py` so an accidental one fails loudly. Tests that
  genuinely need an endpoint live in `tests/live/` and are skipped unless
  `AARON_LIVE_TESTS=1`.
- **No telemetry, ever**, and no network call at import time or install time.
- **A credential never leaves the process.** Not in a record, an exception, a repr, or a
  log line. If your change touches request building, headers, errors or audit, add to
  `tests/test_security.py` rather than only reading the existing tests.
- **No bare `except`, no `print`.** Logging goes through `logging.getLogger("aaron")`.
- **Full type annotations**, `mypy --strict` clean, no `Any` in a public signature.
- **Docstrings on every public function and class**, with `Args`, `Returns` and
  `Raises` where they apply. Comments explain *why*, not *what*: if a line needs a
  comment to say what it does, rewrite the line.
- **Every behaviour change comes with a test**, and the test name says what the
  behaviour is. `test_a_document_is_refused_before_the_call` is the style; `test_ollama_3`
  is not.

## Adding a provider

Think first about whether it belongs in the library at all. If the endpoint speaks the
OpenAI chat shape, `openai_compat` already covers it with a `base_url` and no new code.
Aaron has five providers deliberately, and breadth is LiteLLM's job.

If it is genuinely a different API:

1. One file in `src/aaron/providers/`, under 400 lines, in the same five part order as
   the others: metadata, `build_request`, `parse_response`, `chunk_events`, `map_error`.
2. Fixtures in `tests/fixtures/` captured from a real response with every identifier and
   credential removed. Never paste a real key into a fixture, even a revoked one.
3. Pass `tests/test_provider_contract.py`, which is the shared contract every provider
   is held to.
4. Registry entries in `src/aaron/registry/models.yaml` with prices, context window,
   capabilities, region and a `last_verified` date.
5. Optionally, live tests in `tests/live/`.

You can also ship a provider as your own package, with no fork and no pull request, by
registering an entry point in the `aaron.providers` group. That is documented in
[docs/providers.md](docs/providers.md), and for a niche endpoint it is the better
option for both of us.

## Changing the model registry

`models.yaml` is data a human maintains. Quote dates, keep entries alphabetical within a
provider, and update `last_verified` when you check a price against the provider's
pricing page. Do not update the date without actually checking.

```sh
python -m aaron.registry --check
```

## Pull requests

Small and single purpose. A pull request that adds a feature and reformats a file is two
pull requests.

Write the commit message for someone reading `git log` in two years: what changed and
why, in prose, wrapped at 72 characters. If the change fixes a bug, say what the bug
did, not just that it is fixed.

By contributing you agree that your contribution is licensed under the Apache License,
Version 2.0, the same as the project.

## Releasing

`.github/workflows/release.yml` publishes to PyPI when a GitHub release is published,
so the notes exist before the artifacts do. It runs every gate first, refuses to build
when the tag does not match `__version__`, refuses to release a registry price with no
`last_verified` date, and attaches build provenance attestations to what it uploads.

Publication uses PyPI trusted publishing, so no API token is stored in this repository
or in GitHub secrets: PyPI verifies the workflow's OIDC identity instead. That needs a
one time publisher entry on PyPI naming the owner, the repository, `release.yml` and the
`pypi` environment. Running the workflow manually builds and gates without publishing,
which is the way to check a release before committing to it.

```sh
# bump src/aaron/_version.py, move the Unreleased section of CHANGELOG.md under the
# new version, then:
git tag -a v0.1.1 -m "aaron-llm 0.1.1"
git push origin main --tags
gh release create v0.1.1 --notes-from-tag   # this is what triggers the publish
```

## Where decisions are recorded

If you find yourself choosing between two reasonable designs, take the one with the
smaller dependency tree and the more readable code, and add an entry to
`docs/decisions.md` stating the ambiguity, the choice and the cost. Sixteen entries are
already there; the file is the reason nobody has to re-litigate them.

---

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
Author and maintainer: Prof. Shahab Anbarjafari, shb@3sholding.com
