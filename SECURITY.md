# Security Policy

## Reporting a vulnerability

Email **shb@3sholding.com** with the details. Please do not open a public issue for a
vulnerability.

Include what you need to make the problem reproducible: the version of `aaron-llm`,
what you did, what happened, and what you expected. A minimal script is worth more
than a description. If you have a suggested fix, say so.

What to expect:

- An acknowledgement within three working days.
- An assessment, and if it is a real issue, a fix or a documented mitigation.
- Credit in the changelog, if you would like it, and no credit if you would not.

Please do not include a real API key in your report. If you believe a key of yours was
exposed by Aaron, rotate it first, then tell us.

## Supported versions

Aaron is at 0.1.x. Fixes go into the latest minor version. There is no long term
support branch to backport to yet, and this section will say so honestly when there is.

## Scope

In scope, and treated as a vulnerability:

- **A credential leaving the process anywhere it should not.** An API key, an
  `Authorization` header or an `x-api-key` value appearing in an audit record, an
  exception message, a `repr`, a log line, or a `dry_run` output.
- **Prompt or completion content in an audit record when `record_content` is off.**
  Including indirectly, for example through an error message that quotes the input.
- **A policy rule that can be bypassed.** A model that a `deny` pattern should have
  matched, a residency requirement satisfied by a model with an unknown region, a
  fallback that skips re-evaluation, or a redactor that is silently not applied.
- **A network call Aaron makes that the caller did not ask for.** Aaron has no
  telemetry and fetches nothing at import time or install time. Any traffic other than
  the provider call you requested, or an image URL you explicitly passed, is a bug of
  this kind.
- **Anything that runs code from data**, for example a registry file, a policy file or
  a provider response leading to execution.

Out of scope, though still worth reporting as a normal issue:

- A vulnerability in `httpx`, `pydantic`, or a provider's own API. Report those
  upstream; we will bump a pinned version.
- A stale price or context window in `models.yaml`. It is data, it goes stale, and
  `python -m aaron.registry --check` exists to catch it.
- A model producing bad, biased or unsafe output. Aaron transports your request; it
  does not evaluate the answer.
- The consequences of setting `record_content=True`. That is a documented opt in with a
  warning attached, not a defect.

## What the codebase does about this

So you can check the claims rather than trust them:

- API keys are read at call time, never cached in a module global, and stored wrapped
  in a type whose `repr` is masked. Passing a callable re-reads it on every call.
- `dry_run()` masks auth headers, so its output can be pasted into a review.
- `tests/test_security.py` makes calls with a known fixture key and scans every
  serialised audit record, exception string and repr for it. A leak fails the build.
- Audit records carry `error_type`, not provider error messages, because a provider
  message often quotes the input back.
- Attachment bytes become digests in a record, never base64 in a log line.
- No `__init__.py` side effects beyond imports. No telemetry, and nothing to opt out
  of.
- CI runs `pip-audit` against a hash pinned lock and fails on a known vulnerability,
  plus `mypy --strict`, `ruff`, a dependency policy check that fails if a vendor SDK or
  analytics package appears, and a 90 percent coverage floor.
- No test opens a socket. `tests/conftest.py` patches `socket.connect` to raise, so a
  test that tries to reach the network fails rather than quietly succeeding.

## A note on the audit log

If you enable `record_content=True`, the log file becomes a processing activity of its
own: it may contain personal data, and it needs a lawful basis, a retention period,
access control and encryption at rest. Aaron will not make that decision for you, and
[docs/audit.md](docs/audit.md) says so at more length.

---

Copyright (c) 2026 3S Holding OU. Licensed under the Apache License, Version 2.0.
Maintainer: Prof. Shahab Anbarjafari, shb@3sholding.com
