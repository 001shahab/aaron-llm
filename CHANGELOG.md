# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-09-12

First release.

### Added

- **One client for five providers.** `Aaron` and `AsyncAaron`, with `chat`, `stream`,
  `extract`, `batch`, `dry_run` and `alias`. Model strings are always
  `provider/model`. `openai`, `anthropic`, `google` and `ollama` each have one readable
  file written against the provider's documented HTTP API; `openai_compat` covers Groq,
  Mistral, Together, DeepSeek, Fireworks, OpenRouter, vLLM, LM Studio and llama.cpp's
  server with an explicit `base_url`.
- **Two runtime dependencies**, `httpx` and `pydantic`. No vendor SDKs. Optional extras
  `aaron-llm[otel]` and `aaron-llm[yaml]` are imported inside functions, never at module
  import time.
- **Streaming** as a typed event union, over SSE for four providers and NDJSON for
  Ollama, with `StreamAssembler` building the same `Response` a non-streaming call would
  have returned. Sync and async share one parser per provider.
- **Tools.** `Tool.from_function` derives a JSON Schema from a signature and a docstring;
  `Tool.from_model` derives one from a pydantic model. `response.tool_calls` has parsed
  `dict` arguments on every provider, never a JSON string.
- **Structured output.** `extract(..., schema=Model)` returns a validated instance,
  using each provider's native mechanism, including a forced tool call on Anthropic, and
  validating locally whatever the provider claims to support.
- **Policy**, evaluated entirely before any network call, in a fixed order: deny, allow,
  residency, capabilities, input tokens, cost, redaction. An unknown processing region
  is rejected rather than assumed compliant. Refusals are structured
  `PolicyViolation`s carrying `rule`, `model` and `detail`. `Policy.from_file` loads
  YAML, and `aaron.policy.yaml` is picked up with no configuration at all.
- **Audit.** Exactly one `AuditRecord` per call, whether it succeeded, was refused or
  failed. `JsonlSink`, `CallbackSink`, `OtelSink` and `NullSink`, with content recording
  off by default and a sink failure that can never break a call. Prompt and response
  hashes let you prove which prompt produced which answer without storing either.
  `python -m aaron.audit summarise` reports calls, tokens, cost and errors by model and
  tag.
- **Model registry** in `models.yaml` with prices, context windows, capabilities and
  processing regions, mergeable with your own file. `python -m aaron.registry --check`
  lists every entry with the date its price was last verified.
- **Retries** with exponential backoff and full jitter, honouring `Retry-After`, only on
  errors that are actually retryable, with an idempotency key on every request.
- **A typed error hierarchy** under `AaronError`, mapped from each provider's own status
  codes and error bodies.
- **Documentation**: a README that says plainly when LiteLLM or Bifrost is the better
  choice, plus reference pages for providers, policy and audit written for a compliance
  reader, and `docs/decisions.md` recording every design decision with its cost.

### Security

- API keys are read at call time, never cached in a module global, and stored wrapped in
  a type whose `repr` is masked. A callable key source is re-read on every call, so
  rotation takes effect immediately.
- `dry_run()` masks auth headers so its output is safe to paste into a review.
- Audit records carry `error_type` rather than provider error messages, attachment bytes
  become digests, and a test scans serialised records, exceptions and reprs for a known
  fixture key to prove no credential escapes.
- No telemetry of any kind, and no network access at import time or install time.
- CI enforces `mypy --strict`, `ruff`, `pip-audit` against a hash pinned lock, a
  dependency policy check that fails if a vendor SDK or analytics package appears, and
  a 90 percent coverage floor. No test opens a socket.

[Unreleased]: https://github.com/001shahab/aaron-llm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/001shahab/aaron-llm/releases/tag/v0.1.0
