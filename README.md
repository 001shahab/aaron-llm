# Aaron

**One client for every model, with a policy and an audit trail.**

Aaron is a small, auditable Python client that talks to OpenAI, Anthropic, Google
and Ollama through one interface, and records what happened on every call.

```sh
pip install aaron-llm
```

Two runtime dependencies, `httpx` and `pydantic`. No vendor SDKs. One readable file
per provider. The whole library is small enough that a security reviewer can read it
in an afternoon, which is the point.

## Thirty second quickstart

No API key needed. [Install Ollama](https://ollama.com/download), then:

```sh
ollama pull llama3.1
```

```python
from aaron import Aaron

client = Aaron()
reply = client.chat("ollama/llama3.1", "Summarise the EU AI Act in one sentence.")
print(reply.text)
print(reply.usage.total_tokens, "tokens,", f"${reply.cost.usd:.4f}")

for event in client.stream("ollama/llama3.1", "Write a haiku about Tartu."):
    if event.type == "text":
        print(event.text, end="", flush=True)
```

Swap the model string for `openai/gpt-4o`, `anthropic/claude-sonnet-4-5` or
`google/gemini-2.5-pro` and set the matching environment variable. Nothing else in
your code changes.

<details>
<summary>The rest of the surface, in one block</summary>

```python
from pydantic import BaseModel
from aaron import Aaron, AsyncAaron, Message, Tool

client = Aaron()

# Multi part messages, tools, and a provider specific escape hatch
reply = client.chat(
    model="anthropic/claude-sonnet-4-5",
    messages=[
        Message.system("You are a careful assistant."),
        Message.user("What is in this chart?", images=["./chart.png"]),
    ],
    max_tokens=1024,
    temperature=0.2,
    tools=[Tool.from_function(get_weather)],
    provider_options={"cache_control": {"type": "ephemeral"}},
)
for call in reply.tool_calls:
    print(call.name, call.arguments)  # arguments are a parsed dict, never a string


# Structured output, validated locally whatever the provider claims to support
class Invoice(BaseModel):
    number: str
    total: float


invoice = client.extract("google/gemini-2.5-pro", "Parse this: ...", schema=Invoice)

# Short names for long model strings
client.alias("fast", "ollama/llama3.1:8b")
client.alias("smart", "anthropic/claude-sonnet-4-5")
reply = client.chat("fast", "Hello")

# Async, including a concurrency limited batch
aclient = AsyncAaron()
reply = await aclient.chat("openai/gpt-4o", "Hello")
results = await aclient.batch(
    [{"model": "openai/gpt-4o", "messages": q} for q in questions],
    concurrency=8,
)  # list[Response | AaronError], in input order, exceptions captured not raised

# See exactly what would go over the wire, and send nothing
print(client.dry_run("openai/gpt-4o", "Hello").body)
```

</details>

## Supported providers

| Model string | Endpoint | Tools | Streaming | Vision | Documents | Key |
| --- | --- | :-: | :-: | :-: | :-: | --- |
| `openai/gpt-4o` | `POST /v1/chat/completions` | yes | yes | yes | PDF | `OPENAI_API_KEY` |
| `anthropic/claude-sonnet-4-5` | `POST /v1/messages` | yes | yes | yes | PDF | `ANTHROPIC_API_KEY` |
| `google/gemini-2.5-pro` | `generateContent` | yes | yes | yes | PDF | `GOOGLE_API_KEY` |
| `ollama/llama3.1:8b` | `POST /api/chat` | yes | yes | model dependent | no | none |
| `openai_compat/<model>` | your `base_url` | yes | yes | endpoint dependent | endpoint dependent | optional |

`openai_compat` covers Groq, Mistral, Together, DeepSeek, Fireworks, OpenRouter,
vLLM, LM Studio and llama.cpp's server without five more provider files. Give it an
explicit `base_url`:

```python
client = Aaron(
    base_urls={"openai_compat": "https://api.groq.com/openai/v1"},
    api_keys={"openai_compat": os.environ["GROQ_API_KEY"]},
)
reply = client.chat("openai_compat/llama-3.3-70b-versatile", "Hello")
```

Reasoning models, prompt caching, safety settings and everything else provider
specific is reachable through `provider_options`, which is merged into the outgoing
JSON body verbatim. The abstraction never blocks a provider feature. See
[docs/providers.md](docs/providers.md).

## Policy

This is the reason to choose Aaron over a general purpose wrapper. Declare what is
allowed, and it is enforced before any network call happens.

```python
from aaron import Aaron, Policy, EmailRedactor, RegexRedactor

policy = Policy(
    allow=["ollama/*", "anthropic/*"],
    deny=["*/gpt-3.5*"],
    residency="eu",  # rejects a model processed outside the EU
    max_usd_per_call=0.50,
    max_input_tokens=100_000,
    require_capabilities=["streaming"],
    fallback=["ollama/llama3.1:8b"],  # tried in order, re evaluated against every rule
    redactors=[EmailRedactor(), RegexRedactor(r"\b\d{11}\b", "[ID]")],
    on_violation="fallback",
)

client = Aaron(policy=policy)
reply = client.chat("anthropic/claude-sonnet-4-5", "Contact me at ada@example.com")
# anthropic is processed in the US, so this runs on the local llama3.1 instead,
# and the email address never leaves the process.
```

A refusal is a typed, structured exception, not a string to parse:

```python
try:
    client.chat("openai/gpt-4o", "Hello")
except PolicyViolation as violation:
    print(violation.rule)  # 'residency'
    print(violation.detail)  # 'openai/gpt-4o is processed in region 'us', which does not ...'
```

An unknown processing region is **rejected**, never assumed compliant. Policy also
loads from a file so it can live in version control and be reviewed like code:

```python
client = Aaron(policy=Policy.from_file("aaron.policy.yaml"))
```

Written for a compliance reader: [docs/policy.md](docs/policy.md).

## Audit

Every call produces exactly one structured record, whether it succeeded, was refused
by policy, or failed at the provider.

```python
from aaron import Aaron, JsonlSink

client = Aaron(audit=JsonlSink("calls.jsonl"))  # record_content=False by default
client.audit.tag(tenant="acme", purpose="support")
reply = client.chat("ollama/llama3.1", "Hello", tags={"ticket": "4417"})
print(reply.audit_id)  # ties the response back to its record
```

One line of `calls.jsonl`, reformatted:

```json
{
  "id": "6f1c0e2a-...", "timestamp": "2026-09-12T21:04:11.882Z",
  "model_requested": "ollama/llama3.1", "model_resolved": "llama3.1",
  "provider": "ollama", "provider_region": "local", "base_url": "http://localhost:11434",
  "outcome": "ok", "error_type": null,
  "usage": {"input_tokens": 26, "output_tokens": 298, "cached_input_tokens": 0, "reasoning_tokens": 0},
  "cost": {"usd": 0.0, "estimated": false},
  "latency_ms": 1204, "attempts": 1,
  "policy_snapshot": {"residency": "eu", "redactors": ["email"], "...": "..."},
  "redactions": 1, "message_count": 2, "input_chars": 84, "output_chars": 1190,
  "prompt_sha256": "9f2b...", "response_sha256": "1ac4...",
  "tags": {"tenant": "acme", "purpose": "support", "ticket": "4417"},
  "content": null
}
```

Prompts and completions are **not** recorded by default. Enabling
`record_content=True` may place personal data in the log; if you do it, treat the
file as a processing record and set a retention period. Credentials never appear in
a record, an exception, a `__repr__` or a log line, and there is a test that asserts
it by scanning serialised records for the fixture key.

```sh
python -m aaron.audit summarise calls.jsonl   # calls, tokens, cost, errors by model and tag
```

`CallbackSink` hands each record to your own function, and `OtelSink` emits one
OpenTelemetry span per call if you already run tracing.

Written for a compliance reader: [docs/audit.md](docs/audit.md).

## How this compares

Be clear about what you need before choosing.

**[LiteLLM](https://github.com/BerriAI/litellm) does more.** A hundred providers,
embeddings, images, audio, rerank, a proxy server with keys and budgets and a UI, and
a large community. If you want breadth, or a gateway your whole company routes
through, use LiteLLM. Aaron has five providers, chat only, and no server.

**[Bifrost](https://github.com/maximhq/bifrost) is faster.** It is a Go gateway built
for throughput, with clustering and very low overhead per request. If your bottleneck
is proxy latency at high volume, that is the right shape of tool. Aaron is a library
in your process, not a hop in your network.

**The vendor SDKs are more complete.** They ship the same day a feature does. Aaron
gives you `provider_options` and `raw` so you are never blocked, but a brand new
endpoint may need an extra line.

**Aaron is for the case where you have to explain yourself.** A regulated team that
needs one client, a policy in version control, an audit trail that satisfies a GDPR
or EU AI Act reviewer, and a dependency footprint small enough to actually read. If
you do not need those, one of the tools above is probably a better fit, and this
README would rather say so than sell you something.

## Dependencies and security posture

- Two runtime dependencies: `httpx` and `pydantic`. That is the whole tree.
- No vendor SDKs. Every provider is written against its documented HTTP API.
- Optional extras (`aaron-llm[yaml]`, `aaron-llm[otel]`) are imported lazily inside
  functions, never at import time.
- No network access at install time or import time.
- **No telemetry of any kind.** Aaron never phones home. There is nothing to opt out
  of, and CI checks that no vendor package or analytics dependency creeps in.
- Keys are read at call time, never cached in a module global, and wrapped in a type
  whose `repr` is masked. `dry_run()` masks the auth header so its output is safe to
  paste into a review.
- CI runs `mypy --strict`, `ruff`, `pip-audit`, and a dependency policy check, and
  enforces 90 percent coverage.

Report a vulnerability privately: see [SECURITY.md](SECURITY.md). To work on Aaron,
see [CONTRIBUTING.md](CONTRIBUTING.md); the release history is in
[CHANGELOG.md](CHANGELOG.md).

Model prices in `models.yaml` go stale. Verify them before you rely on a cost figure:

```sh
python -m aaron.registry --check
```

## Licence and attribution

Copyright (c) 2026 3S Holding OU. All rights reserved.
Licensed under the [Apache License, Version 2.0](LICENSE).

Author and developer: **Prof. Shahab Anbarjafari** — shb@3sholding.com
[3S Holding OU](https://3sholding.com)
