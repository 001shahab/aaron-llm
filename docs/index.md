# Aaron

**One client for every model, with a policy and an audit trail.**

The [README](../README.md) is the sales pitch and the quickstart. These pages are the
reference, written for two readers who are not the same person: the engineer wiring
Aaron into a service, and the compliance reviewer who has to sign off on it.

| Page | For |
| --- | --- |
| [providers.md](providers.md) | Which endpoint each provider calls, what it supports, and how to reach a feature the abstraction does not name |
| [policy.md](policy.md) | The rules, the order they run in, and what a residency claim does and does not prove |
| [audit.md](audit.md) | Every field of the record, what is deliberately absent, and how to keep it lawful |
| [decisions.md](decisions.md) | Design decisions and the reasoning behind them, in the order they were made |

## The shape of the library

Ten public names, and everything else is an implementation detail you are free to
read but should not import:

```python
from aaron import (
    Aaron,          # synchronous client
    AsyncAaron,     # the same surface, awaitable
    Message,        # one turn of a conversation, with optional images and documents
    Policy,         # what a call must satisfy before it may leave the process
    Tool,           # a function the model may ask you to run
    Response,       # what came back: text, tool_calls, usage, cost, audit_id, raw
    JsonlSink,      # append one JSON object per call to a file
    CallbackSink,   # hand each record to your own function
    PolicyViolation,
    AaronError,     # the base of every error Aaron raises
)
```

The five names in the build specification are `Aaron`, `AsyncAaron`, `Message`,
`Policy` and `Tool`. The rest of `aaron.__all__` is there because you cannot use the
five without them: the types that come back, the sinks, and the error classes.

Five methods on the client, plus the bookkeeping:

| Method | Returns | Notes |
| --- | --- | --- |
| `chat(model, messages, ...)` | `Response` | One request, one answer |
| `stream(model, messages, ...)` | iterator of events | Ends with a `DoneEvent` carrying the assembled `Response` |
| `extract(model, messages, schema=Model)` | an instance of `Model` | Validated locally whatever the provider claims |
| `batch(requests, concurrency=8)` | `list[Response \| AaronError]` | Async only, input order, exceptions captured not raised |
| `dry_run(model, messages, ...)` | `PreparedRequest` | Builds the body and sends nothing; auth headers masked |
| `alias(name, target)` | `None` | A short name for a long model string |

The module level functions `aaron.chat`, `aaron.stream` and `aaron.extract` use a
lazily created default client, for a script where a client object is ceremony.

## Model strings

Always `provider/model`. There is no registry of magic short names that changes
meaning between versions, and no automatic quality based routing that picks a model
for you. If you want a short name, make one yourself and it is yours:

```python
client.alias("fast", "ollama/llama3.1:8b")
```

An alias may point at another alias. A cycle is a `ConfigurationError`, not a hang.
An alias may not contain a slash, so it can never shadow a real model string.

## Configuration

Every setting resolves in the same order. The first one that has a value wins:

1. the keyword argument on the call, for example `timeout=5.0`
2. the keyword argument on the client, for example `Aaron(timeout=5.0)`
3. `aaron.toml` in the working directory, or the file named by `config_file=`
4. the environment
5. the built in default

The environment variables:

| Variable | Effect |
| --- | --- |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_COMPAT_API_KEY` | Credentials, read at call time |
| `AARON_DEFAULT_MODEL` | Used when a call omits the model |
| `AARON_TIMEOUT` | Default request timeout in seconds |
| `AARON_POLICY_FILE` | Path to a policy file, overriding `aaron.policy.yaml` |
| `AARON_AUDIT_PATH` | Path to a JSONL audit file, which turns auditing on |
| `AARON_LIVE_TESTS` | Set to `1` only to run the tests that hit real endpoints |

An `aaron.toml`, all keys optional:

```toml
default_model = "ollama/llama3.1"
timeout = 30.0
connect_timeout = 5.0
audit_path = "calls.jsonl"
policy_file = "aaron.policy.yaml"
model_registry = "our-models.yaml"

[aliases]
fast = "ollama/llama3.1:8b"
smart = "anthropic/claude-sonnet-4-5"

[base_urls]
openai_compat = "https://api.groq.com/openai/v1"
```

If `aaron.policy.yaml` exists in the working directory it is loaded with no
configuration at all. That is deliberate: a policy should be hard to forget.

Credentials are the one thing `aaron.toml` will not carry. Pass them as an argument,
or leave them in the environment or your secret manager:

```python
client = Aaron(api_keys={"openai": lambda: vault.read("openai")})
```

A callable is re-read on every call, so a rotated key takes effect immediately and
nothing is cached in a module global.

## Errors

Everything inherits from `AaronError`, so one `except` clause can wrap a call. The
useful branches:

```
AaronError
├── ConfigurationError      the client or call could not work as set up
│   ├── MissingAPIKey
│   ├── UnknownModel
│   └── UnknownProvider
├── PolicyViolation         refused before any network call, carries rule and detail
├── RequestError            we sent something the provider would not accept
│   ├── InvalidRequest
│   ├── ContextLengthExceeded
│   └── ToolArgumentError   the model called a tool with arguments that failed validation
├── ProviderError           the provider itself failed, or refused
│   ├── AuthenticationError
│   ├── PermissionError
│   ├── RateLimitError      carries retry_after when the provider gave one
│   ├── ServiceOverloaded
│   ├── ServerError
│   └── ContentFilterError
└── TransportError          no usable response at all
    ├── TimeoutError
    ├── ConnectionError
    └── LocalProviderUnavailable
```

`TimeoutError`, `ConnectionError` and `PermissionError` reuse familiar names but they
are Aaron's, not the builtins, and they do not inherit from them. For that reason
they are not re-exported from the `aaron` namespace, where they would shadow the
builtins in your code. Catch `TransportError`, `ProviderError` or `AaronError`, or
import the specific class from `aaron.errors`. `except OSError` will not catch them.

Every error carries `provider`, `model`, `status_code` and `request_id` where they
are known, and its `str` and `repr` are safe to log: a credential never appears in
either.

## What Aaron will not do

Named here so you do not go looking. No proxy or gateway process, no dashboard, no
agent framework, no chains, no conversation memory, no RAG, no embeddings, images or
speech, no prompt templates, no automatic quality based model selection, and no
database. It is a client library with a policy and a log.
