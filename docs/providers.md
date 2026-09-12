# Providers

One file per provider, each written against the vendor's documented HTTP API. No
vendor SDK is installed, so nothing here can pull a dependency tree in behind your
back, and `src/aaron/providers/openai.py` is a readable account of exactly what Aaron
sends to OpenAI.

Every file has the same five parts, in the same order, so once you have read one you
can review the others quickly: name and metadata, `build_request`, `parse_response`,
`chunk_events` for streaming, and `map_error`.

## The table

| Provider | Endpoint | Auth header | Key variable |
| --- | --- | --- | --- |
| `openai` | `POST {base_url}/chat/completions` | `Authorization: Bearer …` | `OPENAI_API_KEY` |
| `anthropic` | `POST {base_url}/messages` | `x-api-key` | `ANTHROPIC_API_KEY` |
| `google` | `POST {base_url}/models/{model}:generateContent` | `x-goog-api-key` | `GOOGLE_API_KEY` |
| `ollama` | `POST {base_url}/api/chat` | none | none |
| `openai_compat` | `POST {base_url}/chat/completions` | `Authorization: Bearer …` if a key is set | `OPENAI_COMPAT_API_KEY` |

| Provider | Tools | Streaming | Vision | PDF | Region |
| --- | :-: | :-: | :-: | :-: | --- |
| `openai` | yes | SSE | yes | yes, as a base64 file part | `us` |
| `anthropic` | yes | SSE, typed events | yes | yes, with a beta header added only when a document is present | `us` |
| `google` | yes | SSE | yes | yes, as `inline_data` | `us` |
| `ollama` | yes | NDJSON | model dependent | no | `local` |
| `openai_compat` | endpoint dependent | SSE | endpoint dependent | endpoint dependent | unknown, so a residency rule rejects it |

`google` streams with `:streamGenerateContent?alt=sse`. `ollama` is the only provider
that streams newline delimited JSON rather than server sent events, which is why the
decoder is chosen per provider rather than assumed.

## openai_compat, and the five provider files that do not exist

Groq, Mistral, Together, DeepSeek, Fireworks, OpenRouter, vLLM, LM Studio and
llama.cpp's server all speak the OpenAI chat completions shape. One provider with an
explicit `base_url` covers them:

```python
client = Aaron(
    base_urls={"openai_compat": "https://api.groq.com/openai/v1"},
    api_keys={"openai_compat": os.environ["GROQ_API_KEY"]},
)
reply = client.chat("openai_compat/llama-3.3-70b-versatile", "Hello")
```

There is no default `base_url`: a missing one is a `ConfigurationError` rather than a
silent call to OpenAI with someone else's key.

Two consequences worth knowing before you rely on it. The model is not in the
registry, so `cost.usd` is `0.0` with `estimated=True` unless you supply your own
registry entry. And its region is unknown, so a policy with `residency` set will
reject it until you declare one:

```yaml
# our-models.yaml, passed as model_registry=
openai_compat/llama-3.3-70b-versatile:
  provider_region: eu
  input_usd_per_mtok: 0.59
  output_usd_per_mtok: 0.79
  context_window: 131072
```

## Reaching a provider feature Aaron does not name

`provider_options` is merged into the outgoing JSON body verbatim, after Aaron has
built it. Anything the vendor documents is reachable, the same day it ships:

```python
# OpenAI reasoning effort
client.chat("openai/o3", prompt, provider_options={"reasoning_effort": "high"})

# Anthropic extended thinking
client.chat(
    "anthropic/claude-sonnet-4-5",
    prompt,
    provider_options={"thinking": {"type": "enabled", "budget_tokens": 4096}},
)

# Google safety settings
client.chat(
    "google/gemini-2.5-pro",
    prompt,
    provider_options={
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}
        ]
    },
)

# Ollama sampling knobs, which live under "options" in its API
client.chat("ollama/llama3.1", prompt, provider_options={"options": {"num_ctx": 8192}})
```

The merge is shallow and your keys win, so `provider_options` can also override
something Aaron set. That is intentional. It is your request.

Whatever the provider returned that Aaron did not model is on `response.raw`, the
parsed JSON body, so you never have to fork the library to read a new field.

If you are unsure what a combination produces, ask without sending anything:

```python
prepared = client.dry_run("openai/o3", prompt, provider_options={"reasoning_effort": "high"})
print(prepared.method, prepared.url)
print(json.dumps(prepared.body, indent=2))
print(prepared.headers)  # {'authorization': '****', ...}, safe to paste into a review
```

## Normalisation, and where it stops

Aaron makes four things the same across providers, because they are the things that
break code when they differ:

- **Tool calls.** `response.tool_calls` is a list of `ToolCall` with `id`, `name` and
  `arguments` as a parsed `dict`, never a JSON string you have to remember to parse.
  Anthropic's `tool_use` block, OpenAI's `function.arguments` string and Gemini's
  `functionCall.args` object all arrive in that one shape.
- **Stop reasons.** `stop`, `length`, `tool_use`, `content_filter`, or the provider's
  own string when it is something else.
- **Usage.** `input_tokens`, `output_tokens`, `cached_input_tokens` and
  `reasoning_tokens`, zero when the provider does not report them. Ollama's
  `prompt_eval_count` and Anthropic's `cache_read_input_tokens` are mapped in.
- **Errors.** The table in [index.md](index.md#errors), from each provider's own
  status codes and error bodies.

Everything else stays the provider's. Aaron does not invent a common `temperature`
scale, does not emulate tools for a model that lacks them, and does not silently
retry a refusal as a different request.

## Documents and images

```python
Message.user("What is in this chart?", images=["./chart.png"])
Message.user("Summarise this", documents=["./contract.pdf"])
```

Each accepts a path, raw `bytes`, or a `data:` URL. Images may also be an `https`
URL, which is passed through to the provider rather than downloaded, so nothing is
fetched on your behalf.

A `DocumentPart` carries an optional `name`, which becomes the filename OpenAI needs
and is ignored elsewhere. Sending a document to Ollama is an `InvalidRequest` raised
while the request is being built, because the chat API has nowhere to put a file and
dropping it quietly would answer a question about a document the model never saw.

## Adding a provider without forking

`Provider` is a `Protocol`, so a class with the right methods is enough. Register it
on a client, or ship it as an entry point in the `aaron.providers` group and it is
found lazily by name:

```toml
[project.entry-points."aaron.providers"]
my_gateway = "my_package.provider:MyGateway"
```

Then `client.chat("my_gateway/some-model", "Hello")` works, and its calls are subject
to the same policy and produce the same audit records. The contract is the six
members in `src/aaron/providers/base.py`, and `tests/test_provider_contract.py` is
the test suite your provider should also pass.

## Keeping the registry honest

`models.yaml` carries prices, context windows, capabilities and regions. Prices go
stale, and a stale price silently produces a wrong cost figure in an audit record:

```sh
python -m aaron.registry --check          # what is in the registry, and how old it is
python -m aaron.registry openai/gpt-4o    # one model
```

Each entry records the date it was last verified. Check before you quote a number to
anyone, and override anything you do not trust with your own registry file rather
than waiting for a release.
