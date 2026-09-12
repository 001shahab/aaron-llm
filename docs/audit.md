# Audit

Every call produces **exactly one record**: one when it succeeds, one when policy
refuses it, one when the provider fails, one when a stream is abandoned half way. Not
zero, and not one per retry. If your log has ten thousand lines, ten thousand calls
were attempted.

The exception is a call that could not be assembled at all, such as a missing
credential or a model string that is not `provider/model`. Those raise a
`ConfigurationError` or an `InvalidRequest` from the client before a request exists,
and they are not recorded, because nothing was attempted and there is nothing to say
about a provider that was never chosen.

The record is designed to be evidence. It says what was asked for, what actually ran,
where it was processed, what it cost, and what was stripped before it left, **without
containing the prompt** unless you explicitly turned that on.

## Turning it on

```python
from aaron import Aaron, JsonlSink

client = Aaron(audit=JsonlSink("calls.jsonl"))
```

Or `AARON_AUDIT_PATH=calls.jsonl`, or `audit_path` in `aaron.toml`. A path is enough:
`Aaron(audit="calls.jsonl")` builds the sink for you. With no audit configured,
records are built and dropped, which costs a few microseconds and keeps one code path
instead of two.

Tags are how a record becomes useful six months later. Set the ones that are true for
the whole client, and add per call detail as you go:

```python
client.audit.tag(tenant="acme", purpose="support-triage", controller="acme-gmbh")
reply = client.chat("ollama/llama3.1", "Hello", tags={"ticket": "4417"})
print(reply.audit_id)  # ties this response back to its record
```

Per call tags win on a collision. Values are coerced to strings, so a record's `tags`
is always `dict[str, str]` and a reader never has to guess at a type.

## Every field

| Field | Meaning |
| --- | --- |
| `id` | UUID4. Also on the response as `audit_id`. |
| `timestamp` | UTC, when the call started. |
| `model_requested` | The canonical `provider/model` string you asked for, after aliases. |
| `model_resolved` | What the provider says it used, for example `gpt-4o-2024-11-20`. Null when the call never reached one. |
| `provider` | Provider name. |
| `provider_region` | The registry's region for the model, or null when it has none. |
| `base_url` | The endpoint that was called. |
| `outcome` | `ok`, `policy_violation`, `provider_error`, `timeout`, `cancelled`. |
| `error_type` | The exception class name, or null. Never a message, which could quote the prompt. |
| `usage` | `input_tokens`, `output_tokens`, `cached_input_tokens`, `reasoning_tokens`. Null when nothing came back. |
| `cost` | `usd`, and `estimated` which is true when the figure came from an estimate rather than reported usage at a known price. |
| `latency_ms` | Wall clock for the whole call, retries included. |
| `attempts` | How many HTTP requests were made. Above 1 means a retry happened. |
| `policy_snapshot` | The rules in force, as plain data. Redactors by **name only**. |
| `policy_detail` | For a refused call, the rule that refused it and every candidate tried. Metadata, present whatever `record_content` is. |
| `redactions` | How many replacements the redactors made. |
| `message_count` | Number of messages sent. |
| `input_chars` | Characters of text sent, attachments excluded. |
| `output_chars` | Characters of text received. |
| `prompt_sha256` | Hash of the exact normalised body that was sent. |
| `response_sha256` | Hash of the provider's response body, or null. |
| `tags` | Your tags, client level merged with per call. |
| `content` | Null unless `record_content=True`. |

The two hashes are the part worth understanding. They let you prove **which prompt
produced which answer** without storing either. Keep the prompt in your own system,
where you already have retention and access controls, hash it the same way, and match.
An auditor gets a verifiable chain; the log stays free of personal data.

## What is deliberately absent

Because someone will ask why:

- **The prompt and the completion**, unless you opted in.
- **Credentials.** No API key, no `Authorization`, no `x-api-key`, in any field, in
  any error type, in any repr. `tests/test_security.py` asserts this by serialising
  records from calls made with a known fixture key and scanning the JSON for it.
- **Attachment bytes.** An image or PDF becomes a digest and a byte count, never
  base64 in a log line. It also never inflates `input_chars`.
- **Provider error messages.** Only `error_type`, because a provider's message often
  quotes the input back at you.
- **Redactor patterns.** A pattern can itself describe personal data, so the snapshot
  names the redactor and your policy file in version control says what that name
  meant at that commit.
- **User identity.** Aaron does not invent a user id, a session, or an IP field. If
  you want one, it is your tag and your lawful basis.

## Recording content

```python
client = Aaron(audit=JsonlSink("calls.jsonl", record_content=True))
```

Read this before you do it in production. **Enabling `record_content` may place
personal data in the log.** The file becomes a processing activity of its own: it
needs a lawful basis, a retention period, access control, encryption at rest, and a
line in your records of processing. Attachment bytes are still replaced by digests,
and credentials still never appear, but everything the user typed is now on disk.

It is genuinely useful for a bug you cannot reproduce, for a period of evaluation
before launch, or for a workload with no personal data in it at all. Turn it on for a
window with a reason, and turn it off again.

## Sinks

| Sink | Behaviour |
| --- | --- |
| `JsonlSink(path, record_content=False)` | One JSON object per line, appended. Opens the file under a lock per write, so a rotated or deleted file does not lose records and multiple threads cannot interleave. |
| `CallbackSink(fn, record_content=False)` | Hands each record to your function. The bridge to whatever you already run. |
| `OtelSink(tracer=None, record_content=False)` | One OpenTelemetry span per call, with GenAI convention attribute names. Needs `pip install "aaron-llm[otel]"`, imported inside the sink so the base install stays at two dependencies. |
| `NullSink()` | Discards. The default. |

`Sink` is a `Protocol`: a `write(record)` and a `close()` are the whole contract.

```python
from aaron import Aaron, CallbackSink

def to_our_pipeline(record):
    logger.info("llm_call", extra=record.model_dump(mode="json"))

client = Aaron(audit=CallbackSink(to_our_pipeline))
```

**A sink failure never breaks a call.** An exception from `write` is caught, logged
once on the `aaron` logger, and the call continues. A broken log is an operational
problem; a broken customer request is worse. If you need the opposite guarantee, that
no call proceeds unless it was recorded, do the write in a callback sink that raises
into your own transaction and be aware that you are choosing it deliberately.

If you already run OpenTelemetry, `OtelSink` emits one span per call named
`chat {model}`, covering the real duration, with `gen_ai.*` attributes an existing
dashboard recognises and `aaron.*` attributes for cost, policy rule, redaction count
and tags. Content stays off the span unless you ask for it, which matters more here
than in a file: spans usually leave your infrastructure.

```python
from aaron import Aaron, OtelSink

client = Aaron(audit=OtelSink())  # uses the global tracer provider
```

Without the extra installed, constructing it is a `ConfigurationError` that names the
install command, rather than an `ImportError` from somewhere in the middle of a call.

## Reading the log

```sh
python -m aaron.audit summarise calls.jsonl
```

Calls, tokens, cost and error rate grouped by model and by tag. Standard library only,
and it **never prints record content** even when the file has it, so the summary of a
content recording log is still safe to paste into a ticket. A malformed line is
counted and reported rather than crashing the run, because a truncated last line is
what a real log looks like after a hard kill.

Everything else is `jq`, because the format is one JSON object per line and that is
the point:

```sh
# Spend by tenant this month
jq -r 'select(.timestamp >= "2026-09") | "\(.tags.tenant) \(.cost.usd)"' calls.jsonl |
  awk '{a[$1]+=$2} END {for (t in a) printf "%s %.2f\n", t, a[t]}'

# Every call that policy refused, and why
jq -r 'select(.outcome=="policy_violation") | "\(.model_requested) \(.policy_detail.rule)"' calls.jsonl

# Calls that left the EU, if any
jq 'select(.provider_region != "eu" and .provider_region != "local")' calls.jsonl

# Where redaction is actually firing
jq -r 'select(.redactions > 0) | .tags.purpose' calls.jsonl | sort | uniq -c
```

## Using it as evidence

What the record supports, stated plainly so you do not over-claim in a filing.

**GDPR.** Article 30 records of processing: the log shows which processor received
which category of request, from which tenant, for which stated purpose, and in which
region. Article 32: content is absent by default and attachments are hashed, which is
data minimisation you can demonstrate rather than assert. Article 15 and 17 requests:
the hashes let you locate the calls associated with a subject whose prompts you hold
in your own system, without the log itself becoming another copy of their data.

**EU AI Act.** For a deployer of a general purpose model, the record covers the
logging and traceability expectations: which model version ran, when, at what cost,
with what result, under which policy, with refusals counted alongside successes. The
`policy_snapshot` on every line is the part reviewers tend to want, because it shows
the control was in force at the time of the call rather than at the time of the
review.

**What it does not do.** It is not a conformity assessment, a DPIA, a risk
classification, or a substitute for your provider's data processing agreement. It does
not prove where a subprocessor ran, only which endpoint was called. And a log nobody
reads is not a control: the reason `summarise` exists is that a number someone looks
at each month is worth more than a perfect record nobody opens.
