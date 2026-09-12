# Policy

This page is written for the person who has to sign off on a deployment, not only for
the person who wrote it. It says what each rule actually checks, what the evidence is
worth, and where the limits are. If you are looking for the short version, the README
has it.

A policy is evaluated **entirely before any network call**. Nothing is sent and then
regretted: a refusal costs one dictionary lookup and a token estimate, and it happens
in your process.

## Declaring one

In code:

```python
from aaron import Aaron, Policy, EmailRedactor, RegexRedactor

policy = Policy(
    allow=["ollama/*", "anthropic/*"],
    deny=["*/gpt-3.5*"],
    residency="eu",
    max_usd_per_call=0.50,
    max_input_tokens=100_000,
    require_capabilities=["tools", "streaming"],
    fallback=["ollama/llama3.1:8b"],
    redactors=[EmailRedactor(), RegexRedactor(r"\b\d{11}\b", "[ID]")],
    on_violation="fallback",
)
client = Aaron(policy=policy)
```

Or in a file, so that it can be reviewed like code, diffed, and pointed at in an
audit. `aaron.policy.yaml` in the working directory is picked up with no
configuration at all, because a policy should be hard to forget:

```yaml
# aaron.policy.yaml
allow:
  - ollama/*
  - anthropic/*
deny:
  - "*/gpt-3.5*"
residency: eu
max_usd_per_call: 0.50
max_input_tokens: 100000
require_capabilities: [tools, streaming]
fallback: [ollama/llama3.1:8b]
redactors:
  - email
  - ip
  - pattern: '\b\d{11}\b'
    replacement: "[ID]"
    name: national-id
on_violation: fallback
```

A key that is not a policy field is a `ConfigurationError` naming the valid fields,
never a silently ignored typo. So is an unknown capability name, an unknown redactor,
an invalid regular expression, and `on_violation: fallback` with an empty fallback
list. All of it fails when the policy loads, not on the first call at three in the
morning.

Loading a file explicitly, or from the environment:

```python
client = Aaron(policy=Policy.from_file("policies/production.yaml"))
```

```sh
export AARON_POLICY_FILE=policies/production.yaml
```

## The rules, in the order they run

The order is fixed and documented because it determines which reason you are told
when a call breaks two rules at once. The first failing rule is the one reported.

| # | Rule | Checks |
| --- | --- | --- |
| 1 | `deny`, then `allow` | Glob patterns against the canonical `provider/model` string. A deny always beats an allow. |
| 2 | `residency` | The model's `provider_region` from the registry against the required jurisdiction. |
| 3 | `require_capabilities` | Capability names from the registry: `tools`, `vision`, `documents`, `json_schema`, `json_mode`, `streaming`, `thinking`. |
| 4 | `max_input_tokens` | An estimate of the input size, before sending. |
| 5 | `max_usd_per_call` | The worst case cost: estimated input tokens plus `max_tokens` of output, at registry prices. |
| 6 | `redactors` | Applied to every outbound text part, in the order given. |

Redaction is last because it is not a gate. The first five can refuse a call; the
sixth changes it.

In globs only `*` is special, and it matches across slashes. `ollama/*` is every
Ollama model, `*/gpt-4*` is every GPT-4 on any provider, and a `?` or a `[` is a
literal character, not a wildcard. That is deliberately narrower than `fnmatch`: a
policy pattern that behaves in a surprising way is a security problem.

## Residency, and what the claim is worth

Read this paragraph before you rely on `residency` in a filing.

`residency` compares the required jurisdiction with the `provider_region` recorded in
the model registry. That field records **the jurisdiction of the provider's default
endpoint**, which is the only thing a client library in your process can honestly
know. It is not a statement about where the provider's subprocessors run, about a
zero data retention agreement you may have signed, or about a regional endpoint you
were granted. It is evidence for a review, not a legal conclusion.

What is accepted:

| `residency` | Accepted regions |
| --- | --- |
| `eu`, `eea` | `eu`, `eea`, `local` |
| `us` | `us`, `local` |
| `uk` | `uk`, `local` |
| `ch` | `ch`, `local` |
| `local` | `local` only |

`local` means the model runs on the caller's own machine, so it satisfies every
jurisdiction: no personal data crosses any border at all. That is why an Ollama model
is the natural fallback for a residency policy.

**An unknown region is rejected, never assumed compliant.** A model with no
`provider_region` in the registry fails a residency rule, and the message says so in
those words. A residency requirement that quietly passed because Aaron had no
metadata would be worse than no requirement at all.

If you have a regional endpoint the shipped registry does not know about, declare it
and Aaron will believe you. That is the honest division of responsibility: you know
your contract, Aaron knows what it shipped with.

```yaml
# our-models.yaml, passed as Aaron(model_registry="our-models.yaml").
# Our deployment is Azure Sweden Central, so pair this with
# base_urls={"openai": "https://our-resource.openai.azure.com/openai/v1"}.
openai/gpt-4o:
  provider_region: eu
  last_verified: "2026-09-12"
```

A field name Aaron does not know is ignored so that an entry written for a newer
version still loads, but it is logged as a warning on the `aaron` logger. Turn logging
on when you write a registry file, or a misspelled `provider_region` will be dropped
and the model will simply fail the residency rule.

A residency value Aaron has never heard of is treated as an exact region match, so a
private jurisdiction code works as long as the same string is in your registry.

## Cost and token ceilings

`max_input_tokens` and `max_usd_per_call` use estimates, and estimates are the point:
a ceiling that could only be checked after the call would not be a ceiling.

The input estimate is a character based approximation, not the provider's tokeniser.
It is deliberately rough and slightly pessimistic. Aaron does not vendor a tokeniser
for each provider, because that would mean a large dependency and a version treadmill
for a number that is only used to refuse obviously oversized calls. If you need exact
accounting, take it from `response.usage`, which is what the provider actually
charged.

The cost ceiling uses the worst case: estimated input tokens plus the full
`max_tokens` you allowed, at the prices in the registry. A call that would only have
produced fifty tokens can still be refused for a generous `max_tokens`. That is the
safe direction to be wrong in, and setting `max_tokens` makes the estimate tighter.

A model with no price in the registry costs `0.0`, so a cost ceiling does not
constrain it. If you care, put a price in your own registry file.

## Redactors

A redactor rewrites outbound text before it reaches a provider. Three ship:

| Name | Replaces |
| --- | --- |
| `email` | Email addresses, with `[EMAIL]` |
| `ip` | IPv4 and common IPv6 literals, with `[IP]`, because an IP address is personal data under the GDPR |
| a mapping with `pattern` | Whatever your expression matches |

Write your own by implementing the `Redactor` protocol: any object with a `name` and
a `redact(text) -> (text, count)` will do.

```python
class NhsNumberRedactor:
    name = "nhs-number"

    def redact(self, text: str) -> tuple[str, int]:
        return re.subn(r"\b\d{3} ?\d{3} ?\d{4}\b", "[NHS]", text)
```

Be clear about what this buys you. A regular expression is a safety net for the case
you did not anticipate, not a control you should rely on. It will miss a phone number
written in words, a name, a date of birth in an unusual format, and anything in an
image or a PDF you attached: **redactors apply to text parts only.** The right control
is not sending personal data in the first place. Aaron gives you the net because nets
catch things, and the audit record counts what it caught so you can see whether it is
firing more than you expected.

The count of replacements goes into the audit record. The patterns never do, because a
pattern can itself describe personal data. The record names the redactor, and the
policy file in version control shows what that name means at that commit.

## A refusal

Structured, not a sentence to parse:

```python
try:
    client.chat("openai/gpt-4o", "Hello")
except PolicyViolation as violation:
    violation.rule    # 'residency'
    violation.model   # 'openai/gpt-4o'
    violation.detail  # a sentence a compliance reader can act on
```

`rule` is one of `deny`, `allow`, `residency`, `capabilities`, `max_input_tokens`,
`max_usd_per_call`. Branch on it, count it, alert on it. A refusal also produces an
audit record with `outcome: "policy_violation"`, so a review can count refusals as
easily as calls, which is the number that shows a policy is actually load bearing.

## Fallback

With `on_violation="fallback"`, a refused call walks the `fallback` list in order.
**Every rule is re-evaluated for each candidate**, so a fallback cannot be a way to
smuggle a call past the policy. If every candidate is refused, the exception is
raised, and the audit record lists every candidate that was tried with the rule that
refused it.

```python
policy = Policy(
    residency="eu",
    fallback=["ollama/llama3.1:8b"],
    on_violation="fallback",
)
reply = client.chat("anthropic/claude-sonnet-4-5", "Contact me at ada@example.com")
# Anthropic's default endpoint is in the US, so this ran on the local model instead.
# reply.model is the model that actually ran, not the one you asked for.
```

Two things to know before you turn it on. The credential for the model that actually
runs is resolved after the policy has settled, so a fallback to a local model works on
a machine with no cloud key at all. And `reply.model` is the model that ran: check it
if the difference matters to your caller.

## What a policy does not do

- It does not inspect the model's output. Redaction is outbound only, and there is no
  content classifier. A provider's own safety refusal arrives as
  `ContentFilterError`.
- It does not enforce a budget across calls. `max_usd_per_call` is per call, by name.
  There is no shared counter, because Aaron has no database and no server, and a
  budget that only counted one process would be misleading.
- It does not make you compliant. It makes what you decided explicit, enforces it in
  the client, and writes down what happened. The decision is still yours.
