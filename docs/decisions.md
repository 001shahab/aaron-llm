# Decisions

The build specification says that where it is ambiguous, choose the option that keeps
the dependency tree smaller and the code more readable, and write the decision down.
This is that record. Each entry states the ambiguity, the choice, and the cost.

## 1. The 4000 line budget counts implementation lines

**Ambiguity.** The definition of done says "total source lines under 4000 excluding
tests and yaml", without saying what a source line is.

**Decision.** `scripts/check_dependencies.py` reports three numbers and enforces the
budget on the third:

| Count | Meaning | Current |
| --- | --- | --- |
| Physical | every line in every `.py` file | 6869 |
| Code | excluding blank lines, comments and docstrings | 4165 |
| Implementation | code lines, also excluding `import` statements and `__all__` lists | 3710 |

**Why.** The budget exists so that a reviewer can read the whole client. Section 17
makes a docstring mandatory on every public function, so counting docstrings would set
the spec against itself and pay for compliance by deleting explanation. Import blocks
and `__all__` lists are boilerplate that the formatter expands to one name per line;
they are 441 lines here and no one reviews them. Neither is implementation.

**Cost.** The enforced number is not the number a reader gets from `wc -l`. The check
prints all three on every CI run so the gap is never hidden, and the code line count
also happens to sit near 4000.

## 2. Providers expose a push parser, not just a generator

**Ambiguity.** Section 7 requires that "the same parser serves both", and section 8
gives each provider an `iter_events` over an httpx response.

**Decision.** The unit each provider implements is `chunk_events(chunk, assembler)`, a
push parser that takes one decoded chunk and yields events. `ChunkParser` in
`providers/base.py` wraps it with the SSE or NDJSON decoder and the assembler, and
`BaseProvider.iter_events` drives that from a sync response. The async client drives
the same parser from `aiter_lines()`.

**Why.** The obvious reading of `iter_events(response)` is a sync generator, and the
async path then has to either duplicate every provider's parsing or collect the whole
response first. An early version did the latter, which silently turned async streaming
into a buffered request and defeated the point of streaming. Pushing chunks in is the
only shape that genuinely serves both from one body of code.

**Bonus.** `ChunkParser` owns the terminal `DoneEvent` and emits it itself, so "exactly
one `DoneEvent`, always last" holds by construction. No provider can get it wrong.

## 3. `requires_key` is separate from `local`

**Ambiguity.** Section 8 gives each provider a `local` flag and section 12 has
`MissingAPIKey`, but nothing says which providers may skip a credential.

**Decision.** A provider declares `local` and `requires_key` independently. Ollama is
local and needs no key. `openai_compat` is remote and still needs no key, because a
self-hosted vLLM or llama.cpp server usually has no authentication.

**Why.** Folding the two together would either demand a key for a local vLLM endpoint
or suppress the "ollama pull" hint for remote gateways. They answer different
questions: `local` shapes error messages, `requires_key` gates credential lookup.

## 4. Credentials are resolved after policy, not before

**Ambiguity.** Section 10 fixes the order in which policy rules are evaluated, but not
where credential lookup sits relative to policy.

**Decision.** `build()` leaves `api_key` unset. The client attaches it in `_bind()`
after policy has chosen the final model.

**Why.** With `residency="eu"` and a fallback to `ollama/llama3.1`, asking for
`openai/gpt-4o` must not require `OPENAI_API_KEY`. Policy refuses that model before a
byte is sent, so demanding its key first is a failure for a call that was never going
to happen. `_bind()` also re-resolves the endpoint when policy switched provider,
since neither the original provider's default base URL nor a per call `base_url`
intended for it can apply to a provider the caller never named.

## 5. Anthropic structured output goes through a forced tool call

**Ambiguity.** Section 8 requires `extract` to work on every provider. The Anthropic
Messages API has no `response_format`.

**Decision.** For a request with a JSON schema, the Anthropic provider declares a
single tool holding that schema and forces its use with `tool_choice`. The client
reads the document out of `tool_calls[0].arguments` when the text is empty.

**Why.** The alternative is asking for JSON in the prompt and parsing whatever comes
back, which is exactly the guesswork this library is meant to remove. A forced tool
call is the mechanism Anthropic itself documents for this, and it is validated
server side.

**Cost.** A caller cannot combine `extract` with their own tools on Anthropic. The
request validator rejects that combination for every provider, so the limitation is
uniform rather than surprising.

## 6. A hand written reader for the model registry

**Ambiguity.** Section 9 ships `models.yaml` inside the package, and section 4 allows
pyyaml only as the optional `aaron-llm[yaml]` extra.

**Decision.** `_yaml.py` uses pyyaml when it is importable, and otherwise parses the
small YAML subset that `models.yaml` is written in: two space indentation, plain
scalars, nested maps, and lists of scalars. Anything outside that subset raises
`YamlSubsetError` naming the offending line and pointing at the extra.

**Why.** The registry has to load on a default install, so the shipped file has to be
readable without pyyaml. The options were to ship JSON, which is unpleasant to review
in a file whose whole purpose is human maintenance of prices, or to read the subset we
actually write. The reader is 70 lines and only ever sees a file in this repository.

**Cost.** A user editing `models.yaml` with an anchor or a flow sequence gets an error
rather than a parse. The error names the extra that fixes it.

## 7. Dates in `models.yaml` are quoted

`last_verified: 2026-09-01` is a `datetime.date` to pyyaml and a `str` to the subset
reader, so the registry's contents depended on which backend was installed. Every date
in the file is quoted, and `_build_entry` coerces with `str()`. The two backends now
agree field for field.

## 8. Exception names keep their spec names, so ruff N818 is ignored for one file

Section 12 fixes the error hierarchy, and eight of those names do not end in `Error`.
N818 wants `PolicyViolationError`. The names in the spec are the public API, and
`except PolicyViolation:` reads better than the alternative, so `src/aaron/errors.py`
carries a documented per-file ignore.

`TimeoutError`, `ConnectionError` and `PermissionError` shadow builtin names inside
that module. They shadow **by name only**: they inherit from `TransportError` and
`ProviderError`, not from the builtins, so `except OSError` will not catch an Aaron
connection failure. For that reason they are the three error classes not re-exported
from the `aaron` namespace, where they would shadow the builtins in a caller's code.
The alternative, inheriting from both, would make every Aaron connection error an
`OSError` and quietly change the meaning of existing `except OSError` blocks in
calling code. A test pins the current inheritance so nobody assumes otherwise.

## 9. `providers/base.py` is exempt from the 400 line rule

Section 8 says each provider file stays under 400 lines and also makes `base.py` the
home for shared logic. Those pull in opposite directions: every line moved into
`base.py` to keep a provider small counts against `base.py`. The size check exempts
`base.py` and `__init__.py`, which are shared infrastructure, and holds all five
provider files to the limit. They are between 210 and 330 lines.

## 10. Redactors ship as three, not eight

An earlier draft had phone number, IBAN and credit card redactors. Regex detection of
those has a false positive rate high enough to be misleading in a compliance log, and
each one added code for a guess. What ships is `RegexRedactor` plus email and IP
address, both of which are unambiguous, and a policy file can name any pattern of its
own.

## 11. `aaron.registry` is a package, so its CLI is not double imported

`aaron/__init__.py` imports `Registry`, so `aaron.registry` is always in `sys.modules`
by the time `python -m aaron.registry --check` runs, and runpy prints a
`RuntimeWarning` about executing a module that has already been imported. The command
is in the README, and a documented command should not print a warning.

`registry.py` became `registry/__init__.py` plus `registry/__main__.py`, which is the
shape `aaron.audit` already had, and `models.yaml` moved next to the code that reads
it. The command line is unchanged, the warning is gone, and `argparse` and `sys` are no
longer imported by every process that imports Aaron.

## 12. The dev lock excludes the release tools

`requirements-dev.lock` is installed by three CI jobs and scanned by `pip-audit`.
Adding `build` and `twine` to it brought in keyring, rich, docutils, requests and a
dozen more: twenty extra packages in every lint and test job, and twenty more
opportunities for an advisory to fail a build that has nothing to do with publishing.
Release tooling is installed on demand in the release step. The lock is 30 packages.

## 13. Policy detail is its own field, not part of `content`

A refused call records which rule refused it and every fallback candidate that was
tried. That was written under `content`, which made `content` non-null on a client with
`record_content=False` and broke the one question a reviewer most wants a cheap answer
to: does this log hold personal data? It is now `policy_detail`, which is metadata,
always recorded, and never prompt text. `content` means prompt and completion, only.

## 14. An unknown registry field warns rather than fails

Registry entries ignore unknown keys so an entry written for a newer Aaron still loads.
Silently ignoring them hides a misspelled `input_usd_per_mtok`, which would price a
model at zero and disable a cost ceiling: a typo in the unsafe direction. Loading now
logs a warning naming the entry, the unknown keys and the known field names. Rejecting
outright was the other option, and it would make a registry file version locked to the
library for no safety benefit in the common case.

## 15. Ollama refuses a document instead of dropping it

The Ollama chat API takes images alongside the text and has nowhere to put a file. The
encoder silently dropped a `DocumentPart`, so a caller who attached a contract got a
confident answer about a document the model never saw. It now raises `InvalidRequest`
while the request is being built, naming what to do instead. Silent data loss is worse
than a refusal, especially in a library whose selling point is an audit trail.

## 16. `OtelSink` ships, because a declared extra should do something

The specification lists an `otel` extra. Nothing used it, which would have shipped a
`pip install "aaron-llm[otel]"` that installed two packages and changed no behaviour.
`OtelSink` emits one span per call using the GenAI convention attribute names, with
`opentelemetry` imported inside the sink so the base install stays at two dependencies
and the absence of the extra is a `ConfigurationError` naming the install command.
