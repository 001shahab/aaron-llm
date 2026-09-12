# Aaron (`aaron-llm`) — working agreement

One client for every model, with a policy and an audit trail.

Copyright (c) 2026 3S Holding OU. Author: Prof. Shahab Anbarjafari <shb@3sholding.com>.

## Authorship (non-negotiable)

Prof. Shahab Anbarjafari is the sole author, committer and contributor. Commit as
`Prof. Shahab Anbarjafari <shb@3sholding.com>` and never add `Co-Authored-By`,
`Signed-off-by` or any AI attribution trailer to a commit, tag or release note.
See `.cursor/rules/solo-authorship.mdc`.

## Hard constraints

- Runtime dependencies are exactly `httpx` and `pydantic`. Nothing else, ever.
- No vendor SDKs. Providers are written against documented HTTP APIs.
- Optional extras (`pyyaml`, OpenTelemetry) are imported lazily inside functions,
  never at module import time.
- Each file in `src/aaron/providers/` stays self-contained and under 400 lines.
  Shared logic goes in `providers/base.py`.
- Total source under 4000 lines excluding tests and YAML.
- No network calls in the test suite. Mock with `respx`, fixtures in `tests/fixtures/`.
- API keys never appear in a log, an exception, a `__repr__` or an audit record.

## Gates

```sh
.venv/bin/ruff format --check . && .venv/bin/ruff check .
.venv/bin/mypy --strict src/aaron
.venv/bin/pytest --cov=src/aaron --cov-fail-under=90
```

## File header

Every source file starts with:

```python
# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>
```
