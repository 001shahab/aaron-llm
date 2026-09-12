# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Any OpenAI compatible endpoint, with a mandatory explicit ``base_url``.

This is the whole reason Aaron does not need five more provider files. Groq,
Mistral, Together, DeepSeek, Fireworks, OpenRouter, vLLM, LM Studio, llama.cpp's
server and most self hosted gateways all speak ``/chat/completions``, so they are
reached like this:

```python
client = Aaron(base_urls={"openai_compat": "https://api.groq.com/openai/v1"},
               api_keys={"openai_compat": os.environ["GROQ_API_KEY"]})
reply = client.chat("openai_compat/llama-3.3-70b-versatile", "Hello")
```

or per call:

```python
reply = client.chat(
    "openai_compat/mistral-large-latest",
    "Hello",
    base_url="https://api.mistral.ai/v1",
)
```

Two things differ from :mod:`aaron.providers.openai`. There is no default endpoint,
so forgetting ``base_url`` is a configuration error rather than a call to OpenAI by
accident. And no credential is required, because a local vLLM or llama.cpp server
usually has none.

Compatibility is never complete. Some endpoints ignore ``response_format``, some
omit the usage object, and some reject ``stream_options``. Usage is estimated when
it is missing, which shows up as ``Cost.estimated = True``; anything else that a
particular endpoint needs goes through ``provider_options``.
"""

from __future__ import annotations

from .openai import OpenAIProvider


class OpenAICompatProvider(OpenAIProvider):
    """The OpenAI provider without a default endpoint and without a required key."""

    name = "openai_compat"
    default_base_url = ""
    env_key = "OPENAI_COMPAT_API_KEY"
    local = False
    requires_key = False
