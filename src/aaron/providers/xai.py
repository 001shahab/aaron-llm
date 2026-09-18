# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""xAI, against ``POST /v1/chat/completions`` on ``https://api.x.ai/v1``.

xAI serves the same chat completions shape as OpenAI, so this is the OpenAI
provider with a different endpoint and a different credential rather than another
four hundred lines saying the same thing.

Grok reasons by default at effort ``high``. Unlike the OpenAI reasoning families it
still accepts ``max_tokens``, ``temperature`` and ``top_p``, so nothing is stripped
here; the effort itself is a provider specific field:

```python
reply = client.chat(
    "xai/grok-4.6",
    "Summarise this contract",
    provider_options={"reasoning_effort": "low"},
)
```
"""

from __future__ import annotations

from .openai import OpenAIProvider


class XAIProvider(OpenAIProvider):
    """The OpenAI chat completions translator pointed at xAI."""

    name = "xai"
    default_base_url = "https://api.x.ai/v1"
    env_key = "XAI_API_KEY"
    local = False
    requires_key = True

    @staticmethod
    def is_reasoning_model(model_name: str) -> bool:
        """No Grok model takes the OpenAI reasoning parameter set."""
        return False
