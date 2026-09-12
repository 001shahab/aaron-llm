# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Aaron: one client for every model, with a policy and an audit trail.

```python
from aaron import Aaron

client = Aaron()
reply = client.chat("ollama/llama3.1", "Summarise the EU AI Act in one sentence.")
print(reply.text, reply.usage.total_tokens, reply.cost.usd)
```

Importing this package opens no connections, reads no configuration files and sends
no telemetry. There is none to send: Aaron never phones home, and the only network
traffic it makes is the provider call you asked for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from ._version import __version__
from .audit import AuditRecord, CallbackSink, JsonlSink, NullSink, OtelSink
from .client import Aaron, AsyncAaron
from .errors import (
    AaronError,
    AuthenticationError,
    ConfigurationError,
    ContentFilterError,
    ContextLengthExceeded,
    InvalidRequest,
    LocalProviderUnavailable,
    MissingAPIKey,
    PolicyViolation,
    ProviderError,
    RateLimitError,
    RequestError,
    ServerError,
    ServiceOverloaded,
    ToolArgumentError,
    TransportError,
    UnknownModel,
    UnknownProvider,
)
from .policy import (
    EmailRedactor,
    IpAddressRedactor,
    Policy,
    RegexRedactor,
)
from .registry import Capabilities, Registry
from .request import ChatRequest
from .stream import AsyncStream, Stream, StreamEvent
from .types import (
    ContentPart,
    Cost,
    DocumentPart,
    ImagePart,
    Message,
    Response,
    TextPart,
    Tool,
    ToolCall,
    Usage,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    M = TypeVar("M", bound=BaseModel)

__all__ = [
    "Aaron",
    "AaronError",
    "AsyncAaron",
    "AsyncStream",
    "AuditRecord",
    "AuthenticationError",
    "CallbackSink",
    "Capabilities",
    "ChatRequest",
    "ConfigurationError",
    "ContentFilterError",
    "ContentPart",
    "ContextLengthExceeded",
    "Cost",
    "DocumentPart",
    "EmailRedactor",
    "ImagePart",
    "InvalidRequest",
    "IpAddressRedactor",
    "JsonlSink",
    "LocalProviderUnavailable",
    "Message",
    "MissingAPIKey",
    "NullSink",
    "OtelSink",
    "Policy",
    "PolicyViolation",
    "ProviderError",
    "RateLimitError",
    "RegexRedactor",
    "Registry",
    "RequestError",
    "Response",
    "ServerError",
    "ServiceOverloaded",
    "Stream",
    "StreamEvent",
    "TextPart",
    "Tool",
    "ToolArgumentError",
    "ToolCall",
    "TransportError",
    "UnknownModel",
    "UnknownProvider",
    "Usage",
    "__version__",
    "chat",
    "default_client",
    "extract",
    "set_default_client",
    "stream",
]

# The one piece of mutable module state in the library, and it is explicit.
_default_client: Aaron | None = None


def default_client() -> Aaron:
    """Return the module level client, creating it on first use.

    Returns:
        The shared :class:`Aaron` instance used by :func:`chat`, :func:`stream` and
        :func:`extract`.
    """
    global _default_client
    if _default_client is None:
        _default_client = Aaron()
    return _default_client


def set_default_client(client: Aaron | None) -> None:
    """Replace the module level client, or clear it so the next call rebuilds it.

    Use this to give the convenience functions a policy or an audit sink:

    ```python
    aaron.set_default_client(Aaron(policy=Policy(residency="eu"), audit="calls.jsonl"))
    ```

    Args:
        client: The client to use, or None to reset.
    """
    global _default_client
    _default_client = client


def chat(model: str | None = None, messages: Any = None, **kwargs: Any) -> Response:
    """Send one chat request using the module level client. See :meth:`Aaron.chat`."""
    return default_client().chat(model, messages, **kwargs)


def stream(model: str | None = None, messages: Any = None, **kwargs: Any) -> Stream:
    """Start a streaming call using the module level client. See :meth:`Aaron.stream`."""
    return default_client().stream(model, messages, **kwargs)


def extract(
    model: str | None = None, messages: Any = None, *, schema: type[Any], **kwargs: Any
) -> Any:
    """Get a validated object using the module level client. See :meth:`Aaron.extract`."""
    return default_client().extract(model, messages, schema=schema, **kwargs)
