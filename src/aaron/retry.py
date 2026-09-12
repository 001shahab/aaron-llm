# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""The backoff policy.

Only failures that a later attempt could plausibly survive are retried, and only
on non streaming calls. A streaming call that has already handed a text event to
the caller is never retried, because the caller has already seen output that a
second attempt would contradict.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .errors import (
    AaronError,
    ConnectionError,
    RateLimitError,
    ServerError,
    ServiceOverloaded,
    TimeoutError,
)

# The only error classes worth another attempt.
RETRYABLE: tuple[type[AaronError], ...] = (
    RateLimitError,
    ServerError,
    ServiceOverloaded,
    ConnectionError,
    TimeoutError,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Attributes:
        attempts: Total attempts including the first, so 1 disables retrying.
        initial_delay: Seconds before the second attempt, before jitter.
        factor: Multiplier applied per attempt.
        max_delay: Ceiling for a single sleep, after jitter.
        respect_retry_after: Honour a provider's ``Retry-After`` even when it is
            longer than the computed backoff.
    """

    attempts: int = 3
    initial_delay: float = 0.5
    factor: float = 2.0
    max_delay: float = 30.0
    respect_retry_after: bool = True

    def should_retry(self, error: BaseException, *, attempt: int) -> bool:
        """Whether to make another attempt after this failure.

        Args:
            error: The failure from the attempt just made.
            attempt: Which attempt failed, counting from 1.

        Returns:
            True when the error class is retryable and attempts remain.
        """
        if attempt >= self.attempts:
            return False
        return isinstance(error, RETRYABLE)

    def delay_for(self, attempt: int, error: BaseException | None = None) -> float:
        """Seconds to sleep before the next attempt.

        Full jitter is used, that is a uniform draw over the whole window, because
        it spreads a thundering herd better than equal jitter does.

        Args:
            attempt: Which attempt just failed, counting from 1.
            error: The failure, consulted for a ``Retry-After`` hint.

        Returns:
            A non negative number of seconds, capped at ``max_delay``.
        """
        window = min(self.initial_delay * self.factor ** (attempt - 1), self.max_delay)
        # Jitter, not cryptography, so the default random source is the right one.
        delay = random.uniform(0, window)
        if (
            self.respect_retry_after
            and isinstance(error, RateLimitError)
            and error.retry_after is not None
        ):
            delay = max(delay, min(float(error.retry_after), self.max_delay))
        return delay
