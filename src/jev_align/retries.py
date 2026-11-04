"""Shared bounded retry delays for evaluation providers."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping
from datetime import timezone
from email.utils import parsedate_to_datetime

RETRYABLE_HTTP_STATUSES = frozenset({408, 429, *range(500, 600)})

DEFAULT_ATTEMPTS = 4  # Includes the initial request.
INITIAL_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 30.0
MAX_RETRY_AFTER_SECONDS = 60.0


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    values = {key.lower(): value for key, value in (headers or {}).items()}
    for name, divisor in (("retry-after", 1.0), ("retry-after-ms", 1000.0)):
        value = values.get(name)
        if value is None:
            continue
        try:
            delay = float(value) / divisor
        except (TypeError, ValueError):
            if name != "retry-after":
                continue
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                delay = max(0.0, date.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(delay) and delay >= 0:
            return delay
    return None


def retry_delay(
    attempt: int,
    headers: Mapping[str, str] | None = None,
    *,
    initial_delay: float = INITIAL_BACKOFF_SECONDS,
) -> float | None:
    """Wait after a failed zero-based attempt, or stop if cooldown exceeds a minute.

    Never shorten a valid server cooldown. Missing or invalid headers use capped
    exponential backoff plus up to 25% jitter. The caller enforces attempt limits.
    """
    delay = _retry_after(headers)
    if delay is not None:
        return delay if delay <= MAX_RETRY_AFTER_SECONDS else None
    delay = min(initial_delay * 2.0 ** min(attempt, 30), MAX_BACKOFF_SECONDS)
    return min(delay + random.uniform(0.0, delay * 0.25), MAX_BACKOFF_SECONDS)
