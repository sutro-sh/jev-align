from __future__ import annotations

from datetime import datetime, timezone
from email.utils import format_datetime

import pytest

from jev_align import retries


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"Retry-After": "5"}, 5.0),
        ({"retry-after": "0.25"}, 0.25),
        ({"RETRY-AFTER": "0"}, 0.0),
        ({"Retry-After-Ms": "250"}, 0.25),
        ({"Retry-After": "2", "retry-after-ms": "500"}, 2.0),
        ({"Retry-After": "invalid", "retry-after-ms": "500"}, 0.5),
        ({"Retry-After": "61"}, None),
    ],
)
def test_server_retry_delays(headers, expected):
    assert retries.retry_delay(0, headers) == expected


@pytest.mark.parametrize("offset,expected", [(10, 10), (-10, 0), (61, None)])
def test_http_date_retry_after(monkeypatch, offset, expected):
    monkeypatch.setattr(retries.time, "time", lambda: 1_700_000_000)
    date = datetime.fromtimestamp(1_700_000_000 + offset, timezone.utc)
    header = format_datetime(date, usegmt=True)
    assert retries.retry_delay(0, {"Retry-After": header}) == expected


@pytest.mark.parametrize("value", ["invalid", "", "-1", "nan", "inf", "1e999"])
def test_invalid_retry_after_uses_backoff(monkeypatch, value):
    monkeypatch.setattr(retries.random, "uniform", lambda *_args: 0)
    assert retries.retry_delay(1, {"Retry-After": value}) == 1.0


def test_exponential_backoff_and_jitter_are_bounded(monkeypatch):
    monkeypatch.setattr(retries.random, "uniform", lambda low, high: high)
    assert [retries.retry_delay(attempt) for attempt in range(3)] == [0.625, 1.25, 2.5]
    assert retries.retry_delay(100) == 30
