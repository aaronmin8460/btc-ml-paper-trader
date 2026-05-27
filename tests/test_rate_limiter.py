import time

import pytest

from app.utils.rate_limiter import AsyncRateLimiter


@pytest.mark.anyio
async def test_rate_limiter_waits_when_budget_is_exhausted():
    limiter = AsyncRateLimiter(max_calls=1, window_seconds=0.05)

    await limiter.acquire(endpoint="test")
    start = time.monotonic()
    waited = await limiter.acquire(endpoint="test")

    elapsed = time.monotonic() - start
    assert waited >= 0.04
    assert elapsed >= 0.04


@pytest.mark.anyio
async def test_rate_limiter_can_be_disabled():
    limiter = AsyncRateLimiter(max_calls=1, window_seconds=10, enabled=False)

    await limiter.acquire(endpoint="test")
    waited = await limiter.acquire(endpoint="test")

    assert waited == 0


@pytest.mark.anyio
async def test_rate_limiter_tracks_endpoint_usage_and_budget():
    limiter = AsyncRateLimiter(max_calls=3, target_calls=2, window_seconds=10)

    await limiter.acquire(endpoint="latest_quote")
    await limiter.acquire(endpoint="position")

    snapshot = limiter.snapshot()

    assert snapshot["calls_last_minute"] == 2
    assert snapshot["budget_remaining"] == 1
    assert snapshot["endpoint_counts"] == {"latest_quote": 1, "position": 1}
    assert snapshot["api_budget_status"] == "soft_limit"


@pytest.mark.anyio
async def test_rate_limiter_hard_budget_check_stays_under_stop():
    limiter = AsyncRateLimiter(max_calls=2, target_calls=1, window_seconds=10)

    await limiter.acquire(endpoint="latest_quote")
    await limiter.acquire(endpoint="position")

    assert limiter.hard_budget_reached() is True
    assert limiter.snapshot()["calls_last_minute"] == 2
