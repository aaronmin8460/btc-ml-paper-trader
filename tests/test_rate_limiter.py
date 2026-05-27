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
