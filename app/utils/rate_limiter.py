import asyncio
from collections import Counter, deque
from time import monotonic
from typing import Any

from app.config import Settings
from app.monitoring.logger import get_logger


class AsyncRateLimiter:
    def __init__(
        self,
        *,
        max_calls: int,
        window_seconds: float = 60.0,
        enabled: bool = True,
        target_calls: int | None = None,
        settings: Settings | None = None,
        wait_alert_threshold_seconds: float = 2.0,
    ) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.enabled = enabled
        self.target_calls = min(target_calls or max_calls, max_calls)
        self.settings = settings
        self.wait_alert_threshold_seconds = wait_alert_threshold_seconds
        self._last_wait_alert_at = 0.0
        self._calls: deque[tuple[float, str]] = deque()
        self._last_wait_seconds = 0.0
        self._total_wait_seconds = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self, *, endpoint: str) -> float:
        if not self.enabled:
            return 0.0

        total_wait = 0.0
        while True:
            async with self._lock:
                now = monotonic()
                self._drop_expired(now)
                if len(self._calls) < self.max_calls:
                    self._calls.append((now, endpoint))
                    self._last_wait_seconds = total_wait
                    return total_wait

                wait_seconds = max(0.0, self.window_seconds - (now - self._calls[0][0]))
                self._log_wait(endpoint=endpoint, wait_seconds=wait_seconds)

            await self._alert_wait_if_needed(endpoint=endpoint, wait_seconds=wait_seconds)
            await asyncio.sleep(wait_seconds)
            total_wait += wait_seconds
            self._total_wait_seconds += wait_seconds

    def _drop_expired(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._calls and self._calls[0][0] <= cutoff:
            self._calls.popleft()

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "calls_last_minute": 0,
                "budget_remaining": None,
                "endpoint_counts": {},
                "limiter_wait_seconds": 0.0,
                "api_budget_status": "disabled",
                "target_calls_per_minute": self.target_calls,
                "hard_stop_calls_per_minute": self.max_calls,
            }
        now = monotonic()
        self._drop_expired(now)
        endpoint_counts = Counter(endpoint for _, endpoint in self._calls)
        calls = len(self._calls)
        status = "ok"
        if calls >= self.max_calls:
            status = "hard_stop"
        elif calls >= self.target_calls:
            status = "soft_limit"
        return {
            "enabled": True,
            "calls_last_minute": calls,
            "budget_remaining": max(0, self.max_calls - calls),
            "endpoint_counts": dict(endpoint_counts),
            "limiter_wait_seconds": self._last_wait_seconds,
            "total_limiter_wait_seconds": self._total_wait_seconds,
            "api_budget_status": status,
            "target_calls_per_minute": self.target_calls,
            "hard_stop_calls_per_minute": self.max_calls,
        }

    def soft_budget_reached(self, *, required_calls: int = 1) -> bool:
        if not self.enabled:
            return False
        now = monotonic()
        self._drop_expired(now)
        return len(self._calls) + required_calls > self.target_calls

    def hard_budget_reached(self, *, required_calls: int = 1) -> bool:
        if not self.enabled:
            return False
        now = monotonic()
        self._drop_expired(now)
        return len(self._calls) + required_calls > self.max_calls

    def _log_wait(self, *, endpoint: str, wait_seconds: float) -> None:
        try:
            get_logger().event(
                "alpaca_rate_limit_wait",
                endpoint=endpoint,
                wait_seconds=wait_seconds,
                max_calls=self.max_calls,
                window_seconds=self.window_seconds,
            )
        except Exception:
            pass

    async def _alert_wait_if_needed(self, *, endpoint: str, wait_seconds: float) -> None:
        if self.settings is None or wait_seconds < self.wait_alert_threshold_seconds:
            return
        now = monotonic()
        if now - self._last_wait_alert_at < 60:
            return
        self._last_wait_alert_at = now
        if not (self.settings.discord_alerts_enabled and self.settings.discord_alert_on_error):
            return
        try:
            from app.notifications.discord import DiscordNotifier

            await DiscordNotifier(self.settings).error_alert(
                "alpaca.rate_limiter",
                f"waiting {wait_seconds:.2f}s before {endpoint}; max_calls_per_minute={self.max_calls}",
            )
        except Exception:
            pass


_alpaca_limiter: AsyncRateLimiter | None = None
_alpaca_limiter_key: tuple[bool, int, int, int] | None = None


def get_alpaca_rate_limiter(settings: Settings) -> AsyncRateLimiter:
    global _alpaca_limiter, _alpaca_limiter_key
    hard_stop = min(settings.alpaca_max_calls_per_minute, settings.alpaca_api_budget_hard_stop_per_minute)
    target = min(settings.alpaca_api_budget_target_per_minute, hard_stop)
    key = (
        settings.alpaca_rate_limit_enabled,
        settings.alpaca_max_calls_per_minute,
        settings.alpaca_api_budget_target_per_minute,
        settings.alpaca_api_budget_hard_stop_per_minute,
    )
    if _alpaca_limiter is None or _alpaca_limiter_key != key:
        _alpaca_limiter = AsyncRateLimiter(
            max_calls=hard_stop,
            target_calls=target,
            enabled=settings.alpaca_rate_limit_enabled,
            settings=settings,
        )
        _alpaca_limiter_key = key
    return _alpaca_limiter
