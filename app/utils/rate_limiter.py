import asyncio
from collections import deque
from time import monotonic

from app.config import Settings
from app.monitoring.logger import get_logger


class AsyncRateLimiter:
    def __init__(
        self,
        *,
        max_calls: int,
        window_seconds: float = 60.0,
        enabled: bool = True,
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
        self.settings = settings
        self.wait_alert_threshold_seconds = wait_alert_threshold_seconds
        self._last_wait_alert_at = 0.0
        self._calls: deque[float] = deque()
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
                    self._calls.append(now)
                    return total_wait

                wait_seconds = max(0.0, self.window_seconds - (now - self._calls[0]))
                self._log_wait(endpoint=endpoint, wait_seconds=wait_seconds)

            await self._alert_wait_if_needed(endpoint=endpoint, wait_seconds=wait_seconds)
            await asyncio.sleep(wait_seconds)
            total_wait += wait_seconds

    def _drop_expired(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

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
_alpaca_limiter_key: tuple[bool, int] | None = None


def get_alpaca_rate_limiter(settings: Settings) -> AsyncRateLimiter:
    global _alpaca_limiter, _alpaca_limiter_key
    key = (settings.alpaca_rate_limit_enabled, settings.alpaca_max_calls_per_minute)
    if _alpaca_limiter is None or _alpaca_limiter_key != key:
        _alpaca_limiter = AsyncRateLimiter(
            max_calls=settings.alpaca_max_calls_per_minute,
            enabled=settings.alpaca_rate_limit_enabled,
            settings=settings,
        )
        _alpaca_limiter_key = key
    return _alpaca_limiter
