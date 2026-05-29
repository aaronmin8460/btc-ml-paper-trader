import asyncio
from datetime import UTC, datetime, timedelta

from app.config import Settings, get_settings
from app.monitoring.logger import get_logger
from app.services.trader import KILL_SWITCH_REASONS


class TradingScheduler:
    def __init__(self, trader, settings: Settings | None = None) -> None:
        self.trader = trader
        self.settings = settings or get_settings()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.logger = get_logger()
        self._paused = False
        self._pause_reason: str | None = None
        self._paused_at: datetime | None = None
        self._risk_block_events: dict[str, list[datetime]] = {}
        self._runtime_error_events: list[datetime] = []

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def pause_reason(self) -> str | None:
        return self._pause_reason

    @property
    def paused_at(self) -> datetime | None:
        return self._paused_at

    def start(self) -> bool:
        if self.running:
            return False
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        return True

    async def stop(self) -> bool:
        if not self.running:
            return False
        self._stop.set()
        await self._task
        return True

    async def pause(self, reason: str = "manual_pause", *, send_alert: bool = False) -> bool:
        if self._paused:
            return False
        self._paused = True
        self._pause_reason = reason
        self._paused_at = datetime.now(UTC)
        self.logger.event("auto_trading_paused", reason=reason, paused_at=self._paused_at.isoformat())
        if send_alert:
            await self._send_pause_alert(reason)
        return True

    def resume(self) -> bool:
        was_paused = self._paused
        self._paused = False
        self._pause_reason = None
        self._paused_at = None
        self._risk_block_events.clear()
        self._runtime_error_events.clear()
        self.logger.event("auto_trading_resumed", resumed=was_paused)
        return was_paused

    def status(self) -> dict:
        return {
            "running": self.running,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "paused_at": self.paused_at,
            "auto_trade_enabled": self.settings.auto_trade_enabled,
            "trading_enabled": self.settings.trading_enabled,
            "circuit_breaker_enabled": self.settings.circuit_breaker_enabled,
        }

    async def run_pending_once(self) -> bool:
        if self.paused or not self.settings.auto_trade_enabled:
            return False
        try:
            result = await self.trader.run_once()
            await self.record_run_result(result)
            return True
        except Exception as exc:
            self.logger.event("runtime_error", component="scheduler", error=str(exc))
            await self.record_runtime_error(exc)
            return False

    async def record_run_result(self, result: dict | None) -> None:
        if not self.settings.circuit_breaker_enabled:
            return
        decision = (result or {}).get("decision") or {}
        action = str(decision.get("action") or "").lower()
        reason = str(decision.get("reason") or "")
        if action == "hold" and reason in KILL_SWITCH_REASONS:
            await self.record_risk_block(reason)

    async def record_risk_block(self, reason: str) -> None:
        if not self.settings.circuit_breaker_enabled or self.settings.max_same_risk_blocks_before_pause <= 0:
            return
        now = datetime.now(UTC)
        events = self._pruned_events(self._risk_block_events.get(reason, []), now)
        events.append(now)
        self._risk_block_events[reason] = events
        if len(events) >= self.settings.max_same_risk_blocks_before_pause:
            await self.pause(f"repeated_risk_block:{reason}", send_alert=True)

    async def record_runtime_error(self, error: Exception) -> None:
        if not self.settings.circuit_breaker_enabled or self.settings.max_runtime_errors_before_pause <= 0:
            return
        now = datetime.now(UTC)
        self._runtime_error_events = self._pruned_events(self._runtime_error_events, now)
        self._runtime_error_events.append(now)
        if len(self._runtime_error_events) >= self.settings.max_runtime_errors_before_pause:
            await self.pause("repeated_runtime_errors", send_alert=True)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await self.run_pending_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.settings.scan_interval_seconds)
            except asyncio.TimeoutError:
                continue

    def _pruned_events(self, events: list[datetime], now: datetime) -> list[datetime]:
        cutoff = now - timedelta(seconds=self.settings.circuit_breaker_window_seconds)
        return [event_at for event_at in events if event_at >= cutoff]

    async def _send_pause_alert(self, reason: str) -> None:
        notifier = getattr(self.trader, "notifier", None)
        alert = getattr(notifier, "auto_trading_paused_alert", None)
        if alert is None:
            return
        try:
            await alert(reason)
        except Exception as exc:
            try:
                self.logger.event("discord_alert_failed", alert_type="auto_pause", error_type=type(exc).__name__)
            except Exception:
                pass
