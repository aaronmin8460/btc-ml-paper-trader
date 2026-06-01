import asyncio
from datetime import UTC, datetime, timedelta

from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.db.database import SessionLocal
from app.db.repository import Repository
from app.monitoring.logger import get_logger
from app.services.trader import KILL_SWITCH_REASONS


SCALPING_KILL_SWITCH_PREFIX = "scalping_kill_switch:"
SCALPING_HOURLY_LOSS_REASON = f"{SCALPING_KILL_SWITCH_PREFIX}hourly_loss_limit"
SCALPING_IOC_CANCEL_STREAK_REASON = f"{SCALPING_KILL_SWITCH_PREFIX}ioc_cancel_streak"
SCALPING_LOSS_STREAK_REASON = f"{SCALPING_KILL_SWITCH_PREFIX}loss_streak"
SCALPING_RUNTIME_ERRORS_REASON = f"{SCALPING_KILL_SWITCH_PREFIX}runtime_errors"


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
        self._resume_if_expired()
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
        self.restore_pause_state()
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
        self._persist_scalping_pause(reason=reason, paused_at=self._paused_at)
        self.logger.event("auto_trading_paused", reason=reason, paused_at=self._paused_at.isoformat())
        self.logger.event(
            "kill_switch_pause",
            symbol=ALLOWED_SYMBOL,
            reason=reason,
            paused_at=self._paused_at.isoformat(),
        )
        if send_alert:
            await self._send_pause_alert(reason)
        return True

    def resume(self) -> bool:
        was_paused = self._paused
        previous_reason = self._pause_reason
        self._paused = False
        self._pause_reason = None
        self._paused_at = None
        self._risk_block_events.clear()
        self._runtime_error_events.clear()
        if was_paused and _is_scalping_kill_switch_reason(previous_reason):
            self._persist_scalping_resume(reason=previous_reason or "manual_resume")
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
            self.logger.event(
                "runtime_error",
                component="scheduler",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            await self._send_runtime_error_alert(exc)
            await self.record_runtime_error(exc)
            return False

    async def record_run_result(self, result: dict | None) -> None:
        decision = (result or {}).get("decision") or {}
        action = str(decision.get("action") or "").lower()
        reason = str(decision.get("reason") or "")
        scalping_reason = str((result or {}).get("scalping_kill_switch_reason") or "")
        if not scalping_reason and _is_scalping_kill_switch_reason(reason):
            scalping_reason = reason
        if self._scalping_kill_switch_enabled() and _is_scalping_kill_switch_reason(scalping_reason):
            await self.pause(scalping_reason, send_alert=True)
            return
        if (
            self._scalping_kill_switch_enabled()
            and action == "hold"
            and reason == "max_consecutive_losses_reached"
        ):
            await self.pause(SCALPING_LOSS_STREAK_REASON, send_alert=True)
            return
        if not self.settings.circuit_breaker_enabled:
            return
        if action == "hold" and reason in KILL_SWITCH_REASONS:
            await self.record_risk_block(reason)

    async def record_risk_block(self, reason: str) -> None:
        if not self.settings.circuit_breaker_enabled or self.settings.max_same_risk_blocks_before_pause <= 0:
            return
        now = datetime.now(UTC)
        events = self._pruned_events(self._risk_block_events.get(reason, []), now)
        events.append(now)
        self._risk_block_events[reason] = events
        self.logger.event(
            "risk_block",
            symbol=ALLOWED_SYMBOL,
            reason=reason,
            relevant_limit=self.settings.max_same_risk_blocks_before_pause,
            current_value=len(events),
            reset_time=(now + timedelta(seconds=self.settings.circuit_breaker_window_seconds)).isoformat(),
        )
        if len(events) >= self.settings.max_same_risk_blocks_before_pause:
            await self.pause(f"repeated_risk_block:{reason}", send_alert=True)

    async def record_runtime_error(self, error: Exception) -> None:
        if (
            not self.settings.circuit_breaker_enabled
            and not self._scalping_kill_switch_enabled()
        ) or self.settings.max_runtime_errors_before_pause <= 0:
            return
        now = datetime.now(UTC)
        self._runtime_error_events = self._pruned_events(self._runtime_error_events, now)
        self._runtime_error_events.append(now)
        if len(self._runtime_error_events) >= self.settings.max_runtime_errors_before_pause:
            reason = SCALPING_RUNTIME_ERRORS_REASON if self._scalping_kill_switch_enabled() else "repeated_runtime_errors"
            await self.pause(reason, send_alert=True)

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

    def restore_pause_state(self, *, now: datetime | None = None) -> bool:
        if not self._scalping_kill_switch_enabled():
            return False
        current_time = _ensure_utc(now or datetime.now(UTC))
        try:
            with SessionLocal() as db:
                event = Repository(db).latest_scalping_kill_switch_event()
        except Exception as exc:
            self._log_persistence_failure("restore", exc)
            return False
        if event is None or event.event_type != "scalping_kill_switch_pause":
            return False
        paused_at = _ensure_utc(event.created_at)
        if current_time >= self._pause_expires_at(reason=event.reason, paused_at=paused_at):
            self._persist_scalping_resume(reason=event.reason, created_at=current_time)
            return False
        self._paused = True
        self._pause_reason = event.reason
        self._paused_at = paused_at
        self.logger.event("scalping_kill_switch_restored", reason=event.reason, paused_at=paused_at.isoformat())
        return True

    def _resume_if_expired(self) -> bool:
        if not self._paused or not _is_scalping_kill_switch_reason(self._pause_reason) or self._paused_at is None:
            return False
        if datetime.now(UTC) < self._pause_expires_at(reason=self._pause_reason, paused_at=self._paused_at):
            return False
        reason = self._pause_reason
        resumed = self.resume()
        if resumed:
            self.logger.event("scalping_kill_switch_expired", reason=reason)
        return resumed

    def _pause_expires_at(self, *, reason: str, paused_at: datetime) -> datetime:
        return _ensure_utc(paused_at) + timedelta(seconds=self._pause_duration_seconds(reason))

    def _pause_duration_seconds(self, reason: str) -> int:
        if reason == SCALPING_RUNTIME_ERRORS_REASON:
            return self.settings.scalping_pause_after_runtime_errors_seconds
        if reason == SCALPING_IOC_CANCEL_STREAK_REASON:
            return self.settings.ioc_cancel_escalation_cooldown_seconds
        return self.settings.scalping_pause_after_loss_streak_seconds

    def _scalping_kill_switch_enabled(self) -> bool:
        return self.settings.scalping_mode_enabled and self.settings.scalping_kill_switch_enabled

    def _persist_scalping_pause(self, *, reason: str, paused_at: datetime) -> None:
        if not _is_scalping_kill_switch_reason(reason):
            return
        self._persist_scalping_event(event_type="scalping_kill_switch_pause", reason=reason, created_at=paused_at)

    def _persist_scalping_resume(self, *, reason: str, created_at: datetime | None = None) -> None:
        self._persist_scalping_event(
            event_type="scalping_kill_switch_resume",
            reason=reason,
            created_at=created_at or datetime.now(UTC),
        )

    def _persist_scalping_event(self, *, event_type: str, reason: str, created_at: datetime) -> None:
        try:
            with SessionLocal() as db:
                Repository(db).add_risk_event(event_type=event_type, reason=reason, created_at=created_at)
        except Exception as exc:
            self._log_persistence_failure(event_type, exc)

    def _log_persistence_failure(self, operation: str, error: Exception) -> None:
        try:
            self.logger.event(
                "scalping_kill_switch_persistence_failed",
                operation=operation,
                error_type=type(error).__name__,
            )
        except Exception:
            pass

    async def _send_pause_alert(self, reason: str) -> None:
        notifier = getattr(self.trader, "notifier", None)
        alert = getattr(notifier, "kill_switch_pause_alert", None) or getattr(notifier, "auto_trading_paused_alert", None)
        if alert is None:
            return
        try:
            await alert(reason)
        except Exception as exc:
            try:
                self.logger.event("discord_alert_failed", alert_type="auto_pause", error_type=type(exc).__name__)
            except Exception:
                pass

    async def _send_runtime_error_alert(self, error: Exception) -> None:
        notifier = getattr(self.trader, "notifier", None)
        alert = getattr(notifier, "error_alert", None)
        if alert is None:
            return
        try:
            await alert("scheduler.run_pending_once", error)
        except Exception as exc:
            try:
                self.logger.event("discord_alert_failed", alert_type="runtime_error", error_type=type(exc).__name__)
            except Exception:
                pass


def _is_scalping_kill_switch_reason(reason: str | None) -> bool:
    return bool(reason and reason.startswith(SCALPING_KILL_SWITCH_PREFIX))


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
