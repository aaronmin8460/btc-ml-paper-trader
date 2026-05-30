from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Callable

import pandas as pd

from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.data.market_data import MarketDataClient
from app.ml.train import train_model_from_bars
from app.monitoring.logger import get_logger
from app.notifications.discord import DiscordNotifier
from app.risk.risk_manager import account_state_from_payload


Trainer = Callable[..., dict]


class TrainingScheduler:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        market_data: MarketDataClient | None = None,
        broker: Any | None = None,
        notifier: DiscordNotifier | None = None,
        trainer: Trainer = train_model_from_bars,
    ) -> None:
        self.settings = settings or get_settings()
        self.market_data = market_data or MarketDataClient(self.settings)
        self.broker = broker
        self.notifier = notifier or DiscordNotifier(self.settings)
        self.trainer = trainer
        self.logger = get_logger()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self._last_started_at: datetime | None = None
        self._last_finished_at: datetime | None = None
        self._last_status: str | None = None
        self._last_reason: str | None = None
        self._last_model_path: str | None = None
        self._last_accepted: bool | None = None
        self._last_metrics: dict[str, Any] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def in_progress(self) -> bool:
        return self._lock.locked()

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

    async def run_now(self) -> dict[str, Any]:
        return await self.run_once(trigger="manual")

    async def run_once(self, *, trigger: str = "scheduled") -> dict[str, Any]:
        if self._lock.locked():
            return {
                **self.status(),
                "started": False,
                "last_training_status": "busy",
                "last_training_reason": "training_already_running",
            }

        async with self._lock:
            self._last_started_at = datetime.now(UTC)
            self._last_finished_at = None
            self._last_status = "running"
            self._last_reason = None
            self._last_model_path = None
            self._last_accepted = None
            self._last_metrics = None
            self.logger.event("training_started", symbol=ALLOWED_SYMBOL, trigger=trigger)

            try:
                bars = await self._fetch_training_bars()
                starting_equity = await self._starting_equity()
                result = await asyncio.to_thread(
                    self.trainer,
                    bars,
                    self.settings,
                    starting_equity=starting_equity,
                )
            except Exception as exc:
                self._record_finished(
                    status="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    model_path=None,
                    accepted=False,
                    metrics=None,
                )
                self.logger.event(
                    "training_failed",
                    symbol=ALLOWED_SYMBOL,
                    reason=self._last_reason,
                )
                await self._send_failure_alert(exc)
                return {**self.status(), "started": True}

            accepted = bool(result.get("accepted"))
            reason = str(result.get("reason") or ("accepted" if accepted else "rejected"))
            model_path = result.get("model_path")
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else None
            self._record_finished(
                status="accepted" if accepted else "rejected",
                reason=reason,
                model_path=str(model_path) if model_path else None,
                accepted=accepted,
                metrics=metrics,
            )
            self.logger.event(
                "training_finished",
                symbol=ALLOWED_SYMBOL,
                status=self._last_status,
                reason=reason,
                model_path=self._last_model_path,
                accepted=accepted,
            )
            await self._send_model_alert()
            return {**self.status(), "started": True, "result": result}

    def status(self) -> dict[str, Any]:
        return {
            "auto_train_enabled": self.settings.auto_train_enabled,
            "running": self.running,
            "in_progress": self.in_progress,
            "last_training_started_at": self._last_started_at,
            "last_training_finished_at": self._last_finished_at,
            "last_training_status": self._last_status,
            "last_training_reason": self._last_reason,
            "last_training_model_path": self._last_model_path,
            "last_training_accepted": self._last_accepted,
            "last_training_metrics": self._last_metrics,
        }

    async def _loop(self) -> None:
        if await self._sleep_or_stopped(self.settings.auto_train_startup_delay_seconds):
            return
        while not self._stop.is_set():
            await self.run_once(trigger="scheduled")
            if await self._sleep_or_stopped(self.settings.auto_train_interval_seconds):
                return

    async def _sleep_or_stopped(self, seconds: int | float) -> bool:
        timeout = max(0.0, float(seconds))
        if timeout == 0:
            return self._stop.is_set()
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _fetch_training_bars(self) -> pd.DataFrame:
        limit = max(
            int(self.settings.auto_train_min_bars),
            int(self.settings.min_training_rows) + 200,
        )
        return await self.market_data.fetch_bars(ALLOWED_SYMBOL, limit=limit)

    async def _starting_equity(self) -> float | None:
        if self.broker is None:
            return None
        credentials_available = getattr(self.broker, "credentials_available", lambda: True)
        try:
            if not credentials_available():
                return None
            account = await self.broker.get_account()
        except Exception:
            return None
        account_state = account_state_from_payload(account)
        return account_state.equity or account_state.portfolio_value

    def _record_finished(
        self,
        *,
        status: str,
        reason: str,
        model_path: str | None,
        accepted: bool | None,
        metrics: dict[str, Any] | None,
    ) -> None:
        self._last_finished_at = datetime.now(UTC)
        self._last_status = status
        self._last_reason = reason
        self._last_model_path = model_path
        self._last_accepted = accepted
        self._last_metrics = metrics

    async def _send_model_alert(self) -> None:
        if not self.settings.auto_train_send_discord_alerts:
            return
        try:
            await self.notifier.model_alert(
                self._last_model_path or "none",
                accepted=bool(self._last_accepted),
                reason=self._last_reason or "unknown",
                metrics=self._last_metrics,
                force=True,
            )
        except Exception as exc:
            self._log_alert_failure(exc)

    async def _send_failure_alert(self, error: Exception) -> None:
        if not self.settings.auto_train_send_discord_alerts:
            return
        try:
            await self.notifier.error_alert("training_scheduler", error, force=True)
        except Exception as exc:
            self._log_alert_failure(exc)

    def _log_alert_failure(self, error: Exception) -> None:
        try:
            self.logger.event("discord_alert_failed", alert_type="training", error_type=type(error).__name__)
        except Exception:
            pass
