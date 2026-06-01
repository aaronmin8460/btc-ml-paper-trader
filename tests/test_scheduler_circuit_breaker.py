from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.database import Base, connect_args_for_database_url
from app.db.repository import Repository
from app.services import scheduler as scheduler_module
from app.services.scheduler import (
    SCALPING_HOURLY_LOSS_REASON,
    SCALPING_IOC_CANCEL_STREAK_REASON,
    SCALPING_RUNTIME_ERRORS_REASON,
    TradingScheduler,
)


class FakeNotifier:
    def __init__(self) -> None:
        self.pause_alerts: list[str] = []

    async def auto_trading_paused_alert(self, reason: str) -> None:
        self.pause_alerts.append(reason)


class FakeTrader:
    def __init__(self, results: list[dict] | None = None, errors: list[Exception] | None = None) -> None:
        self.results = list(results or [])
        self.errors = list(errors or [])
        self.notifier = FakeNotifier()
        self.run_once_calls = 0

    async def run_once(self) -> dict:
        self.run_once_calls += 1
        if self.errors:
            raise self.errors.pop(0)
        if self.results:
            return self.results.pop(0)
        return {"decision": {"action": "hold", "reason": "waiting_for_signal"}}


def _settings(**overrides) -> Settings:
    values = {
        "auto_trade_enabled": True,
        "circuit_breaker_enabled": True,
        "max_same_risk_blocks_before_pause": 2,
        "max_runtime_errors_before_pause": 2,
        "circuit_breaker_window_seconds": 900,
        "scan_interval_seconds": 1,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _risk_result(reason: str) -> dict:
    return {"decision": {"action": "hold", "reason": reason}}


def _persisted_scheduler_session(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'trading.db'}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(scheduler_module, "SessionLocal", Session)
    return engine, Session


@pytest.mark.anyio
async def test_repeated_same_risk_block_triggers_pause():
    trader = FakeTrader(
        [
            _risk_result("trade_cooldown_active"),
            _risk_result("trade_cooldown_active"),
        ]
    )
    scheduler = TradingScheduler(trader, _settings())

    assert await scheduler.run_pending_once() is True
    assert scheduler.paused is False

    assert await scheduler.run_pending_once() is True

    assert scheduler.paused is True
    assert scheduler.pause_reason == "repeated_risk_block:trade_cooldown_active"
    assert scheduler.paused_at is not None


@pytest.mark.anyio
async def test_pause_prevents_run_once():
    trader = FakeTrader([_risk_result("trade_cooldown_active")])
    scheduler = TradingScheduler(trader, _settings())

    changed = await scheduler.pause("manual_pause")
    ran = await scheduler.run_pending_once()

    assert changed is True
    assert ran is False
    assert trader.run_once_calls == 0


@pytest.mark.anyio
async def test_scheduler_does_not_start_when_auto_trade_is_disabled():
    scheduler = TradingScheduler(FakeTrader(), _settings(auto_trade_enabled=False))

    assert scheduler.start() is False
    assert scheduler.running is False


@pytest.mark.anyio
async def test_scheduler_starts_only_one_loop_per_process():
    scheduler = TradingScheduler(FakeTrader(), _settings())

    assert scheduler.start() is True
    assert scheduler.start() is False
    assert scheduler.running is True
    assert await scheduler.stop() is True
    assert scheduler.running is False


@pytest.mark.anyio
async def test_resume_clears_pause():
    scheduler = TradingScheduler(FakeTrader(), _settings())

    await scheduler.pause("manual_pause")
    resumed = scheduler.resume()

    assert resumed is True
    assert scheduler.paused is False
    assert scheduler.pause_reason is None
    assert scheduler.paused_at is None


@pytest.mark.anyio
async def test_different_risk_reasons_are_counted_separately():
    trader = FakeTrader(
        [
            _risk_result("trade_cooldown_active"),
            _risk_result("api_budget_exhausted"),
            _risk_result("trade_cooldown_active"),
        ]
    )
    scheduler = TradingScheduler(trader, _settings())

    await scheduler.run_pending_once()
    await scheduler.run_pending_once()

    assert scheduler.paused is False

    await scheduler.run_pending_once()

    assert scheduler.paused is True
    assert scheduler.pause_reason == "repeated_risk_block:trade_cooldown_active"


@pytest.mark.anyio
async def test_ioc_cancel_cooldown_does_not_pause_scheduler():
    trader = FakeTrader(
        [
            _risk_result("ioc_cancel_cooldown_active"),
            _risk_result("ioc_cancel_cooldown_active"),
            _risk_result("ioc_cancel_cooldown_active"),
        ]
    )
    scheduler = TradingScheduler(trader, _settings(max_same_risk_blocks_before_pause=2))

    assert await scheduler.run_pending_once() is True
    assert await scheduler.run_pending_once() is True
    assert await scheduler.run_pending_once() is True

    assert scheduler.paused is False
    assert scheduler.pause_reason is None


@pytest.mark.anyio
async def test_discord_pause_alert_sent_once():
    scheduler = TradingScheduler(FakeTrader(), _settings())

    await scheduler.record_risk_block("trade_cooldown_active")
    await scheduler.record_risk_block("trade_cooldown_active")
    await scheduler.record_risk_block("trade_cooldown_active")

    assert scheduler.paused is True
    assert scheduler.trader.notifier.pause_alerts == ["repeated_risk_block:trade_cooldown_active"]


@pytest.mark.anyio
async def test_repeated_runtime_errors_trigger_pause():
    trader = FakeTrader(errors=[RuntimeError("boom"), RuntimeError("boom again")])
    scheduler = TradingScheduler(trader, _settings())

    assert await scheduler.run_pending_once() is False
    assert scheduler.paused is False

    assert await scheduler.run_pending_once() is False

    assert scheduler.paused is True
    assert scheduler.pause_reason == "repeated_runtime_errors"
    assert trader.notifier.pause_alerts == ["repeated_runtime_errors"]


@pytest.mark.anyio
async def test_scalping_hourly_loss_limit_pauses_with_clear_reason(tmp_path, monkeypatch):
    engine, _ = _persisted_scheduler_session(tmp_path, monkeypatch)
    trader = FakeTrader(
        [
            {
                "decision": {"action": "hold", "reason": SCALPING_HOURLY_LOSS_REASON},
                "scalping_kill_switch_reason": SCALPING_HOURLY_LOSS_REASON,
            }
        ]
    )
    try:
        scheduler = TradingScheduler(trader, _settings(scalping_mode_enabled=True))

        assert await scheduler.run_pending_once() is True

        assert scheduler.paused is True
        assert scheduler.pause_reason == "scalping_kill_switch:hourly_loss_limit"
    finally:
        engine.dispose()


@pytest.mark.anyio
async def test_scalping_runtime_errors_pause_with_clear_reason(tmp_path, monkeypatch):
    engine, _ = _persisted_scheduler_session(tmp_path, monkeypatch)
    trader = FakeTrader(errors=[RuntimeError("boom"), RuntimeError("boom again")])
    try:
        scheduler = TradingScheduler(trader, _settings(scalping_mode_enabled=True))

        assert await scheduler.run_pending_once() is False
        assert await scheduler.run_pending_once() is False

        assert scheduler.paused is True
        assert scheduler.pause_reason == SCALPING_RUNTIME_ERRORS_REASON
    finally:
        engine.dispose()


@pytest.mark.anyio
async def test_scheduler_runtime_state_persists_success_stale_data_and_error(tmp_path, monkeypatch):
    engine, Session = _persisted_scheduler_session(tmp_path, monkeypatch)
    trader = FakeTrader(
        results=[_risk_result("stale_market_data")],
        errors=[RuntimeError("private details are not persisted")],
    )
    settings = _settings(max_runtime_errors_before_pause=1)
    try:
        scheduler = TradingScheduler(trader, settings)

        assert await scheduler.run_pending_once() is False
        assert scheduler.pause_reason == "repeated_runtime_errors"
        scheduler.resume()
        assert await scheduler.run_pending_once() is True

        with Session() as db:
            stored = Repository(db).scheduler_state()
            assert stored is not None
            assert stored.paused is False
            assert stored.last_successful_run_at is not None
            assert stored.last_runtime_error_at is not None
            assert stored.last_runtime_error == "RuntimeError"
            assert "private details" not in stored.last_runtime_error
            assert stored.last_stale_data_at is not None
            assert stored.last_stale_data_reason == "stale_market_data"
    finally:
        engine.dispose()


@pytest.mark.anyio
async def test_persisted_manual_pause_stays_paused_until_explicit_resume(tmp_path, monkeypatch):
    engine, Session = _persisted_scheduler_session(tmp_path, monkeypatch)
    settings = _settings()
    try:
        scheduler = TradingScheduler(FakeTrader(), settings)
        assert await scheduler.pause("manual_pause") is True
        assert scheduler.paused_at is not None

        restarted = TradingScheduler(FakeTrader(), settings)
        assert restarted.restore_pause_state(now=scheduler.paused_at + timedelta(days=30)) is True
        assert restarted.paused is True
        assert restarted.pause_reason == "manual_pause"

        with Session() as db:
            stored = Repository(db).scheduler_state()
            assert stored is not None
            assert stored.paused is True
            assert stored.pause_reason == "manual_pause"

        assert restarted.resume() is True
        with Session() as db:
            stored = Repository(db).scheduler_state()
            assert stored is not None
            assert stored.paused is False
            assert stored.pause_reason is None
            assert stored.paused_at is None
    finally:
        engine.dispose()


@pytest.mark.anyio
async def test_ioc_cancel_streak_pause_is_restored_after_restart(tmp_path, monkeypatch):
    engine, Session = _persisted_scheduler_session(tmp_path, monkeypatch)
    try:
        trader = FakeTrader(
            [
                {
                    "decision": {"action": "hold", "reason": SCALPING_IOC_CANCEL_STREAK_REASON},
                    "scalping_kill_switch_reason": SCALPING_IOC_CANCEL_STREAK_REASON,
                }
            ]
        )
        settings = _settings(
            scalping_mode_enabled=True,
            ioc_cancel_escalation_cooldown_seconds=600,
        )
        scheduler = TradingScheduler(trader, settings)

        assert await scheduler.run_pending_once() is True
        assert scheduler.paused is True
        assert scheduler.pause_reason == "scalping_kill_switch:ioc_cancel_streak"

        with Session() as db:
            stored = Repository(db).latest_scalping_kill_switch_event()
            assert stored is not None
            assert stored.event_type == "scalping_kill_switch_pause"
            assert stored.reason == SCALPING_IOC_CANCEL_STREAK_REASON

        restarted = TradingScheduler(FakeTrader(), settings)
        assert restarted.restore_pause_state(now=scheduler.paused_at + timedelta(seconds=1)) is True
        assert restarted.paused is True
        assert restarted.pause_reason == SCALPING_IOC_CANCEL_STREAK_REASON
        assert restarted.paused_at == scheduler.paused_at
    finally:
        engine.dispose()


def test_expired_scalping_pause_is_not_restored(tmp_path, monkeypatch):
    engine, Session = _persisted_scheduler_session(tmp_path, monkeypatch)
    paused_at = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)
    settings = _settings(
        scalping_mode_enabled=True,
        ioc_cancel_escalation_cooldown_seconds=10,
    )
    try:
        with Session() as db:
            Repository(db).add_risk_event(
                event_type="scalping_kill_switch_pause",
                reason=SCALPING_IOC_CANCEL_STREAK_REASON,
                created_at=paused_at,
            )

        restarted = TradingScheduler(FakeTrader(), settings)
        restored = restarted.restore_pause_state(now=paused_at + timedelta(seconds=11))

        assert restored is False
        assert restarted.paused is False
        with Session() as db:
            latest = Repository(db).latest_scalping_kill_switch_event()
            assert latest is not None
            assert latest.event_type == "scalping_kill_switch_resume"
    finally:
        engine.dispose()


@pytest.mark.anyio
async def test_application_startup_does_not_start_scheduler_when_auto_trade_is_disabled(monkeypatch):
    from app import main

    class StartupScheduler:
        paused = False
        pause_reason = None
        starts = 0

        def restore_pause_state(self):
            return False

        def start(self):
            self.starts += 1
            return True

    class Logger:
        def event(self, *args, **kwargs):
            return None

    scheduler = StartupScheduler()
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "settings", _settings(auto_trade_enabled=False))
    monkeypatch.setattr(main, "scheduler", scheduler)
    monkeypatch.setattr(main, "get_logger", lambda: Logger())

    await main.startup()

    assert scheduler.starts == 0


@pytest.mark.anyio
async def test_application_startup_starts_scheduler_once_when_auto_trade_is_enabled(monkeypatch):
    from app import main

    class StartupScheduler:
        paused = False
        pause_reason = None
        starts = 0

        def restore_pause_state(self):
            return False

        def start(self):
            self.starts += 1
            return True

    class Logger:
        def event(self, *args, **kwargs):
            return None

    scheduler = StartupScheduler()
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "settings", _settings(auto_trade_enabled=True))
    monkeypatch.setattr(main, "scheduler", scheduler)
    monkeypatch.setattr(main, "get_logger", lambda: Logger())

    await main.startup()

    assert scheduler.starts == 1
