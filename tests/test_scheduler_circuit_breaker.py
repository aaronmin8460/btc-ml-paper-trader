import pytest

from app.config import Settings
from app.services.scheduler import TradingScheduler


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
