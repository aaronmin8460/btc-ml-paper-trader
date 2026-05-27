from datetime import UTC, datetime, timedelta

from app.config import Settings
from app.risk import risk_manager
from app.risk.risk_manager import PositionState, RiskManager, TradeFrequencyState


class FakeLogger:
    def __init__(self) -> None:
        self.events = []

    def event(self, event_type: str, **payload) -> None:
        self.events.append((event_type, payload))


def test_buy_cannot_happen_if_already_holding_btc():
    risk = RiskManager(Settings(_env_file=None))
    approved, reason = risk.approve_buy(notional=25, position=PositionState(qty=0.01), latest_price=65000)
    assert approved is False
    assert reason == "already_holding_btc"


def test_risk_manager_blocks_oversized_orders():
    risk = RiskManager(Settings(_env_file=None))
    approved, reason = risk.approve_buy(notional=10_000, position=PositionState(), latest_price=65000)
    assert approved is False
    assert "exceeds" in reason


def test_buy_blocked_after_hourly_trade_limit():
    settings = Settings(_env_file=None, max_trades_per_hour=2)
    risk = RiskManager(settings)

    approved, reason = risk.approve_buy(
        notional=25,
        position=PositionState(),
        latest_price=65000,
        trade_frequency=TradeFrequencyState(trades_last_hour=2),
    )

    assert approved is False
    assert reason == "max_trades_per_hour_reached"


def test_kill_switch_block_logs_risk_block(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(risk_manager, "get_logger", lambda: logger)
    settings = Settings(_env_file=None, max_trades_per_hour=1)
    risk = RiskManager(settings)

    risk.approve_buy(
        notional=25,
        position=PositionState(),
        latest_price=65000,
        trade_frequency=TradeFrequencyState(trades_last_hour=1),
    )

    assert logger.events == [
        ("risk_block", {"symbol": "BTC/USD", "reason": "max_trades_per_hour_reached"})
    ]


def test_buy_blocked_after_daily_trade_limit():
    settings = Settings(_env_file=None, max_daily_trades=3)
    risk = RiskManager(settings)

    approved, reason = risk.approve_buy(
        notional=25,
        position=PositionState(),
        latest_price=65000,
        trade_frequency=TradeFrequencyState(trades_today=3),
    )

    assert approved is False
    assert reason == "max_daily_trades_reached"


def test_buy_blocked_during_trade_cooldown():
    now = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)
    settings = Settings(_env_file=None, min_seconds_between_trades=30)
    risk = RiskManager(settings)

    approved, reason = risk.approve_buy(
        notional=25,
        position=PositionState(),
        latest_price=65000,
        trade_frequency=TradeFrequencyState(last_trade_at=now - timedelta(seconds=10)),
        now=now,
    )

    assert approved is False
    assert reason == "trade_cooldown_active"


def test_buy_blocked_after_consecutive_loss_limit():
    settings = Settings(_env_file=None, max_consecutive_losses=3)
    risk = RiskManager(settings)

    approved, reason = risk.approve_buy(
        notional=25,
        position=PositionState(),
        latest_price=65000,
        trade_frequency=TradeFrequencyState(consecutive_losses=3),
    )

    assert approved is False
    assert reason == "max_consecutive_losses_reached"


def test_trade_frequency_limits_do_not_block_position_closing_sell():
    settings = Settings(_env_file=None, max_trades_per_hour=0, max_daily_trades=0, max_consecutive_losses=0)
    risk = RiskManager(settings)

    approved, reason = risk.approve_sell(PositionState(qty=0.01))

    assert approved is True
    assert reason == "approved"


def test_stop_loss_and_take_profit_force_sell_behavior_remains_intact():
    risk = RiskManager(Settings(_env_file=None, stop_loss_pct=0.01, take_profit_pct=0.02))
    position = PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000)
    now = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)

    should_sell, reason = risk.should_force_sell(position=position, latest_price=64000, now=now)
    assert should_sell is True
    assert reason == "stop_loss"

    should_sell, reason = risk.should_force_sell(position=position, latest_price=66500, now=now)
    assert should_sell is True
    assert reason == "take_profit"


def test_scalping_force_sell_uses_scalping_thresholds():
    risk = RiskManager(
        Settings(
            _env_file=None,
            scalping_mode_enabled=True,
            scalping_stop_loss_pct=0.002,
            scalping_take_profit_pct=0.003,
        )
    )
    position = PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000)
    now = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)

    should_sell, reason = risk.should_force_sell(position=position, latest_price=65196, now=now)
    assert should_sell is True
    assert reason == "scalping_take_profit"
