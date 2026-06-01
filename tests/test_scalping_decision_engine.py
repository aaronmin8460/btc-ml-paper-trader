from datetime import UTC, datetime, timedelta

import pandas as pd

from app.config import Settings
from app.risk.risk_manager import PositionState
from app.services.trader import Trader
from app.strategy.decision_engine import DecisionEngine
from app.strategy.scalping_decision_engine import ScalpingDecisionEngine


NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "scalping_mode_enabled": True,
        "trading_enabled": True,
        "max_spread_bps": 5,
        "min_quote_imbalance": -0.2,
        "scalping_max_data_age_seconds": 120,
        "scalping_buy_probability_floor": 0.6,
        "scalping_confidence_gap_required": 0.1,
        "scalping_sell_probability_floor": 0.6,
        "scalping_exit_confidence_gap_required": 0.1,
        "scalping_min_momentum_pct": 0.0001,
        "scalping_stop_loss_pct": 0.002,
        "emergency_stop_loss_pct": 0.01,
        "scalping_take_profit_pct": 0.002,
        "scalping_trailing_stop_pct": 0.001,
        "trailing_stop_arm_profit_pct": 0.001,
        "scalping_max_position_seconds": 30,
        "scalping_min_hold_seconds": 0,
        "min_seconds_between_trades": 0,
    }
    values.update(overrides)
    return Settings(**values)


def _feature_row(**overrides) -> pd.Series:
    values = {
        "timestamp": NOW,
        "close": 100.0,
        "scalping_spread_bps": 2.0,
        "scalping_quote_imbalance": 0.1,
        "scalping_momentum_3": 0.001,
        "scalping_high_breakout_5": -0.001,
    }
    values.update(overrides)
    return pd.Series(values)


def _prediction(**overrides) -> dict:
    values = {
        "symbol": "BTC/USD",
        "buy_probability": 0.8,
        "sell_probability": 0.1,
        "model_available": True,
        "prediction_source": "model",
    }
    values.update(overrides)
    return values


def test_scalping_buy_is_approved_when_all_conditions_pass():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(),
        feature_row=_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        now=NOW,
    )

    assert decision.action == "buy"
    assert decision.reason == "scalping_entry_approved"
    assert decision.notional == 25


def test_scalping_buy_is_blocked_by_wide_spread():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(),
        feature_row=_feature_row(scalping_spread_bps=6.0),
        position=PositionState(),
        trading_enabled=True,
        now=NOW,
    )

    assert decision.action == "hold"
    assert decision.reason == "spread_too_wide"


def test_scalping_buy_is_blocked_by_stale_market_data():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(),
        feature_row=_feature_row(timestamp=NOW - timedelta(seconds=121)),
        position=PositionState(),
        trading_enabled=True,
        now=NOW,
    )

    assert decision.action == "hold"
    assert decision.reason == "stale_market_data"


def test_scalping_buy_is_blocked_when_model_is_unavailable():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(model_available=False, prediction_source="fallback"),
        feature_row=_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        now=NOW,
    )

    assert decision.action == "hold"
    assert decision.reason == "model_unavailable"


def test_scalping_sell_is_triggered_by_stop_loss():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(),
        feature_row=_feature_row(close=99.7),
        position=PositionState(qty=0.1, avg_entry_price=100.0, highest_price=100.0),
        now=NOW,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_stop_loss"
    assert decision.qty == 0.1


def test_scalping_sell_is_triggered_by_take_profit():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(),
        feature_row=_feature_row(close=100.3),
        position=PositionState(qty=0.1, avg_entry_price=100.0, highest_price=100.3),
        now=NOW,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_take_profit"


def test_scalping_sell_is_triggered_by_max_holding_time():
    decision = ScalpingDecisionEngine(_settings(scalping_max_position_seconds=10)).decide(
        prediction=_prediction(),
        feature_row=_feature_row(),
        position=PositionState(
            qty=0.1,
            avg_entry_price=100.0,
            highest_price=100.0,
            opened_at=NOW - timedelta(seconds=11),
        ),
        now=NOW,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_max_position_seconds"


def test_scalping_sell_is_triggered_by_trailing_stop():
    decision = ScalpingDecisionEngine(_settings(scalping_take_profit_pct=0.01)).decide(
        prediction=_prediction(),
        feature_row=_feature_row(close=100.3),
        position=PositionState(qty=0.1, avg_entry_price=100.0, highest_price=100.5),
        now=NOW,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_trailing_stop"


def test_scalping_stale_data_reduces_open_position_conservatively():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(),
        feature_row=_feature_row(timestamp=NOW - timedelta(seconds=121)),
        position=PositionState(qty=0.1, avg_entry_price=100.0, highest_price=100.0),
        now=NOW,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_stale_data_exit"


def test_scalping_sell_model_probability_can_close_position():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(buy_probability=0.1, sell_probability=0.8),
        feature_row=_feature_row(),
        position=PositionState(qty=0.1, avg_entry_price=100.0, highest_price=100.0),
        now=NOW,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_model_sell_signal"


def test_scalping_unfavorable_quote_can_close_position():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(),
        feature_row=_feature_row(scalping_quote_imbalance=-0.2),
        position=PositionState(qty=0.1, avg_entry_price=100.0, highest_price=100.0),
        now=NOW,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_weak_quote_exit"


def test_scalping_minimum_hold_blocks_soft_exit():
    decision = ScalpingDecisionEngine(_settings(scalping_min_hold_seconds=10)).decide(
        prediction=_prediction(),
        feature_row=_feature_row(scalping_quote_imbalance=-0.2),
        position=PositionState(
            qty=0.1,
            avg_entry_price=100.0,
            highest_price=100.0,
            opened_at=NOW - timedelta(seconds=1),
        ),
        now=NOW,
    )

    assert decision.action == "hold"
    assert decision.reason == "scalping_min_hold_active"


def test_scalping_profit_guard_does_not_block_emergency_stop_loss():
    decision = ScalpingDecisionEngine(
        _settings(
            scalping_profit_guard_enabled=True,
            profit_only_exit_enabled=True,
            emergency_stop_loss_pct=0.005,
        )
    ).decide(
        prediction=_prediction(),
        feature_row=_feature_row(close=99.0),
        position=PositionState(qty=0.1, avg_entry_price=100.0, highest_price=100.0),
        now=NOW,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_emergency_stop_loss"


def test_trader_selects_scalping_engine_only_when_scalping_mode_is_enabled():
    assert isinstance(Trader(_settings()).decision_engine, ScalpingDecisionEngine)
    assert isinstance(Trader(Settings(_env_file=None, scalping_mode_enabled=False)).decision_engine, DecisionEngine)
