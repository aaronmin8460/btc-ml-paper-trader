from datetime import UTC, datetime, timedelta

import pytest

from app.broker.execution_guard import BTCOnlyViolation
from app.config import Settings
from app.data.feature_engineering import latest_feature_row
from app.data.market_data import MarketDataClient
from app.risk.risk_manager import AccountState, PositionState, TradeFrequencyState
from app.strategy.decision_engine import DecisionEngine


def _feature_row():
    bars = MarketDataClient.synthetic_btc_bars(120)
    return latest_feature_row(bars).iloc[-1]


def _buy_prediction(symbol="BTC/USD"):
    return {"symbol": symbol, "buy_probability": 0.9, "sell_probability": 0.1}


def _scalping_feature_row():
    feature_row = _feature_row()
    feature_row["orderbook_spread"] = 0.0001
    feature_row["quote_imbalance"] = 0.2
    feature_row["sma_20_distance"] = -0.002
    feature_row["log_return_3"] = 0
    feature_row["log_return_5"] = -0.0015
    return feature_row


def test_decision_engine_does_not_buy_when_ml_probability_low():
    engine = DecisionEngine(Settings(_env_file=None, scalping_mode_enabled=False))
    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.2, "sell_probability": 0.8},
        feature_row=_feature_row(),
        position=PositionState(),
        trading_enabled=True,
    )
    assert decision.action == "hold"
    assert decision.reason == "buy_probability_below_threshold"


def test_decision_engine_does_not_buy_when_already_holding():
    engine = DecisionEngine(Settings(_env_file=None, scalping_mode_enabled=False))
    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.9, "sell_probability": 0.1},
        feature_row=_feature_row(),
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000),
        trading_enabled=True,
    )
    assert decision.action in {"hold", "sell"}
    assert decision.action != "buy"


def test_scalping_mode_blocks_buy_when_spread_too_wide():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, trading_enabled=True, max_spread_bps=8)
    engine = DecisionEngine(settings)
    feature_row = _feature_row()
    feature_row["orderbook_spread"] = 0.001
    feature_row["quote_imbalance"] = 0

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=feature_row,
        position=PositionState(),
        trading_enabled=True,
    )

    assert decision.action == "hold"
    assert decision.reason == "spread_too_wide"


def test_scalping_mode_blocks_buy_when_quote_imbalance_too_weak():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, trading_enabled=True, min_quote_imbalance=-0.25)
    engine = DecisionEngine(settings)
    feature_row = _feature_row()
    feature_row["orderbook_spread"] = 0.0001
    feature_row["quote_imbalance"] = -0.5

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=feature_row,
        position=PositionState(),
        trading_enabled=True,
    )

    assert decision.action == "hold"
    assert decision.reason == "quote_imbalance_too_weak"


def test_scalping_mode_blocks_buy_when_latest_close_invalid():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, trading_enabled=True)
    engine = DecisionEngine(settings)
    feature_row = _feature_row()
    feature_row["close"] = 0

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=feature_row,
        position=PositionState(),
        trading_enabled=True,
    )

    assert decision.action == "hold"
    assert decision.reason == "invalid_price"


def test_non_scalping_behavior_ignores_scalping_quote_imbalance_filter():
    settings = Settings(_env_file=None, scalping_mode_enabled=False, trading_enabled=True)
    engine = DecisionEngine(settings)
    feature_row = _feature_row()
    feature_row["orderbook_spread"] = 0.0001
    feature_row["quote_imbalance"] = -1

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=feature_row,
        position=PositionState(),
        trading_enabled=True,
    )

    assert decision.reason != "quote_imbalance_too_weak"


def test_decision_engine_still_enforces_btc_only_symbol():
    engine = DecisionEngine(Settings(_env_file=None, scalping_mode_enabled=True, trading_enabled=True))

    with pytest.raises(BTCOnlyViolation):
        engine.decide(
            prediction=_buy_prediction(symbol="ETH/USD"),
            feature_row=_feature_row(),
            position=PositionState(),
            trading_enabled=True,
        )


def test_scalping_mode_uses_buy_probability_floor_instead_of_old_threshold():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        trading_enabled=True,
        min_buy_probability=0.58,
        scalping_buy_probability_floor=0.50,
    )
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.52, "sell_probability": 0.48},
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
    )

    assert decision.action == "buy"
    assert decision.reason == "scalping_dip_entry"


def test_scalping_buy_blocked_by_trade_cooldown():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        trading_enabled=True,
        min_seconds_between_trades=10,
    )
    engine = DecisionEngine(settings)
    now = datetime.now(UTC)

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        trade_frequency=TradeFrequencyState(last_trade_at=now - timedelta(seconds=2)),
    )

    assert decision.action == "hold"
    assert decision.reason == "trade_cooldown_active"


def test_scalping_buy_blocked_by_order_attempt_frequency_limit():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        trading_enabled=True,
        max_order_attempts_per_hour=2,
    )
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        order_attempt_frequency=TradeFrequencyState(trades_last_hour=2),
    )

    assert decision.action == "hold"
    assert decision.reason == "max_order_attempts_per_hour_reached"


def test_scalping_buy_blocked_by_filled_trade_frequency_limit():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        trading_enabled=True,
        max_trades_per_hour=1,
    )
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        trade_frequency=TradeFrequencyState(trades_last_hour=0),
        filled_trade_frequency=TradeFrequencyState(trades_last_hour=1),
    )

    assert decision.action == "hold"
    assert decision.reason == "max_trades_per_hour_reached"


def test_scalping_sell_triggers_on_take_profit():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, scalping_take_profit_pct=0.003)
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65200

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65200),
        trading_enabled=True,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_take_profit"


def test_scalping_sell_triggers_on_emergency_stop_loss():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, emergency_stop_loss_pct=0.002)
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 64800

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000),
        trading_enabled=True,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_emergency_stop_loss"


def test_scalping_stop_loss_can_sell_before_weak_quote_min_hold():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        emergency_stop_loss_pct=0.001,
        min_hold_seconds_before_weak_quote_exit=30,
    )
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65000
    feature_row["orderbook_spread"] = 0.001

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(
            qty=0.01,
            avg_entry_price=65000,
            highest_price=65000,
            opened_at=datetime.now(UTC),
        ),
        trading_enabled=True,
        quote={"bid_price": 64900, "ask_price": 64910},
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_emergency_stop_loss"


def test_scalping_take_profit_can_sell_before_weak_quote_min_hold():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        scalping_take_profit_pct=0.001,
        min_hold_seconds_before_weak_quote_exit=30,
    )
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65000
    feature_row["orderbook_spread"] = 0.001

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(
            qty=0.01,
            avg_entry_price=65000,
            highest_price=65000,
            opened_at=datetime.now(UTC),
        ),
        trading_enabled=True,
        quote={"bid_price": 65200, "ask_price": 65210},
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_take_profit"


def test_scalping_sell_triggers_on_max_position_seconds():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        scalping_max_position_seconds=180,
        scalping_take_profit_pct=0.01,
    )
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65200

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(
            qty=0.01,
            avg_entry_price=65000,
            highest_price=65200,
            opened_at=datetime.now(UTC) - timedelta(seconds=181),
        ),
        trading_enabled=True,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_max_position_seconds"


def test_scalping_quote_first_sell_triggers_take_profit():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, scalping_take_profit_pct=0.0015)
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65000

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000),
        trading_enabled=True,
        quote={"bid_price": 65200, "ask_price": 65210},
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_take_profit"


def test_scalping_quote_first_sell_triggers_emergency_stop_loss():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, emergency_stop_loss_pct=0.001)
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65000

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000),
        trading_enabled=True,
        quote={"bid_price": 64920, "ask_price": 64930},
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_emergency_stop_loss"


def test_scalping_quote_first_sell_triggers_trailing_stop():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        scalping_take_profit_pct=0.01,
        scalping_trailing_stop_pct=0.0008,
    )
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65000

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65400),
        trading_enabled=True,
        quote={"bid_price": 65330, "ask_price": 65340},
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_trailing_stop"


def test_api_hard_budget_blocks_buy_but_not_sell():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, trading_enabled=True)
    engine = DecisionEngine(settings)
    exhausted_budget = {"api_budget_status": "hard_stop", "budget_remaining": 0}

    buy = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        api_budget=exhausted_budget,
    )
    sell = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000),
        trading_enabled=True,
        api_budget=exhausted_budget,
        quote={"bid_price": 65200, "ask_price": 65210},
    )

    assert buy.action == "hold"
    assert buy.reason == "api_budget_exhausted"
    assert sell.action == "sell"


def test_account_drawdown_blocks_buy_but_not_sell():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, trading_enabled=True, max_account_drawdown_pct=0.03)
    engine = DecisionEngine(settings)

    buy = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        account_state=AccountState(available=True, equity=96_000, buying_power=1_000, drawdown_pct=0.04),
    )
    sell = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000),
        trading_enabled=True,
        account_state=AccountState(available=True, equity=96_000, buying_power=1_000, drawdown_pct=0.04),
        quote={"bid_price": 65200, "ask_price": 65210},
    )

    assert buy.action == "hold"
    assert buy.reason == "account_drawdown_reached"
    assert sell.action == "sell"


def test_recent_ioc_cancels_block_scalping_buy():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, trading_enabled=True)
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        recent_ioc_canceled_buys=3,
    )

    assert decision.action == "hold"
    assert decision.reason == "recent_ioc_cancels_too_high"


def test_recent_ioc_cancels_threshold_uses_settings():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        trading_enabled=True,
        max_recent_ioc_cancels=4,
    )
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        recent_ioc_canceled_buys=3,
    )

    assert decision.action == "buy"


def test_scalping_confidence_gap_blocks_weak_buy():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        trading_enabled=True,
        scalping_buy_probability_floor=0.50,
        scalping_confidence_gap_required=0.06,
    )
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.52, "sell_probability": 0.48},
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
    )

    assert decision.action == "hold"
    assert decision.reason == "scalping_confidence_gap_too_small"


def test_ioc_cancel_cooldown_blocks_scalping_buy_after_recent_cancel():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        trading_enabled=True,
        ioc_cancel_cooldown_seconds=120,
    )
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        recent_ioc_canceled_buys=1,
        latest_ioc_canceled_buy_at=datetime.now(UTC) - timedelta(seconds=30),
    )

    assert decision.action == "hold"
    assert decision.reason == "ioc_cancel_cooldown_active"


def test_old_ioc_cancel_does_not_block_scalping_buy_after_cooldown():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        trading_enabled=True,
        ioc_cancel_cooldown_seconds=120,
    )
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        recent_ioc_canceled_buys=1,
        latest_ioc_canceled_buy_at=datetime.now(UTC) - timedelta(seconds=121),
    )

    assert decision.action == "buy"


def test_ioc_cancel_cooldown_clears_after_configured_time():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        trading_enabled=True,
        ioc_cancel_cooldown_seconds=10,
    )
    engine = DecisionEngine(settings)
    canceled_at = datetime.now(UTC) - timedelta(seconds=11)

    decision = engine.decide(
        prediction=_buy_prediction(),
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        recent_ioc_canceled_buys=1,
        latest_ioc_canceled_buy_at=canceled_at,
    )

    assert decision.action == "buy"


def test_fallback_prediction_cannot_open_scalping_trade_by_default():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, trading_enabled=True)
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction={
            "symbol": "BTC/USD",
            "buy_probability": 0.9,
            "sell_probability": 0.1,
            "prediction_source": "fallback",
        },
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
    )

    assert decision.action == "hold"
    assert decision.reason == "model_unavailable"


def test_fallback_prediction_can_trade_when_explicitly_enabled():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        trading_enabled=True,
        allow_fallback_trading=True,
    )
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction={
            "symbol": "BTC/USD",
            "buy_probability": 0.9,
            "sell_probability": 0.1,
            "prediction_source": "fallback",
        },
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
    )

    assert decision.action == "buy"


def test_buy_probability_equal_to_scalping_floor_does_not_buy():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        trading_enabled=True,
        scalping_buy_probability_floor=0.50,
        scalping_confidence_gap_required=0.0,
    )
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction={
            "symbol": "BTC/USD",
            "buy_probability": 0.50,
            "sell_probability": 0.10,
            "prediction_source": "model",
            "model_available": True,
        },
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
    )

    assert decision.action == "hold"
    assert decision.reason == "scalping_buy_probability_below_floor"


def test_strong_model_prediction_can_still_buy():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, trading_enabled=True)
    engine = DecisionEngine(settings)

    decision = engine.decide(
        prediction={
            "symbol": "BTC/USD",
            "buy_probability": 0.90,
            "sell_probability": 0.10,
            "prediction_source": "model",
            "model_available": True,
        },
        feature_row=_scalping_feature_row(),
        position=PositionState(),
        trading_enabled=True,
    )

    assert decision.action == "buy"
    assert decision.reason == "scalping_dip_entry"


def test_fallback_prediction_still_allows_hard_risk_exit():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, emergency_stop_loss_pct=0.001)
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65000

    decision = engine.decide(
        prediction={
            "symbol": "BTC/USD",
            "buy_probability": 0.9,
            "sell_probability": 0.1,
            "prediction_source": "fallback",
        },
        feature_row=feature_row,
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000),
        trading_enabled=True,
        quote={"bid_price": 64900, "ask_price": 64910},
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_emergency_stop_loss"


def test_weak_quote_exit_waits_for_minimum_hold_time():
    now = datetime.now(UTC)
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        min_hold_seconds_before_weak_quote_exit=30,
        max_spread_bps=6,
        scalping_take_profit_pct=0.01,
    )
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65200
    feature_row["orderbook_spread"] = 0.001

    early = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(
            qty=0.01,
            avg_entry_price=65000,
            highest_price=65200,
            opened_at=now - timedelta(seconds=5),
        ),
        trading_enabled=True,
    )
    later = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(
            qty=0.01,
            avg_entry_price=65000,
            highest_price=65200,
            opened_at=now - timedelta(seconds=31),
        ),
        trading_enabled=True,
    )

    assert early.action == "hold"
    assert early.reason == "weak_quote_exit_min_hold_active"
    assert later.action == "sell"
    assert later.reason == "scalping_weak_quote_exit"


def test_weak_quote_sell_is_blocked_at_loss():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        scalping_take_profit_pct=0.01,
        min_hold_seconds_before_weak_quote_exit=0,
        max_spread_bps=6,
    )
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 64850
    feature_row["orderbook_spread"] = 0.001

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000),
        trading_enabled=True,
    )

    assert decision.action == "hold"
    assert decision.reason == "profit_guard_holding_at_loss"


def test_weak_quote_sell_is_allowed_at_profit():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        scalping_take_profit_pct=0.01,
        min_hold_seconds_before_weak_quote_exit=0,
        max_spread_bps=6,
    )
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65200
    feature_row["orderbook_spread"] = 0.001

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65200),
        trading_enabled=True,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_weak_quote_exit"


def test_model_sell_is_blocked_at_loss():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=False,
        min_sell_probability=0.55,
        confidence_gap_required=0.08,
    )
    engine = DecisionEngine(settings)
    feature_row = _feature_row()
    feature_row["close"] = 64850

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.1, "sell_probability": 0.9},
        feature_row=feature_row,
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000),
        trading_enabled=True,
    )

    assert decision.action == "hold"
    assert decision.reason == "profit_guard_holding_at_loss"


def test_emergency_stop_loss_is_allowed_only_when_enabled():
    feature_row = _scalping_feature_row()
    feature_row["close"] = 64500
    position = PositionState(qty=0.01, avg_entry_price=65000, highest_price=65000)

    enabled = DecisionEngine(
        Settings(
            _env_file=None,
            scalping_mode_enabled=True,
            allow_emergency_stop_loss=True,
            emergency_stop_loss_pct=0.006,
        )
    ).decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=position,
        trading_enabled=True,
    )
    disabled = DecisionEngine(
        Settings(
            _env_file=None,
            scalping_mode_enabled=True,
            allow_emergency_stop_loss=False,
            emergency_stop_loss_pct=0.006,
        )
    ).decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=position,
        trading_enabled=True,
    )

    assert enabled.action == "sell"
    assert enabled.reason == "scalping_emergency_stop_loss"
    assert disabled.action == "hold"
    assert disabled.reason == "profit_guard_holding_at_loss"


def test_trailing_stop_is_not_armed_before_profit_threshold():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        scalping_take_profit_pct=0.01,
        scalping_trailing_stop_pct=0.0008,
        trailing_stop_arm_profit_pct=0.002,
    )
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65040

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65100),
        trading_enabled=True,
    )

    assert decision.action == "hold"
    assert decision.reason == "scalping_holding_position"


def test_trailing_stop_works_after_profit_threshold():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        scalping_take_profit_pct=0.01,
        scalping_trailing_stop_pct=0.0008,
        trailing_stop_arm_profit_pct=0.002,
    )
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65330

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65400),
        trading_enabled=True,
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_trailing_stop"


def test_max_holding_sell_is_blocked_at_loss_when_configured():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        scalping_take_profit_pct=0.01,
        scalping_max_position_seconds=90,
        max_holding_sell_requires_profit=True,
    )
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 64850

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(
            qty=0.01,
            avg_entry_price=65000,
            highest_price=65000,
            opened_at=datetime.now(UTC) - timedelta(seconds=91),
        ),
        trading_enabled=True,
    )

    assert decision.action == "hold"
    assert decision.reason == "profit_guard_holding_at_loss"
