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
    from datetime import UTC, datetime, timedelta

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


def test_scalping_sell_triggers_on_stop_loss():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, scalping_stop_loss_pct=0.002)
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
    assert decision.reason == "scalping_stop_loss"


def test_scalping_sell_triggers_on_max_position_seconds():
    from datetime import UTC, datetime, timedelta

    settings = Settings(_env_file=None, scalping_mode_enabled=True, scalping_max_position_seconds=180)
    engine = DecisionEngine(settings)
    feature_row = _scalping_feature_row()
    feature_row["close"] = 65010

    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.4, "sell_probability": 0.6},
        feature_row=feature_row,
        position=PositionState(
            qty=0.01,
            avg_entry_price=65000,
            highest_price=65010,
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
        quote={"bid_price": 65100, "ask_price": 65110},
    )

    assert decision.action == "sell"
    assert decision.reason == "scalping_take_profit"


def test_scalping_quote_first_sell_triggers_stop_loss():
    settings = Settings(_env_file=None, scalping_mode_enabled=True, scalping_stop_loss_pct=0.001)
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
    assert decision.reason == "scalping_stop_loss"


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
        position=PositionState(qty=0.01, avg_entry_price=65000, highest_price=65200),
        trading_enabled=True,
        quote={"bid_price": 65130, "ask_price": 65140},
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
        quote={"bid_price": 65100, "ask_price": 65110},
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
        quote={"bid_price": 65100, "ask_price": 65110},
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
