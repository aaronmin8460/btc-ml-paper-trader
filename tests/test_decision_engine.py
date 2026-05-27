import pytest

from app.broker.execution_guard import BTCOnlyViolation
from app.config import Settings
from app.data.feature_engineering import latest_feature_row
from app.data.market_data import MarketDataClient
from app.risk.risk_manager import PositionState
from app.strategy.decision_engine import DecisionEngine


def _feature_row():
    bars = MarketDataClient.synthetic_btc_bars(120)
    return latest_feature_row(bars).iloc[-1]


def _buy_prediction(symbol="BTC/USD"):
    return {"symbol": symbol, "buy_probability": 0.9, "sell_probability": 0.1}


def test_decision_engine_does_not_buy_when_ml_probability_low():
    engine = DecisionEngine(Settings())
    decision = engine.decide(
        prediction={"symbol": "BTC/USD", "buy_probability": 0.2, "sell_probability": 0.8},
        feature_row=_feature_row(),
        position=PositionState(),
        trading_enabled=True,
    )
    assert decision.action == "hold"
    assert decision.reason == "buy_probability_below_threshold"


def test_decision_engine_does_not_buy_when_already_holding():
    engine = DecisionEngine(Settings())
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
