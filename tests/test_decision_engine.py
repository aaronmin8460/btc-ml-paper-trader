from app.config import Settings
from app.data.feature_engineering import latest_feature_row
from app.data.market_data import MarketDataClient
from app.risk.risk_manager import PositionState
from app.strategy.decision_engine import DecisionEngine


def _feature_row():
    bars = MarketDataClient.synthetic_btc_bars(120)
    return latest_feature_row(bars).iloc[-1]


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
