from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.broker.execution_guard import BTCOnlyViolation, LongOnlyViolation, assert_btc_only, validate_order_request
from app.config import Settings
from app.data.market_data import MarketDataClient
from app.data.scalping_features import build_scalping_features
from app.risk.risk_manager import PositionState, TradeFrequencyState
from app.strategy.scalping_decision_engine import ScalpingDecisionEngine
from app.strategy.strategies import MarketContext, MarketRegimeFilter, MeanReversionScalpingStrategy


NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "scalping_mode_enabled": True,
        "trading_enabled": True,
        "max_spread_bps": 5,
        "min_quote_imbalance": -0.2,
        "scalping_buy_probability_floor": 0.6,
        "scalping_confidence_gap_required": 0.1,
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
        "scalping_log_return_3": -0.001,
        "scalping_momentum_3": -0.001,
        "scalping_high_breakout_5": -0.001,
        "scalping_low_breakout_5": -0.001,
        "scalping_ema_5_distance": -0.001,
        "scalping_vwap_distance": -0.001,
        "scalping_rsi_3": 38.0,
        "scalping_volatility_10": 0.001,
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
        "active_model_valid": True,
    }
    values.update(overrides)
    return values


def test_mean_reversion_strategy_is_deterministic_for_fixed_inputs():
    settings = _settings()
    row = _feature_row()
    regime = MarketRegimeFilter(settings).detect(row)
    strategy = MeanReversionScalpingStrategy(settings)

    first = strategy.generate_signal(
        feature_row=row,
        prediction=_prediction(),
        position=PositionState(),
        quote=None,
        market_context=MarketContext(regime=regime),
    )
    second = strategy.generate_signal(
        feature_row=row,
        prediction=_prediction(),
        position=PositionState(),
        quote=None,
        market_context=MarketContext(regime=regime),
    )

    assert first == second
    assert first.action == "buy"
    assert first.strategy_name == "mean_reversion_scalping"


def test_strategy_signal_does_not_change_when_future_bars_change():
    settings = _settings()
    bars = MarketDataClient.synthetic_btc_bars(120)
    changed = bars.copy()
    changed.loc[90:, "close"] = changed.loc[90:, "close"] * 1.5
    original_row = build_scalping_features(bars).loc[60]
    changed_row = build_scalping_features(changed).loc[60]
    original_row["scalping_spread_bps"] = changed_row["scalping_spread_bps"] = 2.0
    original_row["scalping_quote_imbalance"] = changed_row["scalping_quote_imbalance"] = 0.1
    regime_filter = MarketRegimeFilter(settings)
    strategy = MeanReversionScalpingStrategy(settings)

    original = strategy.generate_signal(
        feature_row=original_row,
        prediction=_prediction(),
        position=PositionState(),
        quote=None,
        market_context=MarketContext(regime=regime_filter.detect(original_row)),
    )
    after_future_change = strategy.generate_signal(
        feature_row=changed_row,
        prediction=_prediction(),
        position=PositionState(),
        quote=None,
        market_context=MarketContext(regime=regime_filter.detect(changed_row)),
    )

    assert original == after_future_change


def test_market_regime_filter_blocks_too_volatile_and_not_tradeable_regimes():
    settings = _settings()
    regime_filter = MarketRegimeFilter(settings)

    volatile = regime_filter.detect(_feature_row(scalping_volatility_10=0.03))
    not_tradeable = regime_filter.detect(_feature_row(scalping_spread_bps=10.0))

    assert volatile.regime == "too_volatile"
    assert not_tradeable.regime == "not_tradeable"


def test_scalping_engine_blocks_too_volatile_regime_before_buy():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(),
        feature_row=_feature_row(scalping_volatility_10=0.03),
        position=PositionState(),
        trading_enabled=True,
        now=NOW,
    )

    assert decision.action == "hold"
    assert decision.blocked_by == "regime_filter"
    assert decision.reason == "volatility_too_high"


def test_scalping_engine_keeps_fallback_predictions_from_triggering_buy():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(model_available=False, prediction_source="fallback", active_model_valid=False),
        feature_row=_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        now=NOW,
    )

    assert decision.action == "hold"
    assert decision.reason == "model_unavailable"
    assert decision.blocked_by == "ml_filter"


def test_scalping_engine_reports_active_model_invalid_explicitly():
    decision = ScalpingDecisionEngine(_settings()).decide(
        prediction=_prediction(
            model_available=False,
            prediction_source="fallback_invalid_model",
            active_model_valid=False,
            active_model_path="models/rejected.joblib",
            active_model_reason="model_not_profitable_after_costs",
        ),
        feature_row=_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        now=NOW,
    )

    assert decision.action == "hold"
    assert decision.reason == "active_model_invalid"
    assert decision.blocked_by == "active_model_invalid"


def test_scalping_engine_keeps_risk_manager_as_final_buy_authority():
    decision = ScalpingDecisionEngine(_settings(min_seconds_between_trades=60)).decide(
        prediction=_prediction(),
        feature_row=_feature_row(),
        position=PositionState(),
        trading_enabled=True,
        trade_frequency=TradeFrequencyState(last_trade_at=NOW - timedelta(seconds=5)),
        now=NOW,
    )

    assert decision.action == "hold"
    assert decision.reason == "trade_cooldown_active"
    assert decision.blocked_by == "cooldown"


def test_btc_only_and_long_only_execution_guards_still_enforce_scope():
    with pytest.raises(BTCOnlyViolation):
        assert_btc_only("ETH/USD", context="test")
    with pytest.raises(LongOnlyViolation):
        validate_order_request("BTC/USD", "sell", qty=0.01, current_position_qty=0)
