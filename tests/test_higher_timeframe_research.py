from datetime import UTC, datetime

import pandas as pd
import pytest

from app.config import Settings
from app.risk.risk_manager import PositionState
from app.strategy.strategies import MarketContext, MarketRegime, TrendPullbackStrategy
from scripts.research_higher_timeframe import (
    build_research_summary,
    generate_research_configs,
    paper_forward_readiness_gate,
    research_settings,
)


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "trading_enabled": False,
        "auto_trade_enabled": False,
        "allow_fallback_trading": False,
        "max_backtest_drawdown_pct": 0.01,
    }
    values.update(overrides)
    return Settings(**values)


def _passing_metrics(**overrides):
    metrics = {
        "net_return_pct": 0.02,
        "profit_factor_net": 1.20,
        "number_of_trades": 25,
        "max_drawdown_pct": 0.005,
        "trade_details": [
            {"net_return_pct": 0.004},
            {"net_return_pct": 0.003},
            {"net_return_pct": 0.003},
            {"net_return_pct": -0.001},
        ],
    }
    metrics.update(overrides)
    return metrics


def test_higher_timeframe_research_settings_do_not_enable_trading():
    settings = research_settings(
        _settings(
            trading_enabled=True,
            auto_trade_enabled=True,
            allow_fallback_trading=True,
        )
    )

    assert settings.symbol == "BTC/USD"
    assert settings.paper_trading_only is True
    assert settings.trading_enabled is False
    assert settings.auto_trade_enabled is False
    assert settings.allow_fallback_trading is False


def test_research_config_space_matches_requested_values():
    configs = generate_research_configs()

    assert {config.timeframe for config in configs} == {"5Min", "15Min"}
    assert {config.take_profit_pct for config in configs} == {0.008, 0.01, 0.015, 0.02}
    assert {config.stop_loss_pct for config in configs} == {0.003, 0.005, 0.008}
    assert {config.max_hold_bars for config in configs} == {6, 12, 24, 48}


def test_paper_forward_readiness_blocks_fallback_and_invalid_model():
    settings = _settings()

    fallback = paper_forward_readiness_gate(
        _passing_metrics(),
        settings,
        fallback_prediction_used=True,
        active_model_valid=True,
    )
    invalid_model = paper_forward_readiness_gate(
        _passing_metrics(),
        settings,
        fallback_prediction_used=False,
        active_model_valid=False,
    )

    assert fallback["paper_forward_eligible"] is False
    assert "fallback_prediction_not_allowed" in fallback["rejection_reasons"]
    assert invalid_model["economically_viable"] is True
    assert invalid_model["paper_forward_eligible"] is False
    assert "active_model_invalid" in invalid_model["rejection_reasons"]


def test_paper_forward_readiness_requires_economic_thresholds():
    result = paper_forward_readiness_gate(
        _passing_metrics(net_return_pct=-0.01, profit_factor_net=0.8, number_of_trades=5),
        _settings(),
        fallback_prediction_used=False,
        active_model_valid=True,
    )

    assert result["economically_viable"] is False
    assert result["paper_forward_eligible"] is False
    assert "net_return_not_positive" in result["rejection_reasons"]
    assert "profit_factor_net_below_1_05" in result["rejection_reasons"]
    assert "number_of_trades_below_20" in result["rejection_reasons"]


def test_research_summary_never_auto_applies_or_enables_trading(tmp_path):
    settings = research_settings(_settings())
    summary = build_research_summary(
        [],
        settings,
        data_sources={"5Min": "test", "15Min": "test"},
        csv_path=tmp_path / "research.csv",
        summary_path=tmp_path / "research.json",
        active_model_status={"active_model_valid": False},
    )

    assert summary["auto_apply_best_config"] is False
    assert summary["trading_enabled"] is False
    assert summary["auto_trade_enabled"] is False
    assert summary["fallback_trading_allowed"] is False
    assert summary["paper_forward_eligible_config_count"] == 0


def test_trend_pullback_strategy_is_long_only_when_position_exists():
    strategy = TrendPullbackStrategy(_settings(max_spread_bps=8))
    row = pd.Series(
        {
            "timestamp": datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            "close": 100.0,
            "orderbook_spread": 0.0002,
            "trend_strength_20": 1.2,
            "rsi_14": 48.0,
            "macd_hist": 0.001,
            "atr_14": 0.01,
            "volume_zscore_20": 0.5,
            "ema_fast_distance": 0.0,
            "ema_slow_distance": 0.002,
            "log_return_3": -0.002,
        }
    )
    signal = strategy.generate_signal(
        feature_row=row,
        prediction=None,
        position=PositionState(qty=0.01),
        quote=None,
        market_context=MarketContext(regime=MarketRegime("trending", 0.8, "test")),
    )

    assert signal.action == "hold"
    assert signal.reason == "already_holding_btc"


def test_non_btc_symbol_still_rejected():
    with pytest.raises(ValueError):
        Settings(_env_file=None, symbol="ETH/USD")
