from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from app.config import Settings
from scripts.research_higher_timeframe import (
    VOLATILITY_BREAKOUT_STRATEGY,
    ResearchConfig,
    attach_false_breakout_signal_features,
    evaluate_false_breakout_filter_candidate,
    false_breakout_filter_metrics_pass,
    false_breakout_summary_from_outputs,
)


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "trading_enabled": False,
        "auto_trade_enabled": False,
        "allow_fallback_trading": False,
    }
    values.update(overrides)
    return Settings(**values)


def _config(**overrides) -> ResearchConfig:
    values = {
        "parameter_set_id": "v9m_00021",
        "track_id": "M9_v8m_00086_drawdown_reduction",
        "strategy_name": VOLATILITY_BREAKOUT_STRATEGY,
        "timeframe": "1H",
        "exit_mode": "fixed_tp_sl_timeout",
        "take_profit_pct": 0.045,
        "stop_loss_pct": 0.02,
        "max_hold_bars": 48,
        "breakout_lookback": 20,
        "consolidation_lookback": 12,
        "min_body_vs_avg": 1.2,
        "min_recent_return_pct": 0.003,
        "min_trend_strength": 0.0,
        "max_atr_expansion": 3.0,
        "min_volume_zscore": 0.25,
    }
    values.update(overrides)
    return ResearchConfig(**values)


def _signal_bars(count: int = 80) -> pd.DataFrame:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    rows = []
    for index in range(count):
        close = 100.0 + index * 0.5
        rows.append(
            {
                "timestamp": start + timedelta(hours=index),
                "open": close - 0.2,
                "high": close + 0.6,
                "low": close - 0.5,
                "close": close,
                "volume": 10.0 + index,
            }
        )
    return pd.DataFrame(rows)


def _maker_trades() -> pd.DataFrame:
    bars = _signal_bars()
    timestamps = [bars["timestamp"].iloc[40], bars["timestamp"].iloc[41]]
    return pd.DataFrame(
        {
            "signal_timestamp": [timestamp.isoformat() for timestamp in timestamps],
            "exit_scenario": ["conservative_stop_accounting", "conservative_stop_accounting"],
            "exit_reason": ["stop_loss_accounting", "maker_take_profit"],
            "minutes_to_entry_fill": [5.0, 12.0],
            "max_favorable_excursion_pct": [0.001, 0.05],
            "max_adverse_excursion_pct": [-0.03, -0.005],
            "return_pct": [-0.02, 0.045],
        }
    )


def _feature_rows() -> pd.DataFrame:
    rows = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(5):
        rows.append(
            {
                "signal_timestamp": start + timedelta(hours=index),
                "close_position_in_candle": 0.35,
                "exit_reason": "stop_loss_accounting",
                "return_pct": -0.02,
            }
        )
    for index in range(2):
        rows.append(
            {
                "signal_timestamp": start + timedelta(hours=10 + index),
                "close_position_in_candle": 0.8,
                "exit_reason": "maker_take_profit",
                "return_pct": 0.045,
            }
        )
    for index in range(5):
        rows.append(
            {
                "signal_timestamp": start + timedelta(hours=20 + index),
                "close_position_in_candle": 0.75,
                "exit_reason": "timeout_exit",
                "return_pct": 0.01,
            }
        )
    return pd.DataFrame(rows)


def test_attaches_signal_time_features_to_conservative_trades():
    features = attach_false_breakout_signal_features(_maker_trades(), _signal_bars(), _config())

    assert len(features) == 2
    assert set(features["exit_reason"]) == {"stop_loss_accounting", "maker_take_profit"}
    assert features["signal_open"].notna().all()
    assert features["signal_high"].notna().all()
    assert features["close_position_in_candle"].notna().all()
    assert "max_favorable_excursion_pct" in features.columns


def test_filter_evaluation_removes_stops_and_preserves_tp():
    result = evaluate_false_breakout_filter_candidate(
        _feature_rows(),
        {
            "filter_name": "close_position_in_candle >= 0.6",
            "feature": "close_position_in_candle",
            "operator": ">=",
            "threshold": 0.6,
        },
    )

    assert result["removed_stop_count"] == 5
    assert result["removed_tp_count"] == 0
    assert result["kept_tp_count"] == 2


@pytest.mark.parametrize("feature", ["max_favorable_excursion_pct", "exit_reason"])
def test_filter_cannot_use_future_or_outcome_features(feature):
    with pytest.raises(ValueError, match="signal-time-only"):
        evaluate_false_breakout_filter_candidate(
            _feature_rows(),
            {
                "filter_name": f"{feature} leak",
                "feature": feature,
                "operator": ">=",
                "threshold": 0.1,
            },
        )


def test_false_breakout_filter_found_requires_kept_trade_count_at_least_10():
    assert (
        false_breakout_filter_metrics_pass(
            {
                "kept_trade_count": 9,
                "removed_stop_count": 3,
                "removed_tp_count": 0,
                "kept_return_pct": 0.01,
                "kept_profit_factor": 1.2,
                "kept_max_drawdown_pct": 0.05,
                "uses_signal_time_features_only": True,
            }
        )
        is False
    )


def test_false_breakout_filter_found_requires_removed_tp_count_zero():
    assert (
        false_breakout_filter_metrics_pass(
            {
                "kept_trade_count": 10,
                "removed_stop_count": 3,
                "removed_tp_count": 1,
                "kept_return_pct": 0.01,
                "kept_profit_factor": 1.2,
                "kept_max_drawdown_pct": 0.05,
                "uses_signal_time_features_only": True,
            }
        )
        is False
    )


def test_false_breakout_summary_keeps_orders_placed_zero(tmp_path):
    features = _feature_rows()
    best_filter = {
        "filter_name": "close_position_in_candle >= 0.6",
        "feature": "close_position_in_candle",
        "operator": ">=",
        "threshold": 0.6,
        "uses_signal_time_features_only": True,
        "kept_trade_count": 12,
        "removed_stop_count": 3,
        "removed_tp_count": 0,
        "kept_return_pct": 0.02,
        "kept_profit_factor": 1.2,
        "kept_max_drawdown_pct": 0.05,
        "filter_passed_research_gate": True,
    }

    summary = false_breakout_summary_from_outputs(
        features.assign(signal_open=100.0),
        [best_filter],
        _settings(),
        config=_config(),
        input_trade_count=len(features),
        orders_frame=pd.DataFrame({"orders_placed": [0, 0]}),
        signal_source_used="collected_market_data_derived_from_15min",
        signal_row_count=80,
        synthetic_data_used=False,
        output_paths={
            "summary": tmp_path / "summary.json",
            "features": tmp_path / "features.csv",
            "filter_candidates": tmp_path / "filters.csv",
            "feature_comparison": tmp_path / "comparison.csv",
        },
        input_paths={
            "trades": Path("logs/maker_fill_simulation_v9m_00021_trades.csv"),
            "orders": Path("logs/maker_fill_simulation_v9m_00021_orders.csv"),
        },
    )

    assert summary["orders_placed"] == 0
    assert summary["false_breakout_filter_found"] is True
