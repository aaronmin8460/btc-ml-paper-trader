from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.config import Settings
from scripts.research_higher_timeframe import (
    VOLATILITY_BREAKOUT_STRATEGY,
    ResearchConfig,
    evaluate_maker_fill_simulation_pass_fail,
    simulate_post_only_maker_fill_scenario,
)


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "trading_enabled": False,
        "auto_trade_enabled": False,
        "allow_fallback_trading": False,
        "maker_fee_bps": 0,
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


def _signals(*timestamps: datetime) -> pd.DataFrame:
    if not timestamps:
        timestamps = (datetime(2026, 6, 1, 0, 0, tzinfo=UTC),)
    return pd.DataFrame(
        {
            "timestamp": list(timestamps),
            "close": [100.0 for _ in timestamps],
        }
    )


def _fill_bars(rows: list[tuple[datetime, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [row[0] for row in rows],
            "open": [row[1] for row in rows],
            "high": [row[2] for row in rows],
            "low": [row[3] for row in rows],
            "close": [row[4] for row in rows],
            "volume": [1.0 for _ in rows],
        }
    )


def test_entry_limit_does_not_fill_if_low_never_reaches_limit_price():
    bars = _fill_bars(
        [
            (datetime(2026, 6, 1, 1, 0, tzinfo=UTC), 100.0, 100.2, 99.99, 100.0),
            (datetime(2026, 6, 1, 1, 15, tzinfo=UTC), 100.0, 100.2, 99.99, 100.0),
        ]
    )

    result = simulate_post_only_maker_fill_scenario(
        _signals(),
        bars,
        _config(),
        _settings(),
        maker_entry_offset_bps=2,
        entry_timeout_minutes=30,
        signal_trade_returns=[0.03],
    )

    assert result["entry_filled_count"] == 0
    assert result["entry_unfilled_count"] == 1
    assert result["missed_winner_count"] == 1


def test_entry_limit_fills_when_low_reaches_limit_price_within_timeout():
    bars = _fill_bars(
        [
            (datetime(2026, 6, 1, 1, 0, tzinfo=UTC), 100.0, 100.2, 99.99, 100.0),
            (datetime(2026, 6, 1, 1, 15, tzinfo=UTC), 100.0, 100.2, 99.97, 100.0),
        ]
    )

    result = simulate_post_only_maker_fill_scenario(
        _signals(),
        bars,
        _config(),
        _settings(),
        maker_entry_offset_bps=2,
        entry_timeout_minutes=30,
    )

    assert result["entry_filled_count"] == 1
    assert result["entry_fill_rate"] == pytest.approx(1.0)
    assert result["avg_minutes_to_entry_fill"] == pytest.approx(15.0)


def test_unfilled_entries_are_canceled_and_never_converted_to_market():
    bars = _fill_bars(
        [
            (datetime(2026, 6, 1, 1, 0, tzinfo=UTC), 100.0, 100.2, 99.99, 100.0),
        ]
    )

    result = simulate_post_only_maker_fill_scenario(
        _signals(),
        bars,
        _config(),
        _settings(),
        maker_entry_offset_bps=2,
        entry_timeout_minutes=15,
    )

    assert result["entry_cancel_rate"] == pytest.approx(1.0)
    assert result["order_rows"][0]["status"] == "canceled"
    assert result["order_rows"][0]["market_fallback_used"] is False
    assert result["trade_rows"] == []


def test_market_fallback_used_remains_false_for_all_simulated_orders_and_trades():
    bars = _fill_bars(
        [
            (datetime(2026, 6, 1, 1, 0, tzinfo=UTC), 100.0, 100.2, 99.97, 100.0),
        ]
    )

    result = simulate_post_only_maker_fill_scenario(
        _signals(),
        bars,
        _config(),
        _settings(),
        maker_entry_offset_bps=2,
        entry_timeout_minutes=15,
    )

    assert result["market_fallback_used_count"] == 0
    assert all(row["market_fallback_used"] is False for row in result["order_rows"])
    assert all(row["market_fallback_used"] is False for row in result["trade_rows"])


def test_conservative_stop_accounting_marks_stop_loss_requires_taker_fallback_true():
    bars = _fill_bars(
        [
            (datetime(2026, 6, 1, 1, 0, tzinfo=UTC), 100.0, 100.2, 99.97, 100.0),
            (datetime(2026, 6, 1, 1, 15, tzinfo=UTC), 100.0, 100.2, 97.5, 98.0),
        ]
    )

    result = simulate_post_only_maker_fill_scenario(
        _signals(),
        bars,
        _config(),
        _settings(),
        maker_entry_offset_bps=2,
        entry_timeout_minutes=15,
    )
    conservative = next(row for row in result["trade_rows"] if row["exit_scenario"] == "conservative_stop_accounting")

    assert conservative["exit_reason"] == "stop_loss_accounting"
    assert conservative["stop_loss_requires_taker_fallback"] is True
    assert result["stop_loss_requires_taker_fallback_count"] == 1


def test_no_market_fallback_stress_scenario_records_adverse_excursion():
    bars = _fill_bars(
        [
            (datetime(2026, 6, 1, 1, 0, tzinfo=UTC), 100.0, 100.2, 99.97, 100.0),
            (datetime(2026, 6, 1, 1, 15, tzinfo=UTC), 100.0, 100.2, 95.0, 96.0),
        ]
    )

    result = simulate_post_only_maker_fill_scenario(
        _signals(),
        bars,
        _config(),
        _settings(),
        maker_entry_offset_bps=2,
        entry_timeout_minutes=15,
    )
    stress = next(row for row in result["trade_rows"] if row["exit_scenario"] == "no_market_fallback_stress")

    assert stress["exit_reason"] == "timeout_after_stop_breach_no_market_fallback"
    assert stress["max_adverse_excursion_pct"] < -0.04
    assert result["worst_adverse_excursion_pct"] < -0.04


def _passing_summary(**overrides):
    summary = {
        "data_ready": True,
        "fill_timeframe_used": "1Min",
        "research_result_valid": True,
        "synthetic_data_used": False,
        "orders_placed": 0,
        "trading_enabled": False,
        "auto_trade_enabled": False,
        "fallback_trading_allowed": False,
        "entry_fill_rate": 0.50,
        "filled_trade_count": 10,
        "conservative_stop_accounting_return_pct": 0.01,
        "conservative_stop_accounting_profit_factor": 1.05,
        "conservative_stop_accounting_max_drawdown_pct": 0.10,
        "market_fallback_used_count": 0,
    }
    summary.update(overrides)
    return summary


def test_simulation_fails_if_entry_fill_rate_below_50pct():
    passed, reasons = evaluate_maker_fill_simulation_pass_fail(_passing_summary(entry_fill_rate=0.49))

    assert passed is False
    assert "entry_fill_rate_too_low" in reasons


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"filled_trade_count": 9}, "filled_trade_count_below_10"),
        ({"conservative_stop_accounting_return_pct": 0.0}, "conservative_return_not_positive"),
        ({"conservative_stop_accounting_profit_factor": 1.049}, "conservative_profit_factor_below_1_05"),
        ({"conservative_stop_accounting_max_drawdown_pct": 0.101}, "drawdown_above_10pct"),
        ({"market_fallback_used_count": 1}, "market_fallback_required"),
    ],
)
def test_simulation_passes_only_with_required_trade_return_pf_drawdown_and_no_market_fallback(override, reason):
    passed, reasons = evaluate_maker_fill_simulation_pass_fail(_passing_summary())
    failed, failed_reasons = evaluate_maker_fill_simulation_pass_fail(_passing_summary(**override))

    assert passed is True
    assert failed is False
    assert reason in failed_reasons
