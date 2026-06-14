from pathlib import Path

import pytest

from app.config import Settings
from app.reports.config_health import build_config_health_report, estimated_local_round_trip_cost_pct


PROFILE_A = Path("profiles/profile_a_true_micro_scalping.env.example")


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "paper_trading_only": True,
        "symbol": "BTC/USD",
        "trading_enabled": True,
        "auto_trade_enabled": True,
        "scalping_mode_enabled": True,
        "scalping_take_profit_pct": 0.0015,
        "scalping_stop_loss_pct": 0.0010,
        "scalping_label_take_profit_pct": 0.0015,
        "scalping_label_stop_loss_pct": 0.0010,
        "scalping_label_horizon_bars": 3,
        "scalping_max_position_seconds": 180,
        "paper_fee_bps": 0,
        "paper_slippage_bps": 0,
        "max_spread_bps": 4,
        "max_trades_per_hour": 3,
        "min_seconds_between_trades": 300,
    }
    values.update(overrides)
    return Settings(**values)


def _warn_names(report: dict) -> set[str]:
    return {check["name"] for check in report["checks"] if check["status"] == "WARN"}


def test_profile_a_config_health_passes():
    settings = Settings(_env_file=PROFILE_A)

    report = build_config_health_report(settings)

    assert report["overall_status"] == "PASS"
    assert report["summary"]["warn_count"] == 0
    assert estimated_local_round_trip_cost_pct(settings) == pytest.approx(0.0004)
    assert report["execution_settings"]["SCALPING_TAKE_PROFIT_PCT"] == pytest.approx(
        report["label_settings"]["SCALPING_LABEL_TAKE_PROFIT_PCT"]
    )
    assert report["execution_settings"]["SCALPING_STOP_LOSS_PCT"] == pytest.approx(
        report["label_settings"]["SCALPING_LABEL_STOP_LOSS_PCT"]
    )


def test_mismatched_label_execution_warns():
    report = build_config_health_report(
        _settings(
            scalping_label_take_profit_pct=0.0020,
            scalping_label_stop_loss_pct=0.0008,
        )
    )

    assert report["overall_status"] == "WARN"
    assert {
        "label_take_profit_matches_execution",
        "label_stop_loss_matches_execution",
    } <= _warn_names(report)


def test_huge_max_holding_time_warns():
    report = build_config_health_report(_settings(scalping_max_position_seconds=1000))

    assert report["overall_status"] == "WARN"
    assert "max_holding_time_matches_label_horizon" in _warn_names(report)


def test_take_profit_below_fee_estimate_warns():
    report = build_config_health_report(
        _settings(
            paper_fee_bps=10,
            paper_slippage_bps=10,
            max_spread_bps=6,
        )
    )

    assert report["overall_status"] == "WARN"
    assert "take_profit_covers_local_cost" in _warn_names(report)
    cost = report["cost_estimate"]["round_trip_estimated_cost_pct"]
    assert cost == pytest.approx(0.0046)
    assert report["execution_settings"]["SCALPING_TAKE_PROFIT_PCT"] < cost
