import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from app.reports.strategy_health import (
    build_strategy_health_report,
    buy_probability_bucket,
    calculate_pnl_metrics,
    confidence_gap_bucket,
    holding_time_bucket,
    quote_imbalance_bucket,
    save_strategy_health_report,
    spread_bps_bucket,
)


NOW = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)


def test_strategy_health_report_works_with_empty_db(tmp_path):
    db_path = tmp_path / "empty.db"
    sqlite3.connect(db_path).close()

    report = build_strategy_health_report(db_path, hours=48, now=NOW)

    assert report["metrics"]["total_decisions"] is None
    assert report["metrics"]["completed_trades"] is None
    assert report["breakdowns"]["decision_action"] == []
    assert any("No tables found" in warning for warning in report["warnings"])


def test_strategy_health_report_works_when_optional_tables_are_missing(tmp_path):
    db_path = tmp_path / "signals_only.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE signals ("
            "id INTEGER PRIMARY KEY, created_at TEXT, action TEXT, "
            "buy_probability REAL, sell_probability REAL, reason TEXT)"
        )
        connection.executemany(
            "INSERT INTO signals (created_at, action, buy_probability, sell_probability, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ((NOW - timedelta(minutes=5)).isoformat(), "buy", 0.61, 0.32, "scalping_entry_approved"),
                (
                    (NOW - timedelta(minutes=4)).isoformat(),
                    "hold",
                    0.49,
                    0.45,
                    "scalping_buy_probability_below_floor",
                ),
                ((NOW - timedelta(minutes=3)).isoformat(), "sell", 0.20, 0.72, "scalping_take_profit"),
            ],
        )

    report = build_strategy_health_report(db_path, hours=48, now=NOW)

    assert report["sources"]["decisions"] == "signals"
    assert report["sources"]["trades"] is None
    assert report["metrics"]["total_decisions"] == 3
    assert report["metrics"]["buy_decisions"] == 1
    assert report["metrics"]["sell_decisions"] == 1
    assert report["metrics"]["hold_decisions"] == 1
    assert report["metrics"]["completed_trades"] is None
    assert report["metrics"]["ioc_cancellations"] is None
    assert {"value": "ml_filter", "count": 1, "pct": 1.0} in report["breakdowns"]["blocked_by"]
    assert any("No trade table found" in warning for warning in report["warnings"])
    assert any("No order table found" in warning for warning in report["warnings"])


def test_strategy_health_bucket_functions():
    assert buy_probability_bucket(0.49) == "<0.50"
    assert buy_probability_bucket(0.50) == "0.50-0.54"
    assert buy_probability_bucket(0.55) == "0.55-0.59"
    assert buy_probability_bucket(0.60) == "0.60-0.64"
    assert buy_probability_bucket(0.65) == ">=0.65"

    assert confidence_gap_bucket(0.019) == "<0.02"
    assert confidence_gap_bucket(0.02) == "0.02-0.04"
    assert confidence_gap_bucket(0.04) == "0.04-0.08"
    assert confidence_gap_bucket(0.08) == ">=0.08"

    assert spread_bps_bucket(2) == "<=2"
    assert spread_bps_bucket(4) == "2-4"
    assert spread_bps_bucket(6) == "4-6"
    assert spread_bps_bucket(10) == "6-10"
    assert spread_bps_bucket(10.1) == ">10"

    assert quote_imbalance_bucket(-0.11) == "<-0.10"
    assert quote_imbalance_bucket(-0.10) == "-0.10-0.00"
    assert quote_imbalance_bucket(0.00) == "0.00-0.05"
    assert quote_imbalance_bucket(0.05) == "0.05-0.10"
    assert quote_imbalance_bucket(0.11) == ">0.10"

    assert holding_time_bucket(29.9) == "<30s"
    assert holding_time_bucket(30) == "30-90s"
    assert holding_time_bucket(90) == "90-180s"
    assert holding_time_bucket(180) == "180-900s"
    assert holding_time_bucket(901) == ">900s"


def test_strategy_health_expectancy_calculation_is_correct():
    metrics = calculate_pnl_metrics([10, -4, -6, 0])

    assert metrics["pnl_sample_size"] == 4
    assert metrics["win_rate"] == pytest.approx(0.25)
    assert metrics["average_win"] == pytest.approx(10)
    assert metrics["average_loss"] == pytest.approx(-5)
    assert metrics["expectancy"] == pytest.approx(0)
    assert metrics["total_realized_pnl"] == pytest.approx(0)
    assert metrics["max_consecutive_losses"] == 2


def test_strategy_health_json_report_is_written(tmp_path):
    db_path = tmp_path / "trading.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE signals ("
            "id INTEGER PRIMARY KEY, created_at TEXT, action TEXT, reason TEXT, "
            "blocked_by TEXT, strategy_name TEXT, regime TEXT, "
            "buy_probability REAL, sell_probability REAL, spread_bps REAL, quote_imbalance REAL)"
        )
        connection.execute(
            "CREATE TABLE trades ("
            "id INTEGER PRIMARY KEY, created_at TEXT, side TEXT, net_pnl REAL, hold_seconds REAL, reason TEXT)"
        )
        connection.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, created_at TEXT, status TEXT, raw_response TEXT)"
        )
        connection.executemany(
            "INSERT INTO signals (created_at, action, reason, blocked_by, strategy_name, regime, "
            "buy_probability, sell_probability, spread_bps, quote_imbalance) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    (NOW - timedelta(minutes=10)).isoformat(),
                    "buy",
                    "scalping_entry_approved",
                    None,
                    "mean_reversion_scalping",
                    "mean_reverting",
                    0.61,
                    0.30,
                    3.0,
                    0.07,
                ),
                (
                    (NOW - timedelta(minutes=9)).isoformat(),
                    "hold",
                    "quote_imbalance_too_weak",
                    "quote_imbalance",
                    "momentum_breakout",
                    "trending",
                    0.52,
                    0.47,
                    7.0,
                    -0.04,
                ),
            ],
        )
        connection.executemany(
            "INSERT INTO trades (created_at, side, net_pnl, hold_seconds, reason) VALUES (?, ?, ?, ?, ?)",
            [
                ((NOW - timedelta(minutes=6)).isoformat(), "sell", 3.0, 45.0, "scalping_take_profit"),
                ((NOW - timedelta(minutes=5)).isoformat(), "sell", -2.0, 120.0, "scalping_stop_loss"),
                ((NOW - timedelta(minutes=4)).isoformat(), "buy", 0.0, None, "entry_fill"),
            ],
        )
        connection.executemany(
            "INSERT INTO orders (created_at, status, raw_response) VALUES (?, ?, ?)",
            [
                (
                    (NOW - timedelta(minutes=3)).isoformat(),
                    "canceled",
                    json.dumps({"order_type": "limit", "time_in_force": "ioc"}),
                ),
                (
                    (NOW - timedelta(minutes=2)).isoformat(),
                    "canceled",
                    json.dumps({"order_type": "limit", "time_in_force": "gtc"}),
                ),
                (
                    (NOW - timedelta(minutes=1)).isoformat(),
                    "canceled",
                    json.dumps({"cancel_reason": "ioc_no_fill"}),
                ),
            ],
        )

    report = build_strategy_health_report(db_path, hours=48, now=NOW)
    output_path = save_strategy_health_report(report, tmp_path / "reports")
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.name == "strategy_health_20260612_120000.json"
    assert written["output_path"] == str(output_path)
    assert written["metrics"]["completed_trades"] == 2
    assert written["metrics"]["win_rate"] == pytest.approx(0.5)
    assert written["metrics"]["total_realized_pnl"] == pytest.approx(1.0)
    assert written["metrics"]["average_holding_time_seconds"] == pytest.approx(82.5)
    assert written["metrics"]["median_holding_time_seconds"] == pytest.approx(82.5)
    assert written["metrics"]["ioc_cancellations"] == 2
