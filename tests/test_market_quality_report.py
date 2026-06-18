import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from app.reports.market_quality import (
    bar_age_seconds_bucket,
    build_market_quality_report,
    format_market_quality_report,
    quote_imbalance_bucket,
    save_market_quality_report,
    spread_bps_bucket,
)


NOW = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)


def test_market_quality_report_works_with_empty_db(tmp_path):
    db_path = tmp_path / "empty.db"
    sqlite3.connect(db_path).close()

    report = build_market_quality_report(db_path, hours=48, now=NOW)

    assert report["metrics"]["total_signals"] == 0
    assert report["metrics"]["market_quality_rows"] == 0
    assert report["breakdowns"]["spread_bps_bucket"] == []
    assert any("No tables found" in warning for warning in report["warnings"])


def test_market_quality_bucket_functions():
    assert spread_bps_bucket(2) == "<=2"
    assert spread_bps_bucket(4) == "2-4"
    assert spread_bps_bucket(6) == "4-6"
    assert spread_bps_bucket(10) == "6-10"
    assert spread_bps_bucket(10.1) == ">10"

    assert quote_imbalance_bucket(-0.11) == "<-0.10"
    assert quote_imbalance_bucket(-0.10) == "-0.10-0.00"
    assert quote_imbalance_bucket(0.00) == "0.00-0.05"
    assert quote_imbalance_bucket(0.05) == "0.05-0.10"
    assert quote_imbalance_bucket(0.10) == "0.05-0.10"
    assert quote_imbalance_bucket(0.11) == ">0.10"

    assert bar_age_seconds_bucket(60) == "<=60"
    assert bar_age_seconds_bucket(120) == "60-120"
    assert bar_age_seconds_bucket(180) == "120-180"
    assert bar_age_seconds_bucket(240) == "180-240"
    assert bar_age_seconds_bucket(241) == ">240"


def test_market_quality_report_aggregates_signals_and_market_data(tmp_path):
    db_path = tmp_path / "trading.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE signals ("
            "id INTEGER PRIMARY KEY, created_at TEXT, action TEXT, reason TEXT, "
            "spread_bps REAL, quote_imbalance REAL)"
        )
        connection.execute(
            "CREATE TABLE collected_market_data ("
            "id INTEGER PRIMARY KEY, collected_at TEXT, timestamp TEXT, "
            "spread_bps REAL, quote_imbalance REAL)"
        )
        connection.executemany(
            "INSERT INTO signals (created_at, action, reason, spread_bps, quote_imbalance) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ((NOW - timedelta(minutes=10)).isoformat(), "hold", "spread_too_wide", 5.0, 0.10),
                ((NOW - timedelta(minutes=9)).isoformat(), "hold", "stale_market_data", 3.0, 0.06),
                ((NOW - timedelta(minutes=8)).isoformat(), "hold", "quote_imbalance_too_weak", 1.0, -0.20),
                ((NOW - timedelta(hours=60)).isoformat(), "hold", "spread_too_wide", 99.0, 0.99),
            ],
        )
        connection.executemany(
            "INSERT INTO collected_market_data (collected_at, timestamp, spread_bps, quote_imbalance) "
            "VALUES (?, ?, ?, ?)",
            [
                (
                    (NOW - timedelta(minutes=7)).isoformat(),
                    (NOW - timedelta(minutes=7, seconds=30)).isoformat(),
                    2.0,
                    0.20,
                ),
                (
                    (NOW - timedelta(minutes=6)).isoformat(),
                    (NOW - timedelta(minutes=9)).isoformat(),
                    8.0,
                    -0.05,
                ),
                (
                    (NOW - timedelta(hours=60)).isoformat(),
                    (NOW - timedelta(hours=60, seconds=30)).isoformat(),
                    99.0,
                    0.99,
                ),
            ],
        )

    report = build_market_quality_report(db_path, hours=48, now=NOW)
    metrics = report["metrics"]

    assert report["sources"] == {"signals": "signals", "market_data": "collected_market_data"}
    assert metrics["total_signals"] == 3
    assert metrics["market_quality_rows"] == 5
    assert metrics["spread_too_wide_count"] == 1
    assert metrics["spread_too_wide_pct"] == pytest.approx(1 / 3)
    assert metrics["stale_market_data_count"] == 1
    assert metrics["stale_market_data_pct"] == pytest.approx(1 / 3)
    assert metrics["average_spread_bps"] == pytest.approx(3.8)
    assert metrics["median_spread_bps"] == pytest.approx(3.0)
    assert metrics["p75_spread_bps"] == pytest.approx(5.0)
    assert metrics["p90_spread_bps"] == pytest.approx(6.8)
    assert metrics["p95_spread_bps"] == pytest.approx(7.4)
    assert metrics["min_spread_bps"] == pytest.approx(1.0)
    assert metrics["max_spread_bps"] == pytest.approx(8.0)
    assert metrics["average_quote_imbalance"] == pytest.approx(0.022)
    assert metrics["median_quote_imbalance"] == pytest.approx(0.06)
    assert metrics["pct_spread_bps_lte_max"] == pytest.approx(3 / 5)
    assert metrics["pct_quote_imbalance_gte_min"] == pytest.approx(3 / 5)
    assert metrics["pct_passing_both_filters"] == pytest.approx(2 / 5)
    assert metrics["average_bar_age_seconds"] == pytest.approx(105.0)
    assert metrics["p90_bar_age_seconds"] == pytest.approx(165.0)
    assert metrics["pct_bar_age_lte_max"] == pytest.approx(1 / 2)

    spread_buckets = {row["value"]: row["count"] for row in report["breakdowns"]["spread_bps_bucket"]}
    quote_buckets = {row["value"]: row["count"] for row in report["breakdowns"]["quote_imbalance_bucket"]}
    bar_age_buckets = {row["value"]: row["count"] for row in report["breakdowns"]["bar_age_seconds_bucket"]}

    assert spread_buckets == {"<=2": 2, "2-4": 1, "4-6": 1, "6-10": 1, ">10": 0}
    assert quote_buckets == {
        "<-0.10": 1,
        "-0.10-0.00": 1,
        "0.00-0.05": 0,
        "0.05-0.10": 2,
        ">0.10": 1,
    }
    assert bar_age_buckets == {
        "<=60": 1,
        "60-120": 0,
        "120-180": 1,
        "180-240": 0,
        ">240": 0,
        "(missing)": 3,
    }


def test_market_quality_json_report_is_written(tmp_path):
    db_path = tmp_path / "trading.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE signals ("
            "id INTEGER PRIMARY KEY, created_at TEXT, action TEXT, reason TEXT, "
            "spread_bps REAL, quote_imbalance REAL)"
        )
        connection.execute(
            "INSERT INTO signals (created_at, action, reason, spread_bps, quote_imbalance) "
            "VALUES (?, ?, ?, ?, ?)",
            ((NOW - timedelta(minutes=1)).isoformat(), "hold", "spread_too_wide", 7.0, -0.01),
        )

    report = build_market_quality_report(db_path, hours=48, now=NOW)
    output_path = save_market_quality_report(report, tmp_path / "reports")
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.name == "market_quality_20260612_120000.json"
    assert written["output_path"] == str(output_path)
    assert written["metrics"]["total_signals"] == 1
    assert written["thresholds"]["MAX_SPREAD_BPS"] == pytest.approx(4.0)

    formatted = format_market_quality_report(written)
    assert "BTC/USD Profile A market quality diagnostics" in formatted
    assert "spread_too_wide count" in formatted
