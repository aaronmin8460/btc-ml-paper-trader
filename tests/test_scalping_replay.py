import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from app.config import Settings
from app.reports.scalping_replay import build_scalping_replay_report


START = datetime(2026, 6, 12, 9, 0, tzinfo=UTC)


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "paper_trading_only": True,
        "symbol": "BTC/USD",
        "trading_enabled": True,
        "auto_trade_enabled": True,
        "scalping_mode_enabled": True,
        "order_notional_usd": 10,
        "max_position_notional_usd": 20,
        "max_total_exposure_usd": 20,
        "order_type": "limit",
        "time_in_force": "ioc",
        "limit_price_offset_bps": 2,
        "max_spread_bps": 4,
        "max_slippage_bps": 6,
        "min_quote_imbalance": 0.0,
        "scalping_buy_probability_floor": 0.50,
        "scalping_confidence_gap_required": 0.01,
        "scalping_entry_dip_pct": 0.0001,
        "scalping_take_profit_pct": 0.001,
        "scalping_stop_loss_pct": 0.001,
        "scalping_trailing_stop_pct": 0.01,
        "trailing_stop_arm_profit_pct": 0.01,
        "emergency_stop_loss_pct": 0.10,
        "scalping_max_position_seconds": 300,
        "scalping_min_hold_seconds": 0,
        "min_hold_seconds_before_weak_quote_exit": 0,
        "scalping_profit_guard_enabled": False,
        "profit_only_exit_enabled": False,
        "paper_fee_bps": 0,
        "paper_slippage_bps": 0,
        "max_trades_per_hour": 100,
        "max_trades_per_10_minutes": 100,
        "max_daily_trades": 1000,
        "max_order_attempts_per_hour": 100,
        "max_order_attempts_per_10_minutes": 100,
        "max_order_attempts_per_day": 1000,
        "min_seconds_between_trades": 0,
        "regime_breakout_threshold": 0.0001,
        "regime_trend_strength_threshold": 0.20,
    }
    values.update(overrides)
    return Settings(**values)


def _write_bars(db_path, closes, *, spread_bps=1.0, quote_imbalance=0.25):
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE collected_market_data ("
            "id INTEGER PRIMARY KEY, symbol TEXT, timeframe TEXT, timestamp TEXT, "
            "open REAL, high REAL, low REAL, close REAL, volume REAL, "
            "spread_bps REAL, quote_imbalance REAL)"
        )
        previous = closes[0]
        for index, close in enumerate(closes):
            opened = previous if index else close
            high = max(opened, close) * 1.0001
            low = min(opened, close) * 0.9999
            connection.execute(
                "INSERT INTO collected_market_data "
                "(symbol, timeframe, timestamp, open, high, low, close, volume, spread_bps, quote_imbalance) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "BTC/USD",
                    "1Min",
                    (START + timedelta(minutes=index)).isoformat(),
                    opened,
                    high,
                    low,
                    close,
                    100 + index,
                    spread_bps,
                    quote_imbalance,
                ),
            )
            previous = close


def _replay(db_path, closes, **setting_overrides):
    _write_bars(db_path, closes)
    return build_scalping_replay_report(
        db_path,
        hours=72,
        settings=_settings(**setting_overrides),
        now=START + timedelta(minutes=len(closes) + 1),
    )


def test_scalping_replay_works_with_simple_rising_data(tmp_path):
    closes = [100 + i * 0.08 for i in range(60)]

    report = _replay(tmp_path / "rising.db", closes)

    assert report["metrics"]["number_of_trades"] > 0
    assert report["metrics"]["win_rate"] > 0
    assert report["metrics"]["total_simulated_pnl"] > 0


def test_scalping_replay_works_with_simple_falling_data(tmp_path):
    closes = [100 - i * 0.04 for i in range(60)]

    report = _replay(tmp_path / "falling.db", closes)

    assert report["metrics"]["number_of_trades"] > 0
    assert report["metrics"]["total_simulated_pnl"] < 0


def test_scalping_replay_only_allows_one_position_at_a_time(tmp_path):
    closes = [100 + i * 0.08 for i in range(90)]

    report = _replay(tmp_path / "one_position.db", closes, scalping_take_profit_pct=0.02)

    assert report["metrics"]["number_of_trades"] > 0
    assert report["metrics"]["max_open_positions"] == 1


def test_scalping_replay_max_holding_time_exits_correctly(tmp_path):
    closes = [100 - min(i, 15) * 0.01 for i in range(50)]

    report = _replay(
        tmp_path / "max_hold.db",
        closes,
        scalping_take_profit_pct=0.50,
        scalping_stop_loss_pct=0.50,
        scalping_trailing_stop_pct=0.50,
        trailing_stop_arm_profit_pct=0.50,
        scalping_max_position_seconds=120,
    )

    assert report["metrics"]["number_of_trades"] > 0
    assert report["breakdowns"]["exit_reasons"][0]["value"] == "scalping_max_position_seconds"


def test_scalping_replay_stop_loss_exits_correctly(tmp_path):
    closes = [100 - i * 0.08 for i in range(60)]

    report = _replay(tmp_path / "stop_loss.db", closes, scalping_stop_loss_pct=0.001)

    exit_reasons = {row["value"] for row in report["breakdowns"]["exit_reasons"]}
    assert "scalping_stop_loss" in exit_reasons


def test_scalping_replay_take_profit_exits_correctly(tmp_path):
    closes = [100 + i * 0.08 for i in range(60)]

    report = _replay(tmp_path / "take_profit.db", closes, scalping_take_profit_pct=0.001)

    exit_reasons = {row["value"] for row in report["breakdowns"]["exit_reasons"]}
    assert "scalping_take_profit" in exit_reasons
    assert report["metrics"]["total_simulated_pnl"] > 0
