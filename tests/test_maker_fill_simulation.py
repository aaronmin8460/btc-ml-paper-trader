from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import scripts.research_higher_timeframe as rh
from app.config import Settings
from app.db.database import Base
from app.db.models import CollectedMarketData
from scripts.research_higher_timeframe import (
    MAKER_FILL_DEFAULT_PARAMETER_SET_ID,
    MAKER_FILL_FILTERED_VOLUME_ZSCORE_PARAMETER_SET_ID,
    VOLATILITY_BREAKOUT_STRATEGY,
    ResearchConfig,
    build_and_write_maker_fill_simulation,
    evaluate_maker_fill_simulation_pass_fail,
    research_config_from_result_row,
    resolve_maker_fill_target_row,
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


def _collected_session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'maker_fill_collected.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _insert_collected_bar(
    db,
    *,
    timeframe: str,
    timestamp: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1.0,
) -> None:
    db.add(
        CollectedMarketData(
            symbol="BTC/USD",
            timeframe=timeframe,
            timestamp=timestamp,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            source="test",
            source_used="collected_market_data",
        )
    )


def _insert_15min_signal_and_1min_fill_rows(Session, *, start: datetime) -> None:
    with Session() as db:
        for index in range(8):
            open_ = 100.0 + index
            _insert_collected_bar(
                db,
                timeframe="15Min",
                timestamp=start + timedelta(minutes=15 * index),
                open_=open_,
                high=open_ + 2.0,
                low=open_ - 1.0,
                close=open_ + 0.5,
                volume=10.0 + index,
            )
        _insert_collected_bar(
            db,
            timeframe="1Min",
            timestamp=start + timedelta(hours=1),
            open_=100.0,
            high=105.0,
            low=99.97,
            close=104.5,
            volume=3.0,
        )
        db.commit()


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


def test_filtered_volume_zscore_candidate_uses_min_volume_zscore_one():
    row = resolve_maker_fill_target_row(
        [
            {
                "parameter_set_id": MAKER_FILL_FILTERED_VOLUME_ZSCORE_PARAMETER_SET_ID,
                "min_volume_zscore": 0.25,
                "maker_vs_taker_net_gap": 0.04,
            }
        ],
        MAKER_FILL_FILTERED_VOLUME_ZSCORE_PARAMETER_SET_ID,
    )
    assert row is not None
    config = research_config_from_result_row(row)

    assert config.parameter_set_id == MAKER_FILL_FILTERED_VOLUME_ZSCORE_PARAMETER_SET_ID
    assert config.track_id == "M9_v8m_00086_drawdown_reduction_volume_filter"
    assert config.min_volume_zscore == pytest.approx(1.0)
    assert row["maker_vs_taker_net_gap"] == pytest.approx(0.04)


def test_filtered_candidate_regenerates_signals_and_keeps_trading_disabled(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    signal_timestamp = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)

    def fake_build_strategy_research_trades(
        config,
        *,
        bars,
        settings,
        buy_the_dip_features=None,
        v3_features=None,
    ):
        captured["config"] = config
        captured["signal_bar_count"] = len(bars)
        return pd.DataFrame({"buy_exit_return_pct": [0.045]}), _signals(signal_timestamp)

    def fake_select_maker_fill_bars(settings, signal_frame, config, *, session_factory, now):
        fill_bars = _fill_bars(
            [
                (datetime(2026, 6, 1, 1, 0, tzinfo=UTC), 100.0, 105.0, 99.97, 104.5),
            ]
        )
        return "1Min", fill_bars, "test_collected_market_data"

    monkeypatch.setattr(rh, "build_strategy_research_trades", fake_build_strategy_research_trades)
    monkeypatch.setattr(rh, "select_maker_fill_bars", fake_select_maker_fill_bars)

    summary = build_and_write_maker_fill_simulation(
        rows=[
            {
                "parameter_set_id": "v9m_00021",
                "strategy_name": VOLATILITY_BREAKOUT_STRATEGY,
                "timeframe": "1H",
                "min_volume_zscore": 0.25,
            }
        ],
        bars_by_timeframe={
            "1H": _fill_bars([(signal_timestamp, 100.0, 101.0, 99.0, 100.0)]),
        },
        settings=_settings(),
        data_source_reports={
            "1H": {
                "source_used": "collected_market_data_derived_from_15min",
                "row_count": 4000,
                "synthetic_data_used": False,
                "research_result_valid": True,
            }
        },
        parameter_set_id=MAKER_FILL_FILTERED_VOLUME_ZSCORE_PARAMETER_SET_ID,
        summary_path=tmp_path / "maker_fill_simulation_v9m_00021_f1_volume_zscore_1_summary.json",
        session_factory=None,
        maker_entry_offset_bps=2,
        entry_timeout_minutes=60,
        now=datetime(2026, 6, 2, tzinfo=UTC),
    )
    config = captured["config"]

    assert isinstance(config, ResearchConfig)
    assert config.parameter_set_id == MAKER_FILL_FILTERED_VOLUME_ZSCORE_PARAMETER_SET_ID
    assert config.min_volume_zscore == pytest.approx(1.0)
    assert captured["signal_bar_count"] == 1
    assert summary["signal_generation_mode"] == "regenerate_signals_from_1h_data"
    assert summary["filtered_candidate_source"] == "false_breakout_diagnostic"
    assert summary["applied_filter"] == "volume_zscore >= 1"
    assert summary["original_candidate_id"] == "v9m_00021"
    assert summary["filtered_candidate_id"] == MAKER_FILL_FILTERED_VOLUME_ZSCORE_PARAMETER_SET_ID
    assert summary["orders_placed"] == 0
    assert summary["trading_enabled"] is False
    assert summary["auto_trade_enabled"] is False
    assert summary["fallback_trading_allowed"] is False
    assert summary["live_tradable"] is False
    assert summary["strict_paper_forward_eligible"] is False
    assert summary["signal_count"] == 1
    assert summary["entry_filled_count"] == 1


@pytest.mark.parametrize(
    ("parameter_set_id", "expected_min_volume_zscore"),
    [
        (MAKER_FILL_DEFAULT_PARAMETER_SET_ID, 0.25),
        (MAKER_FILL_FILTERED_VOLUME_ZSCORE_PARAMETER_SET_ID, 1.0),
    ],
)
def test_maker_fill_derives_1h_signal_bars_from_15min_collected_data(
    monkeypatch,
    tmp_path,
    parameter_set_id,
    expected_min_volume_zscore,
):
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    Session = _collected_session_factory(tmp_path)
    _insert_15min_signal_and_1min_fill_rows(Session, start=start)
    captured: dict[str, object] = {}

    def fake_build_strategy_research_trades(
        config,
        *,
        bars,
        settings,
        buy_the_dip_features=None,
        v3_features=None,
    ):
        captured["config"] = config
        captured["bars"] = bars.copy()
        return (
            pd.DataFrame({"buy_exit_return_pct": [0.045]}),
            pd.DataFrame(
                {
                    "timestamp": [bars["timestamp"].iloc[0]],
                    "close": [bars["close"].iloc[0]],
                }
            ),
        )

    monkeypatch.setattr(rh, "build_strategy_research_trades", fake_build_strategy_research_trades)

    summary = build_and_write_maker_fill_simulation(
        rows=[
            {
                "parameter_set_id": MAKER_FILL_DEFAULT_PARAMETER_SET_ID,
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
        ],
        bars_by_timeframe={"1H": pd.DataFrame()},
        settings=_settings(),
        data_source_reports={
            "1H": {
                "source_used": "no_valid_real_data_source",
                "row_count": 0,
                "synthetic_data_used": False,
                "research_result_valid": False,
                "requested_max_rows": 2,
            }
        },
        parameter_set_id=parameter_set_id,
        summary_path=tmp_path / f"maker_fill_simulation_{parameter_set_id}_summary.json",
        session_factory=Session,
        maker_entry_offset_bps=2,
        entry_timeout_minutes=60,
        now=start + timedelta(days=3),
    )
    config = captured["config"]
    signal_bars = captured["bars"]

    assert isinstance(config, ResearchConfig)
    assert config.parameter_set_id == parameter_set_id
    assert config.min_volume_zscore == pytest.approx(expected_min_volume_zscore)
    assert len(signal_bars) == 2
    assert signal_bars["timestamp"].iloc[0] == pd.Timestamp(start)
    assert signal_bars["open"].iloc[0] == pytest.approx(100.0)
    assert signal_bars["high"].iloc[0] == pytest.approx(105.0)
    assert signal_bars["low"].iloc[0] == pytest.approx(99.0)
    assert signal_bars["close"].iloc[0] == pytest.approx(103.5)
    assert signal_bars["volume"].iloc[0] == pytest.approx(46.0)
    assert summary["source_used_by_timeframe"]["1H"] == "collected_market_data_derived_from_15min"
    assert summary["row_count"]["1H"] > 0
    assert summary["signal_count"] == 1
    assert summary["entry_filled_count"] == 1
    assert summary["filled_trade_count"] == 1
    assert summary["fill_timeframe_used"] == "1Min"
    assert summary["source_used_by_timeframe"]["1Min"] == "collected_market_data"
    assert summary["orders_placed"] == 0
    assert summary["trading_enabled"] is False
    assert summary["auto_trade_enabled"] is False
    assert summary["fallback_trading_allowed"] is False
    assert summary["live_tradable"] is False
    assert summary["strict_paper_forward_eligible"] is False
