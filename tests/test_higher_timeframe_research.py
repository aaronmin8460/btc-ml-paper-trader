import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.database import Base, connect_args_for_database_url
from app.db.models import CollectedMarketData
from app.risk.risk_manager import PositionState
from app.strategy.strategies import MarketContext, MarketRegime, TrendPullbackStrategy
from scripts.research_higher_timeframe import (
    BUY_THE_DIP_SIGNAL_PROFILES,
    BUY_THE_DIP_STRATEGY,
    UPTREND_PULLBACK_STRATEGY,
    VOLATILITY_BREAKOUT_STRATEGY,
    VOLATILITY_FOCUS_STRATEGY,
    EXIT_MODE_BREAK_EVEN_AFTER_1R,
    EXIT_MODE_BREAK_EVEN_1R,
    EXIT_MODE_MFE_PROTECT_1R_50,
    EXIT_MODE_MFE_PROTECTION_EXIT,
    EXIT_MODE_TRAILING_AFTER_1R,
    EXIT_MODE_TIME_STOP_MOMENTUM_WEAK,
    EXIT_MODE_TRAILING_1R,
    VOLATILITY_FOCUS_TRACK_A,
    VOLATILITY_FOCUS_TRACK_B,
    VOLATILITY_FOCUS_TRACK_M,
    VOLATILITY_FOCUS_TRACK_M9,
    VOLATILITY_FOCUS_TRACK_T,
    ResearchDataReport,
    ResearchConfig,
    _resolve_requested_timeframes,
    aggregate_trade_diagnostics,
    build_baselines_by_timeframe,
    build_volatility_focus_summary,
    build_trade_audit_rows,
    build_uptrend_pullback_research_trades,
    build_buy_the_dip_research_trades,
    build_volatility_breakout_research_trades,
    build_research_summary,
    classify_cost_sensitivity,
    chronological_walk_forward_splits,
    derive_1d_bars_from_lower_timeframe,
    derive_1h_bars_from_15min,
    derive_4h_bars_from_lower_timeframe,
    evaluate_cost_scenarios_for_config,
    evaluate_research_configs,
    export_trade_audit_logs,
    _fetch_research_bars,
    generate_buy_the_dip_configs,
    generate_buy_the_dip_signal_profiles,
    generate_htf_risk_off_hold_filter_configs,
    generate_htf_configs,
    generate_htf_trend_continuation_configs,
    generate_htf_volatility_expansion_breakout_configs,
    generate_volatility_focus_configs,
    generate_research_configs,
    generate_trend_pullback_configs,
    generate_uptrend_pullback_configs,
    generate_volatility_breakout_configs,
    paper_forward_readiness_gate,
    prepare_buy_the_dip_features,
    research_rank_details,
    research_settings,
    resolve_research_exit,
    run_higher_timeframe_research,
    strategy_baseline_comparison,
    summarize_walk_forward_metrics,
    volatility_focus_maker_research_gate,
    volatility_focus_research_gate,
    volatility_focus_trade_diagnostics,
    write_volatility_focus_outputs,
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


def _session_factory(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'research.db'}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))
    Base.metadata.create_all(bind=engine)
    return engine, sessionmaker(bind=engine)


def _bars(*, latest: datetime, count: int, step_minutes: int = 5) -> pd.DataFrame:
    timestamps = [latest - timedelta(minutes=step_minutes * (count - 1 - index)) for index in range(count)]
    prices = [100.0 + index for index in range(count)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": [price + 1.0 for price in prices],
            "low": [price - 1.0 for price in prices],
            "close": [price + 0.5 for price in prices],
            "volume": [1.0 + index for index in range(count)],
        }
    )


def _insert_collected_rows(Session, *, timeframe: str, latest: datetime, count: int, step_minutes: int) -> None:
    bars = _bars(latest=latest, count=count, step_minutes=step_minutes)
    with Session() as db:
        for _, row in bars.iterrows():
            db.add(
                CollectedMarketData(
                    symbol="BTC/USD",
                    timeframe=timeframe,
                    timestamp=row["timestamp"].to_pydatetime()
                    if hasattr(row["timestamp"], "to_pydatetime")
                    else row["timestamp"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    collected_at=latest,
                )
            )
        db.commit()


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


def test_buy_the_dip_v2_rejected_strategy_documentation_exists():
    note = (Path("docs/strategy_research_v3.md")).read_text(encoding="utf-8").lower()

    assert "buy-the-dip mean reversion v2 rejected" in note
    assert "roughly 180 days" in note
    assert "do not train on it" in note
    assert "zero" in note


def test_research_config_space_matches_requested_values():
    configs = generate_trend_pullback_configs()

    assert {config.timeframe for config in configs} == {"5Min", "15Min"}
    assert {config.take_profit_pct for config in configs} == {0.008, 0.01, 0.015, 0.02}
    assert {config.stop_loss_pct for config in configs} == {0.003, 0.005, 0.008}
    assert {config.max_hold_bars for config in configs} == {6, 12, 24, 48}
    all_configs = generate_research_configs()
    buy_the_dip_configs = generate_buy_the_dip_configs()
    uptrend_configs = generate_uptrend_pullback_configs()
    breakout_configs = generate_volatility_breakout_configs()
    htf_default_configs = generate_htf_configs(timeframes=("15Min", "1H"))
    assert len(all_configs) == (
        48 + len(buy_the_dip_configs) + len(uptrend_configs) + len(breakout_configs) + len(htf_default_configs)
    )
    assert {config.strategy_name for config in configs} == {"trend_pullback"}
    assert {config.strategy_name for config in buy_the_dip_configs} == {BUY_THE_DIP_STRATEGY}
    assert {config.strategy_name for config in uptrend_configs} == {UPTREND_PULLBACK_STRATEGY}
    assert {config.strategy_name for config in breakout_configs} == {VOLATILITY_BREAKOUT_STRATEGY}
    assert {config.timeframe for config in uptrend_configs} == {"15Min", "1H"}
    assert {config.take_profit_pct for config in uptrend_configs} == {0.015, 0.02, 0.03, 0.04}
    assert {config.stop_loss_pct for config in uptrend_configs} == {0.008, 0.012, 0.015, 0.02}
    assert {config.take_profit_pct for config in breakout_configs} == {0.02, 0.03, 0.04, 0.05}
    assert {config.stop_loss_pct for config in breakout_configs} == {0.01, 0.015, 0.02}
    assert {config.take_profit_pct for config in buy_the_dip_configs} == {0.0086, 0.01, 0.0125, 0.015, 0.02, 0.025}
    assert len(generate_buy_the_dip_signal_profiles()) > len(BUY_THE_DIP_SIGNAL_PROFILES)
    assert any(config.timeframe == "15Min" for config in generate_buy_the_dip_configs(max_configs=120))


def test_higher_timeframe_strategy_templates_use_1h_4h_daily():
    trend = generate_htf_trend_continuation_configs(max_configs=1000)
    breakout = generate_htf_volatility_expansion_breakout_configs(max_configs=1000)
    risk_off = generate_htf_risk_off_hold_filter_configs(max_configs=1000)

    assert {config.timeframe for config in trend} == {"1H", "4H", "1D"}
    assert {config.take_profit_pct for config in trend} == {0.03, 0.05, 0.08, 0.12}
    assert {config.stop_loss_pct for config in trend} == {0.015, 0.025, 0.04, 0.06}
    assert {config.timeframe for config in breakout} == {"1H", "4H", "1D"}
    assert {config.take_profit_pct for config in breakout} == {0.04, 0.06, 0.10, 0.15}
    assert {config.stop_loss_pct for config in breakout} == {0.02, 0.03, 0.05}
    assert {config.strategy_name for config in risk_off} == {"htf_risk_off_hold_filter"}


def test_volatility_focus_mode_defaults_to_1h_and_samples_deterministically():
    assert _resolve_requested_timeframes(VOLATILITY_FOCUS_STRATEGY, None) == ("1H",)
    configs = generate_volatility_focus_configs(max_configs=40)
    second = generate_volatility_focus_configs(max_configs=40)

    assert len(configs) == 40
    assert [config.parameter_set_id for config in configs] == [config.parameter_set_id for config in second]
    assert {config.timeframe for config in configs} == {"1H"}
    assert {config.strategy_name for config in configs} == {
        VOLATILITY_BREAKOUT_STRATEGY,
        "htf_volatility_expansion_breakout",
    }
    assert any(config.parameter_set_id.startswith("vff_htfb") for config in configs)
    assert any(config.parameter_set_id.startswith("vff_vbo") for config in configs)
    explicit_15min = generate_volatility_focus_configs(max_configs=4, timeframes=("15Min",))
    assert {config.timeframe for config in explicit_15min} == {"15Min"}


def test_volatility_focus_strategy_generates_only_breakout_families():
    configs = generate_research_configs(
        strategy=VOLATILITY_FOCUS_STRATEGY,
        max_v3_configs=20,
        timeframes=("1H", "4H"),
    )

    assert len(configs) == 20
    assert {config.timeframe for config in configs} <= {"1H", "4H"}
    assert {config.strategy_name for config in configs} == {
        VOLATILITY_BREAKOUT_STRATEGY,
        "htf_volatility_expansion_breakout",
    }
    assert "15Min" not in {config.timeframe for config in configs}


def test_volatility_focus_v7_adds_targeted_1h_tracks_and_exit_modes():
    configs = generate_volatility_focus_configs(max_configs=4000, timeframes=("1H",))
    track_a = [config for config in configs if config.track_id == VOLATILITY_FOCUS_TRACK_A]
    track_b = [config for config in configs if config.track_id == VOLATILITY_FOCUS_TRACK_B]

    assert len(configs) == 4000
    assert {config.timeframe for config in configs} == {"1H"}
    assert {config.strategy_name for config in track_a + track_b} == {VOLATILITY_BREAKOUT_STRATEGY}
    assert len(track_a) == 1500
    assert len(track_b) == 1500
    assert {config.exit_mode for config in track_a + track_b} == {
        "fixed_tp_sl_timeout",
        EXIT_MODE_BREAK_EVEN_1R,
        EXIT_MODE_TRAILING_1R,
        EXIT_MODE_MFE_PROTECT_1R_50,
        EXIT_MODE_TIME_STOP_MOMENTUM_WEAK,
    }
    assert track_a[0].take_profit_pct == 0.06
    assert track_a[0].stop_loss_pct == 0.02
    assert track_a[0].max_hold_bars == 96
    assert track_b[0].take_profit_pct == 0.04
    assert track_b[0].max_hold_bars == 48


def test_volatility_focus_v8_adds_maker_and_taker_survival_tracks():
    configs = generate_volatility_focus_configs(max_configs=5000, timeframes=("1H",))
    track_m = [config for config in configs if config.track_id == VOLATILITY_FOCUS_TRACK_M]
    track_t = [config for config in configs if config.track_id == VOLATILITY_FOCUS_TRACK_T]

    assert len(configs) == 5000
    assert {config.timeframe for config in configs} == {"1H"}
    assert {config.strategy_name for config in configs} == {VOLATILITY_BREAKOUT_STRATEGY}
    assert len(track_m) == 2500
    assert len(track_t) == 2500
    assert {config.exit_mode for config in track_m} == {
        "fixed_tp_sl_timeout",
        EXIT_MODE_TIME_STOP_MOMENTUM_WEAK,
        EXIT_MODE_MFE_PROTECTION_EXIT,
    }
    assert {config.exit_mode for config in track_t} == {
        "fixed_tp_sl_timeout",
        EXIT_MODE_BREAK_EVEN_AFTER_1R,
        EXIT_MODE_TRAILING_AFTER_1R,
        EXIT_MODE_MFE_PROTECTION_EXIT,
        EXIT_MODE_TIME_STOP_MOMENTUM_WEAK,
    }
    assert track_m[0].take_profit_pct == 0.045
    assert track_m[0].stop_loss_pct == 0.02
    assert track_m[0].max_hold_bars == 48
    assert track_m[0].min_volume_zscore == 0.25
    assert track_t[0].take_profit_pct == 0.07
    assert track_t[0].max_hold_bars == 96


def test_volatility_focus_v9_final_drawdown_reduction_track_is_narrow():
    configs = generate_volatility_focus_configs(max_configs=3000, timeframes=("1H",))

    assert len(configs) == 3000
    assert {config.track_id for config in configs} == {VOLATILITY_FOCUS_TRACK_M9}
    assert {config.strategy_name for config in configs} == {VOLATILITY_BREAKOUT_STRATEGY}
    assert {config.timeframe for config in configs} == {"1H"}
    assert {config.exit_mode for config in configs} <= {
        "fixed_tp_sl_timeout",
        EXIT_MODE_TIME_STOP_MOMENTUM_WEAK,
        EXIT_MODE_MFE_PROTECTION_EXIT,
        EXIT_MODE_BREAK_EVEN_AFTER_1R,
    }
    assert {config.parameter_set_id for config in configs} == {f"v9m_{index:05d}" for index in range(3000)}

    anchor = configs[0]
    assert anchor.exit_mode == "fixed_tp_sl_timeout"
    assert anchor.take_profit_pct == 0.045
    assert anchor.stop_loss_pct == 0.022
    assert anchor.max_hold_bars == 48
    assert anchor.breakout_lookback == 20
    assert anchor.consolidation_lookback == 12
    assert anchor.min_body_vs_avg == 1.2
    assert anchor.min_recent_return_pct == 0.003
    assert anchor.min_trend_strength == 0.0
    assert anchor.max_atr_expansion == 3.0
    assert anchor.min_volume_zscore == 0.25


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


def test_volatility_focus_research_gate_keeps_active_model_out_of_research_rejections():
    gate = volatility_focus_research_gate(
        _passing_metrics(number_of_trades=25, max_drawdown_pct=0.005),
        _settings(),
        _volatility_breakout_config(timeframe="1H", stop_loss_pct=0.02),
        cost_summary={
            "net_return_by_cost_scenario": {"current_taker": 0.02},
            "profit_factor_by_cost_scenario": {"current_taker": 1.2},
        },
        source_report=ResearchDataReport(
            timeframe="1H",
            source_used="collected_market_data",
            latest_timestamp="2026-06-07T09:00:00+00:00",
            data_age_minutes=0.0,
            row_count=500,
            synthetic_data_used=False,
            research_result_valid=True,
        ),
        synthetic_data_used=False,
        research_result_valid=True,
        baseline_comparison={
            "beats_buy_hold_risk_adjusted": True,
            "beats_dca_daily_risk_adjusted": True,
        },
        walk_forward={"walk_forward_passed": True, "folds_with_min_trades_count": 3},
        concentration=0.20,
        active_model_valid=False,
        min_focused_trades=20,
    )

    assert gate["research_promising"] is True
    assert gate["economically_viable"] is True
    assert gate["paper_forward_eligible"] is False
    assert "active_model_invalid" not in gate["research_rejection_reasons"]
    assert "active_model_invalid" in gate["paper_forward_rejection_reasons"]
    assert "active_model_invalid" in gate["training_rejection_reasons"]


def test_volatility_focus_research_gate_enforces_trade_fold_cost_and_baseline_gates():
    gate = volatility_focus_research_gate(
        _passing_metrics(number_of_trades=12, max_drawdown_pct=0.02),
        _settings(max_backtest_drawdown_pct=0.01),
        _volatility_breakout_config(timeframe="1H"),
        cost_summary={
            "net_return_by_cost_scenario": {"current_taker": -0.01},
            "profit_factor_by_cost_scenario": {"current_taker": 0.9},
        },
        source_report=ResearchDataReport(
            timeframe="1H",
            source_used="collected_market_data",
            latest_timestamp="2026-06-07T09:00:00+00:00",
            data_age_minutes=0.0,
            row_count=500,
            synthetic_data_used=False,
            research_result_valid=True,
        ),
        synthetic_data_used=False,
        research_result_valid=True,
        baseline_comparison={
            "beats_buy_hold_risk_adjusted": False,
            "beats_dca_daily_risk_adjusted": False,
        },
        walk_forward={"walk_forward_passed": False, "folds_with_min_trades_count": 2},
        concentration=0.20,
        active_model_valid=True,
        min_focused_trades=20,
    )

    reasons = gate["research_rejection_reasons"]
    assert "number_of_trades_below_20" in reasons
    assert "folds_with_min_trades_below_3" in reasons
    assert "walk_forward_not_passed" in reasons
    assert "current_taker_net_return_not_positive" in reasons
    assert "current_taker_profit_factor_below_1_05" in reasons
    assert "does_not_beat_buy_and_hold_risk_adjusted" in reasons
    assert "does_not_beat_dca_risk_adjusted" in reasons


def test_maker_research_gate_does_not_override_current_taker_gate():
    metrics = _passing_metrics(number_of_trades=24, max_drawdown_pct=0.005)
    settings = _settings(max_backtest_drawdown_pct=0.01)
    config = _volatility_breakout_config(timeframe="1H", track_id=VOLATILITY_FOCUS_TRACK_M)
    source_report = ResearchDataReport(
        timeframe="1H",
        source_used="collected_market_data_derived_from_15min",
        latest_timestamp="2026-06-07T09:00:00+00:00",
        data_age_minutes=0.0,
        row_count=3998,
        synthetic_data_used=False,
        research_result_valid=True,
    )
    cost_summary = {
        "net_return_by_cost_scenario": {"current_taker": -0.02, "maker_current": 0.03},
        "profit_factor_by_cost_scenario": {"current_taker": 0.90, "maker_current": 1.20},
    }
    baseline = {
        "beats_buy_hold_risk_adjusted": True,
        "beats_dca_daily_risk_adjusted": True,
    }
    walk_forward = {"walk_forward_passed": True, "folds_with_min_trades_count": 4, "fold_count": 4}

    current_gate = volatility_focus_research_gate(
        metrics,
        settings,
        config,
        cost_summary=cost_summary,
        source_report=source_report,
        synthetic_data_used=False,
        research_result_valid=True,
        baseline_comparison=baseline,
        walk_forward=walk_forward,
        concentration=0.20,
        active_model_valid=True,
        min_focused_trades=20,
    )
    maker_gate = volatility_focus_maker_research_gate(
        metrics,
        settings,
        config,
        cost_summary=cost_summary,
        source_report=source_report,
        synthetic_data_used=False,
        research_result_valid=True,
        baseline_comparison=baseline,
        walk_forward=walk_forward,
        min_focused_trades=20,
    )

    assert current_gate["research_promising"] is False
    assert "current_taker_net_return_not_positive" in current_gate["research_rejection_reasons"]
    assert maker_gate["maker_research_promising"] is True
    assert maker_gate["maker_economically_viable"] is True
    assert maker_gate["maker_rejection_reasons"] == []


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


def test_paper_forward_readiness_rejects_take_profit_below_cost():
    result = paper_forward_readiness_gate(
        _passing_metrics(),
        _settings(),
        fallback_prediction_used=False,
        active_model_valid=True,
        research_result_valid=True,
        take_profit_pct=0.005,
        round_trip_estimated_cost_pct=0.0076,
        promotion_required_return_pct=0.0086,
        source_used="collected_market_data",
    )

    assert result["economically_viable"] is False
    assert result["paper_forward_eligible"] is False
    assert "take_profit_not_above_round_trip_cost" in result["rejection_reasons"]
    assert "take_profit_below_promotion_required_return" in result["rejection_reasons"]


def test_paper_forward_readiness_requires_collected_real_data_source():
    result = paper_forward_readiness_gate(
        _passing_metrics(),
        _settings(),
        fallback_prediction_used=False,
        active_model_valid=True,
        research_result_valid=True,
        take_profit_pct=0.01,
        round_trip_estimated_cost_pct=0.0076,
        promotion_required_return_pct=0.0086,
        source_used="synthetic_explicit_test_demo_mode",
    )

    assert result["economically_viable"] is False
    assert "data_source_not_collected_market_data" in result["rejection_reasons"]


def test_paper_forward_readiness_rejects_high_single_trade_concentration():
    result = paper_forward_readiness_gate(
        _passing_metrics(
            number_of_trades=25,
            trade_details=[
                {"net_return_pct": 0.020},
                {"net_return_pct": 0.002},
                {"net_return_pct": 0.001},
                {"net_return_pct": -0.001},
            ],
        ),
        _settings(),
        fallback_prediction_used=False,
        active_model_valid=True,
        research_result_valid=True,
        take_profit_pct=0.01,
        round_trip_estimated_cost_pct=0.0076,
        promotion_required_return_pct=0.0086,
        source_used="collected_market_data",
    )

    assert result["economically_viable"] is False
    assert "single_trade_return_concentration_too_high" in result["rejection_reasons"]


def test_baselines_cover_buy_hold_dca_and_strategy_comparison():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=8, freq="12h", tz="UTC"),
            "open": [100, 90, 95, 92, 100, 105, 110, 114],
            "high": [101, 96, 96, 101, 106, 111, 115, 121],
            "low": [99, 89, 91, 91, 99, 104, 109, 113],
            "close": [100, 90, 95, 92, 100, 105, 110, 120],
            "volume": [1.0] * 8,
        }
    )
    baselines = build_baselines_by_timeframe({"1H": bars}, _settings(order_notional_usd=25))
    baseline = baselines["1H"]
    comparison = strategy_baseline_comparison(
        {"net_return_pct": 0.25, "max_drawdown_pct": 0.02},
        baseline,
    )

    assert baseline["cash_return_pct"] == 0.0
    assert baseline["buy_and_hold_return_pct"] == pytest.approx(0.20)
    assert baseline["dca_daily_return_pct"] > 0
    assert baseline["dca_weekly_return_pct"] > 0
    assert comparison["strategy_excess_return_vs_buy_hold"] == pytest.approx(0.05)
    assert comparison["beats_any_relevant_baseline_risk_adjusted"] is True


def test_cost_scenarios_and_trade_audit_cost_decomposition_do_not_modify_env():
    env_path = Path(".env")
    before = env_path.read_text(encoding="utf-8") if env_path.exists() else None
    settings = research_settings(_settings(order_type="market", scalping_mode_enabled=True))
    config = ResearchConfig(
        parameter_set_id="audit_cost",
        strategy_name="unit_test_strategy",
        timeframe="15Min",
        take_profit_pct=0.004,
        stop_loss_pct=0.01,
        max_hold_bars=4,
    )
    trades = pd.DataFrame(
        [
            {
                "timestamp": datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
                "entry_timestamp": datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
                "exit_timestamp": datetime(2026, 6, 1, 1, 0, tzinfo=UTC),
                "close": 100.0,
                "entry_price": 100.0,
                "exit_price": 100.4,
                "buy_quality_label": 1,
                "buy_exit_return_pct": 0.004,
                "buy_exit_reason": "research_take_profit",
                "buy_hold_bars": 4,
                "strategy_name": "unit_test_strategy",
                "entry_reason": "unit_test",
                "regime": "unit",
                "max_favorable_excursion_pct": 0.006,
                "max_adverse_excursion_pct": -0.001,
            }
        ]
    )

    cost_summary = evaluate_cost_scenarios_for_config(trades, trades, settings, config)
    metrics = evaluate_cost_scenarios_for_config(
        trades,
        trades,
        settings,
        config,
    )
    audit_rows = build_trade_audit_rows(
        config,
        trades,
        {
            "trade_details": [
                {
                    "gross_return_pct": 0.004,
                    "net_return_pct": metrics["net_return_by_cost_scenario"]["current_taker"],
                    "fee_amount": settings.order_notional_usd * 0.005,
                    "slippage_amount": settings.order_notional_usd * 0.002,
                    "spread_cost": 0.0,
                    "exit_reason": "research_take_profit",
                    "hold_bars": 4,
                }
            ]
        },
        result_row={"rejection_reasons": "unit_test"},
        settings=settings,
        folds=[],
    )

    assert cost_summary["profitable_under_zero_cost"] is True
    assert cost_summary["profitable_under_maker_low_slippage"] is True
    assert cost_summary["profitable_under_current_taker"] is False
    assert cost_summary["cost_sensitivity_classification"] == "signal_positive_low_cost_only"
    assert classify_cost_sensitivity(
        profitable_zero=False,
        profitable_low=False,
        profitable_maker=False,
        profitable_taker=False,
    ) == "signal_negative_even_zero_cost"
    assert audit_rows[0]["was_gross_winner"] is True
    assert audit_rows[0]["was_net_winner"] is False
    assert audit_rows[0]["gross_winner_became_net_loser"] is True
    assert audit_rows[0]["fee_cost_pct"] == pytest.approx(0.005)
    assert audit_rows[0]["slippage_cost_pct"] == pytest.approx(0.002)
    assert audit_rows[0]["total_cost_pct"] == pytest.approx(0.007)
    diagnostics = aggregate_trade_diagnostics(audit_rows)
    assert diagnostics["total_trades"] == 1
    assert diagnostics["gross_winners"] == 1
    assert diagnostics["net_winners"] == 0
    assert diagnostics["trades_by_exit_reason"] == {"research_take_profit": 1}
    if before is not None:
        assert env_path.read_text(encoding="utf-8") == before


def test_volatility_focus_trade_diagnostics_reports_mfe_mae_and_exit_mix():
    config = _volatility_breakout_config(timeframe="1H", stop_loss_pct=0.01, max_hold_bars=12)
    trades = pd.DataFrame(
        [
            {
                "max_favorable_excursion_pct": 0.012,
                "max_adverse_excursion_pct": -0.004,
                "buy_exit_reason": "research_stop_loss",
                "buy_hold_bars": 2,
            },
            {
                "max_favorable_excursion_pct": 0.025,
                "max_adverse_excursion_pct": -0.002,
                "buy_exit_reason": "research_take_profit",
                "buy_hold_bars": 5,
            },
            {
                "max_favorable_excursion_pct": 0.004,
                "max_adverse_excursion_pct": -0.006,
                "buy_exit_reason": "research_max_hold",
                "buy_hold_bars": 12,
            },
        ]
    )

    diagnostics = volatility_focus_trade_diagnostics(
        trades,
        trades,
        {"number_of_trades": 3},
        config,
        walk_forward={
            "per_fold_net_return_pct": [0.01, -0.002],
            "per_fold_number_of_trades": [2, 1],
            "per_fold_profit_factor_net": [1.4, 0.8],
        },
        cost_summary={"cost_sensitivity_classification": "signal_survives_current_taker_cost"},
    )

    assert diagnostics["total_entries"] == 3
    assert diagnostics["total_exits"] == 3
    assert diagnostics["average_mfe"] == pytest.approx((0.012 + 0.025 + 0.004) / 3)
    assert diagnostics["median_mae"] == pytest.approx(-0.004)
    assert diagnostics["pct_trades_reaching_1r_before_stop"] == pytest.approx(2 / 3)
    assert diagnostics["pct_trades_reaching_2r_before_stop"] == pytest.approx(1 / 3)
    assert diagnostics["pct_trades_stopped_quickly"] == pytest.approx(1 / 3)
    assert diagnostics["pct_trades_timing_out"] == pytest.approx(1 / 3)
    assert diagnostics["fold_by_fold_trade_counts"] == [2, 1]


def test_trade_by_trade_audit_export_writes_csv_jsonl_and_diagnostics(tmp_path):
    settings = research_settings(_settings(order_notional_usd=25, min_training_rows=1))
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    bars = pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=15 * i) for i in range(80)],
            "open": [100 + i * 0.1 for i in range(80)],
            "high": [101 + i * 0.1 for i in range(80)],
            "low": [99 + i * 0.1 for i in range(80)],
            "close": [100 + i * 0.1 for i in range(80)],
            "volume": [10.0] * 80,
        }
    )
    row = {
        "parameter_set_id": "btd_00000",
        "strategy_name": BUY_THE_DIP_STRATEGY,
        "timeframe": "15Min",
        "number_of_trades": 1,
        "net_return_pct": -0.001,
        "profit_factor_net": 0.0,
        "max_drawdown_pct": 0.001,
        "rank_score": 1.0,
        "adjusted_rank_score": 1.0,
        "rejection_reasons": "unit_test_rejected",
    }

    outputs = export_trade_audit_logs(
        [row],
        {"15Min": bars},
        settings,
        data_source_reports={
            "15Min": ResearchDataReport(
                timeframe="15Min",
                source_used="collected_market_data",
                latest_timestamp=bars["timestamp"].iloc[-1].isoformat(),
                data_age_minutes=0.0,
                row_count=len(bars),
                synthetic_data_used=False,
                research_result_valid=True,
            )
        },
        strategy=BUY_THE_DIP_STRATEGY,
        max_buy_dip_configs=1,
        max_v3_configs=1,
        walk_forward_splits=4,
        min_trades_per_split=1,
        timeframes=("15Min",),
        output_dir=tmp_path / "audits",
        top_n=1,
        include_rejected=True,
    )

    assert len(outputs) == 1
    csv_path = Path(outputs[0]["csv_path"])
    jsonl_path = Path(outputs[0]["jsonl_path"])
    assert csv_path.exists()
    assert jsonl_path.exists()
    assert "strategy_name,parameter_set_id,timeframe,entry_timestamp" in csv_path.read_text(encoding="utf-8")
    assert "aggregate_diagnostics" in outputs[0]
    assert (tmp_path / "audits" / "trade_audit_diagnostics.json").exists()
    if jsonl_path.read_text(encoding="utf-8").strip():
        assert "gross_winner_became_net_loser" in jsonl_path.read_text(encoding="utf-8")


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
    assert "strategy_breakdown" in summary
    assert "concise_summary" in summary


def test_volatility_focus_summary_and_output_files_include_safety_and_rejections(tmp_path):
    settings = research_settings(_settings())
    row = {
        "parameter_set_id": "vff_vbo_00001",
        "track_id": VOLATILITY_FOCUS_TRACK_B,
        "strategy_name": VOLATILITY_BREAKOUT_STRATEGY,
        "timeframe": "1H",
        "exit_mode": EXIT_MODE_BREAK_EVEN_1R,
        "take_profit_pct": 0.04,
        "stop_loss_pct": 0.02,
        "max_hold_bars": 48,
        "breakout_lookback": 20,
        "consolidation_lookback": 12,
        "number_of_trades": 25,
        "net_return_pct": 0.02,
        "profit_factor_net": 1.2,
        "max_drawdown_pct": 0.004,
        "win_rate_net": 0.52,
        "expectancy": 0.001,
        "walk_forward_passed": True,
        "folds_with_min_trades_count": 3,
        "fold_by_fold_returns": [0.01, 0.005, 0.005, 0.0],
        "fold_by_fold_trade_counts": [7, 6, 6, 6],
        "per_fold_number_of_trades": [7, 6, 6, 6],
        "per_fold_net_return_pct": [0.01, 0.005, 0.005, 0.0],
        "per_fold_profit_factor_net": [1.2, 1.1, 1.1, 1.0],
        "cost_sensitivity_classification": "signal_survives_current_taker_cost",
        "net_return_by_cost_scenario": {
            "current_taker": 0.02,
            "maker_current": 0.025,
            "maker_low_slippage": 0.03,
            "zero_cost_sanity": 0.04,
        },
        "current_taker_net_return_pct": 0.02,
        "maker_current_net_return_pct": 0.025,
        "maker_low_slippage_net_return_pct": 0.03,
        "zero_cost_net_return_pct": 0.04,
        "current_taker_profit_factor": 1.2,
        "maker_current_profit_factor": 1.25,
        "maker_low_slippage_profit_factor": 1.3,
        "zero_cost_profit_factor": 1.4,
        "profit_factor_by_cost_scenario": {"current_taker": 1.2, "maker_current": 1.25},
        "research_promising": True,
        "economically_viable": True,
        "maker_research_promising": True,
        "maker_economically_viable": True,
        "maker_only_candidate": False,
        "paper_forward_eligible": False,
        "estimated_fill_rate_required_to_remain_profitable": 0.0,
        "maker_vs_taker_net_gap": 0.005,
        "max_allowed_taker_fallback_rate_before_net_negative": 1.0,
        "spread_bps_assumption": 5.0,
        "slippage_bps_assumption": 5.0,
        "no_market_fallback_required": True,
        "post_only_required": True,
        "unfilled_cancel_required": True,
        "research_rejection_reasons": "",
        "maker_rejection_reasons": "",
        "paper_forward_rejection_reasons": "active_model_invalid",
        "training_rejection_reasons": "active_model_invalid;training_deferred_volatility_focus_no_ml_yet",
        "adjusted_rank_score": 10.0,
    }
    reports = {
        "1H": ResearchDataReport(
            timeframe="1H",
            source_used="collected_market_data",
            latest_timestamp="2026-06-07T09:00:00+00:00",
            data_age_minutes=0.0,
            row_count=500,
            synthetic_data_used=False,
            research_result_valid=True,
        )
    }
    summary = build_volatility_focus_summary(
        [row],
        settings,
        base_summary={
            "synthetic_data_used": False,
            "research_result_valid": True,
            "timeframes": ["1H"],
            "baselines": {},
        },
        data_source_reports=reports,
        bars_by_timeframe={"1H": _bars(latest=datetime(2026, 6, 7, 9, 0, tzinfo=UTC), count=40, step_minutes=60)},
        min_focused_trades=20,
        target_focused_trades=50,
        max_focused_configs=100,
        focused_summary_path=tmp_path / "volatility_focus_summary.json",
        top_configs_csv_path=tmp_path / "volatility_focus_top_configs.csv",
        rejections_path=tmp_path / "volatility_focus_rejections.json",
        trade_audit_paths=[],
    )
    write_volatility_focus_outputs(
        [row],
        summary,
        summary_path=tmp_path / "volatility_focus_summary.json",
        top_configs_csv_path=tmp_path / "volatility_focus_top_configs.csv",
        rejections_path=tmp_path / "volatility_focus_rejections.json",
    )

    assert summary["orders_placed"] == 0
    assert summary["trading_enabled"] is False
    assert summary["auto_trade_enabled"] is False
    assert summary["synthetic_data_used"] is False
    assert summary["research_promising_count"] == 1
    assert summary["current_taker_research_promising_count"] == 1
    assert summary["maker_research_promising_count"] == 1
    assert summary["maker_economically_viable_count"] == 1
    assert summary["maker_only_candidate_count"] == 0
    assert summary["paper_forward_eligible_count"] == 0
    assert summary["candidate_b_best"]["parameter_set_id"] == "vff_vbo_00001"
    assert summary["best_maker_candidate"]["parameter_set_id"] == "vff_vbo_00001"
    assert summary["best_all_maker_research_gates_passed"]["parameter_set_id"] == "vff_vbo_00001"
    assert summary["best_20_plus_current_cost_positive"]["parameter_set_id"] == "vff_vbo_00001"
    assert summary["best_walk_forward_current_cost_positive"]["parameter_set_id"] == "vff_vbo_00001"
    assert summary["best_all_research_gates_passed"]["parameter_set_id"] == "vff_vbo_00001"
    assert summary["any_config_passed_all_research_gates"] is True
    assert summary["any_current_taker_config_passed_all_research_gates"] is True
    assert summary["any_maker_config_passed_maker_research_gates"] is True
    assert summary["recommendation"] == "candidate_found_keep_trading_disabled"
    assert (tmp_path / "volatility_focus_summary.json").exists()
    top_csv = (tmp_path / "volatility_focus_top_configs.csv").read_text(encoding="utf-8")
    assert "track_id,parameter_set_id,strategy_name,timeframe,exit_mode" in top_csv
    assert "maker_research_promising" in top_csv
    assert "estimated_fill_rate_required_to_remain_profitable" in top_csv
    rejections = (tmp_path / "volatility_focus_rejections.json").read_text(encoding="utf-8")
    assert "active_model_invalid" in rejections


def test_volatility_focus_v9_terminal_recommends_abandon_when_no_maker_gate_passes(tmp_path):
    settings = research_settings(_settings())
    row = {
        "parameter_set_id": "v9m_00042",
        "track_id": VOLATILITY_FOCUS_TRACK_M9,
        "strategy_name": VOLATILITY_BREAKOUT_STRATEGY,
        "timeframe": "1H",
        "exit_mode": "fixed_tp_sl_timeout",
        "take_profit_pct": 0.045,
        "stop_loss_pct": 0.022,
        "max_hold_bars": 48,
        "breakout_lookback": 20,
        "consolidation_lookback": 12,
        "min_body_vs_avg": 1.2,
        "min_recent_return_pct": 0.003,
        "min_trend_strength": 0.0,
        "min_volume_zscore": 0.25,
        "max_atr_expansion": 3.0,
        "number_of_trades": 21,
        "gross_return_pct": 0.11,
        "net_return_pct": -0.01,
        "profit_factor_net": 0.94,
        "max_drawdown_pct": 0.108,
        "win_rate_net": 0.52,
        "expectancy": 0.001,
        "fold_count": 4,
        "walk_forward_passed": True,
        "folds_profitable_count": 3,
        "folds_with_min_trades_count": 4,
        "worst_fold_net_return_pct": -0.02,
        "median_fold_net_return_pct": 0.01,
        "fold_by_fold_returns": [0.02, 0.01, -0.02, 0.015],
        "fold_by_fold_trade_counts": [5, 5, 6, 5],
        "per_fold_number_of_trades": [5, 5, 6, 5],
        "per_fold_net_return_pct": [0.02, 0.01, -0.02, 0.015],
        "per_fold_profit_factor_net": [1.4, 1.2, 0.8, 1.3],
        "net_return_by_cost_scenario": {
            "current_taker": -0.011,
            "maker_current": 0.03,
            "maker_low_slippage": 0.038,
            "zero_cost_sanity": 0.114,
        },
        "profit_factor_by_cost_scenario": {
            "current_taker": 0.94,
            "maker_current": 1.15,
            "maker_low_slippage": 1.2,
            "zero_cost_sanity": 1.76,
        },
        "current_taker_net_return_pct": -0.011,
        "maker_current_net_return_pct": 0.03,
        "maker_low_slippage_net_return_pct": 0.038,
        "zero_cost_net_return_pct": 0.114,
        "current_taker_profit_factor": 0.94,
        "maker_current_profit_factor": 1.15,
        "maker_low_slippage_profit_factor": 1.2,
        "zero_cost_profit_factor": 1.76,
        "single_trade_return_concentration": 0.35,
        "statistically_weak": False,
        "profit_factor_reliable": True,
        "beats_buy_hold_risk_adjusted": True,
        "beats_dca_daily_risk_adjusted": True,
        "research_promising": False,
        "economically_viable": False,
        "maker_research_promising": False,
        "maker_economically_viable": False,
        "maker_only_candidate": False,
        "paper_forward_eligible": False,
        "estimated_fill_rate_required_to_remain_profitable": 0.0,
        "maker_vs_taker_net_gap": 0.041,
        "max_allowed_taker_fallback_rate_before_net_negative": 0.73,
        "no_market_fallback_required": True,
        "post_only_required": True,
        "unfilled_cancel_required": True,
        "source_used": "collected_market_data",
        "synthetic_data_used": False,
        "research_result_valid": True,
        "research_rejection_reasons": "current_taker_net_return_not_positive;max_drawdown_above_configured_limit",
        "maker_rejection_reasons": "max_drawdown_above_configured_limit",
        "paper_forward_rejection_reasons": "current_taker_net_return_not_positive;max_drawdown_above_configured_limit",
        "training_rejection_reasons": "training_deferred_volatility_focus_no_ml_yet",
        "adjusted_rank_score": 10.0,
    }
    reports = {
        "1H": ResearchDataReport(
            timeframe="1H",
            source_used="collected_market_data",
            latest_timestamp="2026-06-07T09:00:00+00:00",
            data_age_minutes=0.0,
            row_count=500,
            synthetic_data_used=False,
            research_result_valid=True,
        )
    }

    summary = build_volatility_focus_summary(
        [row],
        settings,
        base_summary={
            "synthetic_data_used": False,
            "research_result_valid": True,
            "timeframes": ["1H"],
            "baselines": {},
        },
        data_source_reports=reports,
        bars_by_timeframe={"1H": _bars(latest=datetime(2026, 6, 7, 9, 0, tzinfo=UTC), count=40, step_minutes=60)},
        min_focused_trades=20,
        target_focused_trades=50,
        max_focused_configs=3000,
        focused_summary_path=tmp_path / "volatility_focus_v9_summary.json",
        top_configs_csv_path=tmp_path / "volatility_focus_v9_top_configs.csv",
        rejections_path=tmp_path / "volatility_focus_v9_rejections.json",
        trade_audit_paths=[],
    )

    assert summary["v9_track_configs"]["track_id"] == VOLATILITY_FOCUS_TRACK_M9
    assert summary["best_v9_maker_candidate"]["parameter_set_id"] == "v9m_00042"
    assert summary["best_v9_drawdown_reduced_candidate"] is None
    assert summary["best_all_maker_research_gates_passed"] is None
    assert summary["best_candidate_under_drawdown_limit"] is None
    assert summary["terminal_line_failed"] is True
    assert summary["terminal_recommendation"] == "abandon_1h_volatility_breakout_maker_only_and_switch_strategy_family"
    assert summary["exact_blockers"]["present"]["max_drawdown_above_configured_limit"] is True
    assert summary["exact_blockers"]["best_v9_maker_candidate_blockers"] == ["max_drawdown_above_configured_limit"]


def test_volatility_focus_gate_split_allows_research_candidate_without_strict_eligibility(tmp_path):
    settings = research_settings(
        _settings(
            max_backtest_drawdown_pct=0.01,
            max_research_drawdown_pct=0.10,
            max_paper_forward_drawdown_pct=0.10,
        )
    )
    config = _volatility_breakout_config(
        parameter_set_id="v9m_00021",
        track_id=VOLATILITY_FOCUS_TRACK_M9,
        timeframe="1H",
        take_profit_pct=0.045,
        stop_loss_pct=0.02,
        max_hold_bars=48,
        breakout_lookback=20,
        consolidation_lookback=12,
        min_body_vs_avg=1.2,
        min_recent_return_pct=0.003,
        min_trend_strength=0.0,
        max_atr_expansion=3.0,
        min_volume_zscore=0.25,
    )
    metrics = _passing_metrics(number_of_trades=21, max_drawdown_pct=0.0998975)
    cost_summary = {
        "net_return_by_cost_scenario": {
            "current_taker": -0.01,
            "maker_current": 0.04218,
            "maker_low_slippage": 0.05058,
            "zero_cost_sanity": 0.12639,
        },
        "profit_factor_by_cost_scenario": {
            "current_taker": 0.94,
            "maker_current": 1.2335,
            "maker_low_slippage": 1.2870,
            "zero_cost_sanity": 1.76,
        },
    }
    source_report = ResearchDataReport(
        timeframe="1H",
        source_used="collected_market_data_derived_from_15min",
        latest_timestamp="2026-06-07T09:00:00+00:00",
        data_age_minutes=0.0,
        row_count=3998,
        synthetic_data_used=False,
        research_result_valid=True,
    )
    baseline = {
        "beats_buy_hold_risk_adjusted": True,
        "beats_dca_daily_risk_adjusted": True,
    }
    walk_forward = {
        "fold_count": 4,
        "walk_forward_passed": True,
        "folds_profitable_count": 3,
        "folds_with_min_trades_count": 4,
        "per_fold_number_of_trades": [5, 5, 6, 5],
        "per_fold_net_return_pct": [0.02, 0.01, -0.02, 0.015],
        "per_fold_profit_factor_net": [1.4, 1.2, 0.8, 1.3],
        "worst_fold_net_return_pct": -0.02,
        "median_fold_net_return_pct": 0.0125,
    }

    strict_maker_gate = volatility_focus_maker_research_gate(
        metrics,
        settings,
        config,
        cost_summary=cost_summary,
        source_report=source_report,
        synthetic_data_used=False,
        research_result_valid=True,
        baseline_comparison=baseline,
        walk_forward=walk_forward,
        min_focused_trades=20,
    )

    assert settings.max_backtest_drawdown_pct == 0.01
    assert strict_maker_gate["maker_research_promising"] is False
    assert "max_drawdown_above_configured_limit" in strict_maker_gate["maker_rejection_reasons"]

    row = {
        "parameter_set_id": "v9m_00021",
        "track_id": VOLATILITY_FOCUS_TRACK_M9,
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
        "min_volume_zscore": 0.25,
        "max_atr_expansion": 3.0,
        "number_of_trades": 21,
        "gross_return_pct": 0.12639,
        "net_return_pct": -0.01,
        "profit_factor_net": 0.94,
        "max_drawdown_pct": 0.0998975,
        "win_rate_net": 0.52,
        "expectancy": 0.001,
        **walk_forward,
        "net_return_by_cost_scenario": cost_summary["net_return_by_cost_scenario"],
        "profit_factor_by_cost_scenario": cost_summary["profit_factor_by_cost_scenario"],
        "current_taker_net_return_pct": -0.01,
        "maker_current_net_return_pct": 0.04218,
        "maker_low_slippage_net_return_pct": 0.05058,
        "zero_cost_net_return_pct": 0.12639,
        "current_taker_profit_factor": 0.94,
        "maker_current_profit_factor": 1.2335,
        "maker_low_slippage_profit_factor": 1.2870,
        "zero_cost_profit_factor": 1.76,
        "single_trade_return_concentration": 0.35,
        "statistically_weak": False,
        "profit_factor_reliable": True,
        "beats_buy_hold_risk_adjusted": True,
        "beats_dca_daily_risk_adjusted": True,
        "research_promising": False,
        "economically_viable": False,
        "maker_research_promising": False,
        "maker_economically_viable": False,
        "maker_only_candidate": False,
        "paper_forward_eligible": False,
        "estimated_fill_rate_required_to_remain_profitable": 0.2,
        "maker_vs_taker_net_gap": 0.05218,
        "max_allowed_taker_fallback_rate_before_net_negative": 0.8,
        "no_market_fallback_required": True,
        "post_only_required": True,
        "unfilled_cancel_required": True,
        "source_used": "collected_market_data_derived_from_15min",
        "synthetic_data_used": False,
        "research_result_valid": True,
        "research_rejection_reasons": "current_taker_net_return_not_positive;max_drawdown_above_configured_limit",
        "maker_rejection_reasons": "max_drawdown_above_configured_limit",
        "paper_forward_rejection_reasons": "current_taker_net_return_not_positive;max_drawdown_above_configured_limit",
        "training_rejection_reasons": "training_deferred_volatility_focus_no_ml_yet",
        "adjusted_rank_score": 10.0,
    }

    summary = build_volatility_focus_summary(
        [row],
        settings,
        base_summary={
            "synthetic_data_used": False,
            "research_result_valid": True,
            "timeframes": ["1H"],
            "baselines": {},
        },
        data_source_reports={"1H": source_report},
        bars_by_timeframe={"1H": _bars(latest=datetime(2026, 6, 7, 9, 0, tzinfo=UTC), count=40, step_minutes=60)},
        min_focused_trades=20,
        target_focused_trades=50,
        max_focused_configs=3000,
        focused_summary_path=tmp_path / "volatility_focus_gate_split_summary.json",
        top_configs_csv_path=tmp_path / "volatility_focus_gate_split_top_configs.csv",
        rejections_path=tmp_path / "volatility_focus_gate_split_rejections.json",
        trade_audit_paths=[],
    )

    assert summary["max_backtest_drawdown_pct"] == 0.01
    assert summary["max_research_drawdown_pct"] == 0.10
    assert summary["max_paper_forward_drawdown_pct"] == 0.10
    assert summary["maker_only_research_candidate_count"] == 1
    assert summary["maker_only_paper_forward_candidate_count"] == 1
    assert summary["strict_paper_forward_eligible_count"] == 0
    assert summary["live_tradable_count"] == 0
    assert summary["best_maker_only_research_candidate"]["parameter_set_id"] == "v9m_00021"
    assert summary["best_maker_only_research_candidate"]["maker_only_research_candidate"] is True
    assert summary["best_maker_only_research_candidate"]["strict_paper_forward_eligible"] is False
    assert summary["best_maker_only_research_candidate"]["live_tradable"] is False
    assert summary["fallback_trading_allowed"] is False
    assert "market_taker_execution_not_safe_for_maker_only_candidate" in summary["why_not_live_tradable"]
    assert "post_only_maker_fill_simulation" in summary["next_required_validation"]
    assert "paper_forward_validation_with_no_market_fallback" in summary["next_required_validation"]
    assert summary["terminal_line_failed"] is False
    assert summary["terminal_recommendation"] == "maker_only_research_candidate_found_but_not_live_tradable"


def _buy_the_dip_config(**overrides) -> ResearchConfig:
    values = {
        "parameter_set_id": "btd_test",
        "strategy_name": BUY_THE_DIP_STRATEGY,
        "timeframe": "5Min",
        "take_profit_pct": 0.01,
        "stop_loss_pct": 0.006,
        "max_hold_bars": 12,
        "rsi_threshold": 35.0,
        "zscore_threshold": -1.5,
        "vwap_distance_threshold": -0.003,
        "drawdown_threshold": 0.005,
        "min_volume_zscore": 0.5,
        "reversal_confirmation_required": True,
        "higher_timeframe_regime_filter": False,
    }
    values.update(overrides)
    return ResearchConfig(**values)


def _oversold_bounce_bars() -> pd.DataFrame:
    latest = datetime(2026, 6, 7, 9, 30, tzinfo=UTC)
    timestamps = [latest - timedelta(minutes=5 * (89 - index)) for index in range(90)]
    closes = [100.0 + (index % 5) * 0.03 for index in range(60)]
    closes.extend([99.5, 99.0, 98.2, 97.4, 96.6, 95.8, 94.8, 94.2, 95.1, 95.8])
    closes.extend([96.5, 97.2, 97.8, 98.1, 98.4, 98.7, 99.0, 99.2, 99.4, 99.5])
    closes.extend([99.6 + (index % 3) * 0.02 for index in range(90 - len(closes))])
    rows = []
    previous_close = closes[0]
    for index, close in enumerate(closes):
        open_ = previous_close
        high = max(open_, close) * 1.001
        low = min(open_, close) * 0.999
        volume = 10.0
        if 62 <= index <= 69:
            volume = 30.0 + index
        if index == 68:
            open_ = 94.1
            close = 95.1
            high = 95.7
            low = 93.0
            volume = 120.0
        rows.append(
            {
                "timestamp": timestamps[index],
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        previous_close = close
    return pd.DataFrame(rows)


def test_buy_the_dip_mean_reversion_identifies_oversold_bounce_and_is_long_only():
    features = prepare_buy_the_dip_features(_oversold_bounce_bars())

    trades, signals = build_buy_the_dip_research_trades(features, _settings(), _buy_the_dip_config())

    assert not trades.empty
    assert not signals.empty
    assert set(trades["strategy_name"]) == {BUY_THE_DIP_STRATEGY}
    assert set(signals["strategy_name"]) == {BUY_THE_DIP_STRATEGY}
    assert set(trades["ml_sell_probability"]) == {0.0}
    assert "side" not in trades.columns
    assert "short" not in set(str(reason).lower() for reason in trades["entry_reason"])
    assert set(trades["entry_reason"]) == {"buy_the_dip_oversold_reversal_candidate"}


def test_buy_the_dip_v2_wider_profile_finds_more_fixture_signals_than_strict_v1():
    features = prepare_buy_the_dip_features(_oversold_bounce_bars())
    strict = _buy_the_dip_config(
        rsi_threshold=20.0,
        zscore_threshold=-2.5,
        vwap_distance_threshold=-0.010,
        drawdown_threshold=0.020,
        min_volume_zscore=2.0,
        reversal_confirmation_required=True,
        higher_timeframe_regime_filter=True,
    )
    wider = _buy_the_dip_config(
        rsi_threshold=40.0,
        zscore_threshold=-1.0,
        vwap_distance_threshold=-0.002,
        drawdown_threshold=0.003,
        min_volume_zscore=-0.5,
        reversal_confirmation_required=False,
        higher_timeframe_regime_filter=False,
    )

    strict_trades, _ = build_buy_the_dip_research_trades(features, _settings(), strict)
    wider_trades, _ = build_buy_the_dip_research_trades(features, _settings(), wider)

    assert len(wider_trades) > len(strict_trades)


def test_buy_the_dip_mean_reversion_is_deterministic_on_fixture_data():
    features = prepare_buy_the_dip_features(_oversold_bounce_bars())
    config = _buy_the_dip_config()

    first_trades, _ = build_buy_the_dip_research_trades(features, _settings(), config)
    second_trades, _ = build_buy_the_dip_research_trades(features, _settings(), config)

    columns = ["timestamp", "buy_exit_return_pct", "buy_exit_reason", "buy_hold_bars"]
    pd.testing.assert_frame_equal(
        first_trades[columns].reset_index(drop=True),
        second_trades[columns].reset_index(drop=True),
    )


def _uptrend_pullback_config(**overrides) -> ResearchConfig:
    values = {
        "parameter_set_id": "utp_test",
        "strategy_name": UPTREND_PULLBACK_STRATEGY,
        "timeframe": "15Min",
        "take_profit_pct": 0.02,
        "stop_loss_pct": 0.012,
        "max_hold_bars": 12,
        "support": "ema20",
        "support_distance_pct": 0.01,
        "pullback_min_pct": 0.01,
        "pullback_max_pct": 0.08,
        "rsi_min": 35.0,
        "rsi_max": 55.0,
        "confirmation": "bullish_close",
        "min_lower_wick_ratio": 0.20,
        "min_volume_recovery": -0.75,
        "max_atr_expansion": 2.5,
    }
    values.update(overrides)
    return ResearchConfig(**values)


def _volatility_breakout_config(**overrides) -> ResearchConfig:
    values = {
        "parameter_set_id": "vbo_test",
        "strategy_name": VOLATILITY_BREAKOUT_STRATEGY,
        "timeframe": "15Min",
        "take_profit_pct": 0.03,
        "stop_loss_pct": 0.015,
        "max_hold_bars": 12,
        "min_volume_zscore": 0.5,
        "breakout_lookback": 20,
        "consolidation_lookback": 12,
        "min_body_vs_avg": 1.0,
        "min_recent_return_pct": 0.002,
        "min_trend_strength": 0.0,
        "max_atr_expansion": 2.5,
    }
    values.update(overrides)
    return ResearchConfig(**values)


def _v3_feature_rows(*, strategy: str) -> pd.DataFrame:
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    rows = []
    for index in range(48):
        close = 100.0 + index * 0.02
        if strategy == VOLATILITY_BREAKOUT_STRATEGY:
            close = 105.0 + index * 0.03
        rows.append(
            {
                "timestamp": start + timedelta(minutes=15 * index),
                "open": close * 0.998,
                "high": close * 1.001,
                "low": close * 0.997,
                "close": close,
                "volume": 10.0 + index,
                "ema_20": close * 0.99,
                "ema_50": close * 0.985,
                "ema_20_slope_5": 0.001,
                "ema_50_slope_5": 0.001,
                "ema_50_above_200": True,
                "close_above_ema_200": True,
                "pullback_from_high_50": 0.03,
                "support_distance_ema20_abs": 0.004,
                "support_distance_ema50_abs": 0.008,
                "support_distance_vwap_abs": 0.005,
                "rsi_14": 45.0,
                "bullish_close": True,
                "recovers_prior_high": False,
                "lower_wick_ratio": 0.35,
                "close_position_in_candle": 0.65,
                "volume_zscore_20": 1.1,
                "atr_expansion_20": 1.1,
                "atr_downside_explosion": False,
                "extreme_crash_candle": False,
                "log_return_3": 0.004,
                "log_return_5": 0.006,
                "body_vs_avg_20": 1.4,
                "trend_strength_20": 0.5,
                "prior_rolling_high_20": close * 0.995,
                "range_width_12": 0.03,
            }
        )
    frame = pd.DataFrame(rows)
    frame.loc[14:16, "high"] = frame.loc[14:16, "close"] * 1.05
    return frame


def test_uptrend_pullback_produces_expected_long_only_entries_on_fixture_data():
    trades, signals = build_uptrend_pullback_research_trades(
        _v3_feature_rows(strategy=UPTREND_PULLBACK_STRATEGY),
        _settings(),
        _uptrend_pullback_config(),
    )

    assert not trades.empty
    assert not signals.empty
    assert set(trades["strategy_name"]) == {UPTREND_PULLBACK_STRATEGY}
    assert set(trades["entry_reason"]) == {"uptrend_pullback_support_reclaim_candidate"}
    assert set(trades["ml_sell_probability"]) == {0.0}
    assert "side" not in trades.columns


def test_volatility_breakout_produces_expected_long_only_entries_on_fixture_data():
    trades, signals = build_volatility_breakout_research_trades(
        _v3_feature_rows(strategy=VOLATILITY_BREAKOUT_STRATEGY),
        _settings(),
        _volatility_breakout_config(),
    )

    assert not trades.empty
    assert not signals.empty
    assert set(trades["strategy_name"]) == {VOLATILITY_BREAKOUT_STRATEGY}
    assert set(trades["entry_reason"]) == {"volatility_breakout_momentum_continuation_candidate"}
    assert set(trades["ml_sell_probability"]) == {0.0}
    assert "short" not in set(str(reason).lower() for reason in trades["entry_reason"])


def test_volatility_focus_v7_exit_modes_adjust_research_exits():
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    features = pd.DataFrame(
        [
            {"timestamp": start, "close": 100.0, "high": 100.0, "low": 100.0, "ema_20": 99.0, "log_return_3": 0.01, "ema_50_slope_5": 0.01},
            {"timestamp": start + timedelta(hours=1), "close": 103.0, "high": 104.0, "low": 102.5, "ema_20": 99.0, "log_return_3": 0.01, "ema_50_slope_5": 0.01},
            {"timestamp": start + timedelta(hours=2), "close": 101.0, "high": 101.5, "low": 99.5, "ema_20": 99.0, "log_return_3": 0.01, "ema_50_slope_5": 0.01},
            {"timestamp": start + timedelta(hours=3), "close": 100.5, "high": 101.0, "low": 100.0, "ema_20": 101.0, "log_return_3": -0.001, "ema_50_slope_5": -0.001},
            {"timestamp": start + timedelta(hours=4), "close": 100.0, "high": 100.5, "low": 99.8, "ema_20": 101.0, "log_return_3": -0.001, "ema_50_slope_5": -0.001},
        ]
    )

    break_even = resolve_research_exit(
        features,
        0,
        _volatility_breakout_config(take_profit_pct=0.10, stop_loss_pct=0.02, max_hold_bars=4, exit_mode=EXIT_MODE_BREAK_EVEN_1R),
    )
    trailing = resolve_research_exit(
        features,
        0,
        _volatility_breakout_config(take_profit_pct=0.10, stop_loss_pct=0.02, max_hold_bars=4, exit_mode=EXIT_MODE_TRAILING_1R),
    )
    mfe_protect = resolve_research_exit(
        features,
        0,
        _volatility_breakout_config(take_profit_pct=0.10, stop_loss_pct=0.02, max_hold_bars=4, exit_mode=EXIT_MODE_MFE_PROTECT_1R_50),
    )
    time_stop = resolve_research_exit(
        features,
        0,
        _volatility_breakout_config(take_profit_pct=0.10, stop_loss_pct=0.02, max_hold_bars=4, exit_mode=EXIT_MODE_TIME_STOP_MOMENTUM_WEAK),
    )

    assert break_even["exit_reason"] == "research_break_even_stop"
    assert break_even["gross_return"] == pytest.approx(0.0)
    assert trailing["exit_reason"] == "research_trailing_stop"
    assert trailing["gross_return"] == pytest.approx(0.0192)
    assert mfe_protect["exit_reason"] == "research_mfe_protection"
    assert mfe_protect["gross_return"] == pytest.approx(0.02)
    assert time_stop["exit_reason"] == "research_time_stop_momentum_weak"
    assert time_stop["gross_return"] == pytest.approx(0.005)


def test_volatility_focus_quality_filters_block_low_quality_breakouts():
    features = _v3_feature_rows(strategy=VOLATILITY_BREAKOUT_STRATEGY).copy()
    features["normalized_volume"] = 1.4
    features["breakout_candle_atr_multiple"] = 1.4
    features["recent_runup_pct_5"] = 0.025
    features["log_return_1"] = 0.01
    features["range_compression_12"] = 1.2
    features["atr_percentile_200"] = 0.50
    config = _volatility_breakout_config(
        require_ema_trend_filter=True,
        require_positive_ema20_slope=True,
        require_close_above_ema200=True,
        max_breakout_candle_atr_multiple=2.0,
        min_close_position_in_candle=0.60,
        max_recent_runup_pct=0.05,
        min_consolidation_compression=1.0,
        require_volume_expansion=True,
        max_atr_percentile=0.90,
    )

    trades, _ = build_volatility_breakout_research_trades(features, _settings(), config)
    bad_close_position = features.copy()
    bad_close_position["close_position_in_candle"] = 0.20
    filtered_trades, _ = build_volatility_breakout_research_trades(bad_close_position, _settings(), config)
    bad_trend = features.copy()
    bad_trend["ema_50_above_200"] = False
    trend_filtered, _ = build_volatility_breakout_research_trades(bad_trend, _settings(), config)

    assert not trades.empty
    assert filtered_trades.empty
    assert trend_filtered.empty


def test_1h_bars_are_derived_chronologically_from_15min_data_without_partial_future_hour():
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    rows = []
    for index in [5, 0, 1, 2, 3, 4, 6, 7, 8]:
        timestamp = start + timedelta(minutes=15 * index)
        price = 100.0 + index
        rows.append(
            {
                "timestamp": timestamp,
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price + 0.25,
                "volume": 1.0,
            }
        )

    derived = derive_1h_bars_from_15min(pd.DataFrame(rows))

    assert list(derived["timestamp"]) == [pd.Timestamp(start), pd.Timestamp(start + timedelta(hours=1))]
    assert derived.iloc[0]["open"] == 100.0
    assert derived.iloc[0]["close"] == 103.25
    assert derived.iloc[1]["open"] == 104.0
    assert derived.iloc[1]["close"] == 107.25
    assert len(derived) == 2


def test_4h_bars_are_derived_without_future_leakage_or_incomplete_candles():
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    rows = []
    for index in list(range(8)) + list(range(10, 12)):
        timestamp = start + timedelta(hours=index)
        price = 100.0 + index
        rows.append(
            {
                "timestamp": timestamp,
                "open": price,
                "high": price + 2,
                "low": price - 2,
                "close": price + 0.5,
                "volume": 1.0,
            }
        )

    derived = derive_4h_bars_from_lower_timeframe(pd.DataFrame(rows), source_timeframe="1H")

    assert list(derived["timestamp"]) == [pd.Timestamp(start), pd.Timestamp(start + timedelta(hours=4))]
    assert derived.iloc[0]["open"] == 100.0
    assert derived.iloc[0]["high"] == 105.0
    assert derived.iloc[0]["low"] == 98.0
    assert derived.iloc[0]["close"] == 103.5
    assert len(derived) == 2


def test_daily_bars_are_derived_without_future_leakage_or_incomplete_candles():
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    rows = []
    for index in list(range(24)) + list(range(24, 36)):
        timestamp = start + timedelta(hours=index)
        price = 100.0 + index
        rows.append(
            {
                "timestamp": timestamp,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.25,
                "volume": 2.0,
            }
        )

    derived = derive_1d_bars_from_lower_timeframe(pd.DataFrame(rows), source_timeframe="1H")

    assert list(derived["timestamp"]) == [pd.Timestamp(start)]
    assert derived.iloc[0]["open"] == 100.0
    assert derived.iloc[0]["high"] == 124.0
    assert derived.iloc[0]["low"] == 99.0
    assert derived.iloc[0]["close"] == 123.25
    assert derived.iloc[0]["volume"] == 48.0


def test_walk_forward_splits_are_chronological_and_not_random():
    bars = _bars(latest=datetime(2026, 6, 1, 10, 0, tzinfo=UTC), count=24, step_minutes=15)

    folds = chronological_walk_forward_splits(bars.iloc[::-1], splits=4)

    assert len(folds) == 4
    for left, right in zip(folds, folds[1:]):
        assert left["timestamp"].max() < right["timestamp"].min()


def test_one_fold_only_success_does_not_pass_walk_forward():
    summary = summarize_walk_forward_metrics(
        [
            {"net_return_pct": 0.03, "profit_factor_net": 2.0, "number_of_trades": 8, "max_drawdown_pct": 0.001},
            {"net_return_pct": -0.01, "profit_factor_net": 0.8, "number_of_trades": 8, "max_drawdown_pct": 0.005},
            {"net_return_pct": -0.01, "profit_factor_net": 0.8, "number_of_trades": 8, "max_drawdown_pct": 0.005},
            {"net_return_pct": -0.01, "profit_factor_net": 0.8, "number_of_trades": 8, "max_drawdown_pct": 0.005},
        ],
        min_trades_per_split=3,
    )

    assert summary["fold_count"] == 4
    assert summary["folds_profitable_count"] == 1
    assert summary["walk_forward_passed"] is False


def test_buy_the_dip_reports_take_profit_cost_safety_in_summary(tmp_path):
    settings = research_settings(_settings())
    summary = build_research_summary(
        [
            {
                "parameter_set_id": "btd_low_target",
                "strategy_name": BUY_THE_DIP_STRATEGY,
                "timeframe": "5Min",
                "take_profit_pct": 0.005,
                "round_trip_estimated_cost_pct": 0.0076,
                "promotion_required_return_pct": 0.0086,
                "take_profit_vs_cost_safe": False,
                "net_return_pct": 0.01,
                "profit_factor_net": 1.2,
                "max_drawdown_pct": 0.001,
                "number_of_trades": 25,
                "economically_viable": False,
                "paper_forward_eligible": False,
                "rejection_reasons": "take_profit_not_above_round_trip_cost",
                "rank_score": -100.0,
            }
        ],
        settings,
        data_source_reports={
            "5Min": ResearchDataReport(
                timeframe="5Min",
                source_used="collected_market_data",
                latest_timestamp="2026-06-07T09:30:00+00:00",
                data_age_minutes=0.0,
                row_count=1500,
                synthetic_data_used=False,
                research_result_valid=True,
            )
        },
        csv_path=tmp_path / "research.csv",
        summary_path=tmp_path / "research.json",
        active_model_status={"active_model_valid": True},
    )

    assert summary["strategy_breakdown"][BUY_THE_DIP_STRATEGY]["take_profit_vs_cost_safe_count"] == 0
    assert summary["buy_the_dip_configs_tested"] == 1
    assert summary["buy_the_dip_economically_viable_count"] == 0


def test_reality_summary_marks_weak_15min_families_rejected(tmp_path):
    settings = research_settings(_settings())
    rows = [
        {
            "parameter_set_id": "btd_rejected",
            "strategy_name": BUY_THE_DIP_STRATEGY,
            "timeframe": "15Min",
            "number_of_trades": 25,
            "net_return_pct": -0.01,
            "profit_factor_net": 0.8,
            "max_drawdown_pct": 0.02,
            "walk_forward_passed": False,
            "research_promising": False,
            "economically_viable": False,
            "paper_forward_eligible": False,
            "beats_any_relevant_baseline_risk_adjusted": False,
            "rejection_reasons": "net_return_not_positive;walk_forward_not_passed",
            "rank_score": -1.0,
            "adjusted_rank_score": -1.0,
        }
    ]
    bars = {
        "15Min": _bars(latest=datetime(2026, 6, 7, 9, 30, tzinfo=UTC), count=40, step_minutes=15)
    }
    reports = {
        "15Min": ResearchDataReport(
            timeframe="15Min",
            source_used="collected_market_data",
            latest_timestamp="2026-06-07T09:30:00+00:00",
            data_age_minutes=0.0,
            row_count=40,
            synthetic_data_used=False,
            research_result_valid=True,
        )
    }
    summary = build_research_summary(
        rows,
        settings,
        data_source_reports=reports,
        csv_path=tmp_path / "research.csv",
        summary_path=tmp_path / "research.json",
        active_model_status={"active_model_valid": True},
        requested_timeframes=("15Min",),
        audit_mode="reality",
        bars_by_timeframe=bars,
    )

    assert summary["fifteen_min_rejected"] is True
    assert summary["buy_the_dip_rejected"] is True
    assert "15Min buy_the_dip_mean_reversion" in summary["rejected_strategy_families"]
    assert "walk_forward_not_passed" in summary["fifteen_min_rejection_reason"]


def test_one_trade_config_is_statistically_weak_and_ranked_below_reliable_config(tmp_path):
    one_trade_rank = research_rank_details(
        {
            "number_of_trades": 1,
            "net_return_pct": 0.06,
            "profit_factor_net": float("inf"),
            "max_drawdown_pct": 0.0,
        },
        {"economically_viable": False, "paper_forward_eligible": False},
        concentration=1.0,
    )
    reliable_rank = research_rank_details(
        {
            "number_of_trades": 25,
            "net_return_pct": 0.015,
            "profit_factor_net": 1.2,
            "max_drawdown_pct": 0.004,
        },
        {"economically_viable": True, "paper_forward_eligible": False},
        concentration=0.20,
    )
    settings = research_settings(_settings())
    summary = build_research_summary(
        [
            {
                "parameter_set_id": "one_trade",
                "strategy_name": BUY_THE_DIP_STRATEGY,
                "timeframe": "5Min",
                "number_of_trades": 1,
                "net_return_pct": 0.06,
                "profit_factor_net": float("inf"),
                "max_drawdown_pct": 0.0,
                "single_trade_return_concentration": 1.0,
                "economically_viable": False,
                "paper_forward_eligible": False,
                "rejection_reasons": "number_of_trades_below_20;single_trade_return_concentration_too_high",
                "statistically_weak": one_trade_rank["statistically_weak"],
                "adjusted_rank_score": one_trade_rank["adjusted_rank_score"],
                "rank_score": one_trade_rank["raw_rank_score"],
            },
            {
                "parameter_set_id": "reliable",
                "strategy_name": BUY_THE_DIP_STRATEGY,
                "timeframe": "5Min",
                "number_of_trades": 25,
                "net_return_pct": 0.015,
                "profit_factor_net": 1.2,
                "max_drawdown_pct": 0.004,
                "single_trade_return_concentration": 0.20,
                "economically_viable": True,
                "paper_forward_eligible": False,
                "rejection_reasons": "active_model_invalid",
                "statistically_weak": reliable_rank["statistically_weak"],
                "adjusted_rank_score": reliable_rank["adjusted_rank_score"],
                "rank_score": reliable_rank["raw_rank_score"],
            },
        ],
        settings,
        data_source_reports={
            "5Min": ResearchDataReport(
                timeframe="5Min",
                source_used="collected_market_data",
                latest_timestamp="2026-06-07T09:30:00+00:00",
                data_age_minutes=0.0,
                row_count=1500,
                synthetic_data_used=False,
                research_result_valid=True,
            )
        },
        csv_path=tmp_path / "research.csv",
        summary_path=tmp_path / "research.json",
        active_model_status={"active_model_valid": False},
    )

    assert one_trade_rank["statistically_weak"] is True
    assert one_trade_rank["profit_factor_reliable"] is False
    assert one_trade_rank["adjusted_rank_score"] < reliable_rank["adjusted_rank_score"]
    assert summary["all_results"][0]["parameter_set_id"] == "reliable"
    assert summary["buy_the_dip_best_config_20_plus_trades"]["parameter_set_id"] == "reliable"
    assert summary["buy_the_dip_mean_reversion_trade_summary"]["configs_with_20_plus_trades"] == 1
    assert summary["buy_the_dip_profitable_20_plus_trade_configs"] == 1


def test_stale_market_data_client_data_is_rejected(tmp_path):
    now = datetime(2026, 6, 7, 9, 30, tzinfo=UTC)
    engine, Session = _session_factory(tmp_path)

    class StaleClient:
        async def fetch_bars(self, symbol, *, timeframe=None, limit=None, force_refresh=False):
            assert symbol == "BTC/USD"
            assert timeframe == "5Min"
            assert force_refresh is True
            return _bars(latest=now - timedelta(hours=1), count=limit or 3, step_minutes=5)

    try:
        result = asyncio.run(
            _fetch_research_bars(
                StaleClient(),
                _settings(min_training_rows=1),
                timeframe="5Min",
                limit=3,
                session_factory=Session,
                now=now,
            )
        )
    finally:
        engine.dispose()

    assert result.bars.empty
    assert result.report.source_used == "no_valid_real_data_source"
    assert result.report.research_result_valid is False
    assert result.report.synthetic_data_used is False
    market_source = next(source for source in result.report.rejected_sources if source["source"] == "market_data_client")
    assert market_source["status"] == "stale"
    assert "stale_latest_timestamp" in market_source["reason"]
    assert market_source["latest_timestamp"] == "2026-06-07T08:30:00+00:00"


def test_future_market_data_client_data_is_rejected(tmp_path):
    now = datetime(2026, 6, 7, 9, 30, tzinfo=UTC)
    engine, Session = _session_factory(tmp_path)

    class FutureClient:
        async def fetch_bars(self, symbol, *, timeframe=None, limit=None, force_refresh=False):
            assert symbol == "BTC/USD"
            return _bars(latest=now + timedelta(minutes=5), count=limit or 3, step_minutes=5)

    try:
        result = asyncio.run(
            _fetch_research_bars(
                FutureClient(),
                _settings(min_training_rows=1),
                timeframe="5Min",
                limit=3,
                session_factory=Session,
                now=now,
                end=now + timedelta(hours=1),
            )
        )
    finally:
        engine.dispose()

    assert result.bars.empty
    assert result.report.source_used == "no_valid_real_data_source"
    market_source = next(source for source in result.report.rejected_sources if source["source"] == "market_data_client")
    assert "row_count_below_required" in market_source["reason"]


def test_fresh_sqlite_collected_data_is_preferred(tmp_path):
    now = datetime(2026, 6, 7, 9, 30, tzinfo=UTC)
    engine, Session = _session_factory(tmp_path)
    _insert_collected_rows(Session, timeframe="15Min", latest=now - timedelta(minutes=15), count=3, step_minutes=15)

    class ShouldNotFetchClient:
        async def fetch_bars(self, *args, **kwargs):
            raise AssertionError("fresh collected_market_data should be used before fetching client bars")

    try:
        result = asyncio.run(
            _fetch_research_bars(
                ShouldNotFetchClient(),
                _settings(min_training_rows=1),
                timeframe="15Min",
                limit=3,
                session_factory=Session,
                now=now,
            )
        )
    finally:
        engine.dispose()

    assert len(result.bars) == 3
    assert result.report.source_used == "collected_market_data"
    assert result.report.latest_timestamp == "2026-06-07T09:15:00+00:00"
    assert result.report.data_age_minutes == 15.0
    assert result.report.row_count == 3
    assert result.report.synthetic_data_used is False
    assert result.report.research_result_valid is True


def test_synthetic_fallback_cannot_produce_paper_forward_eligible_configs(tmp_path):
    settings = research_settings(_settings())
    summary = build_research_summary(
        [
            {
                "parameter_set_id": "demo",
                "timeframe": "15Min",
                "paper_forward_eligible": True,
                "economically_viable": True,
                "rank_score": 1.0,
                "synthetic_data_used": True,
            }
        ],
        settings,
        data_source_reports={
            "15Min": ResearchDataReport(
                timeframe="15Min",
                source_used="synthetic_explicit_test_demo_mode",
                latest_timestamp="2026-06-07T09:30:00+00:00",
                data_age_minutes=0.0,
                row_count=1500,
                synthetic_data_used=True,
                research_result_valid=False,
                rejection_reason="synthetic_data_not_valid_for_research_decisions",
            )
        },
        csv_path=tmp_path / "research.csv",
        summary_path=tmp_path / "research.json",
        active_model_status={"active_model_valid": True},
    )

    assert summary["synthetic_data_used"] is True
    assert summary["research_result_valid"] is False
    assert summary["paper_forward_eligible_config_count"] == 0
    assert summary["economically_viable_config_count"] == 0
    assert summary["paper_forward_eligible_configs"] == []


def test_run_higher_timeframe_research_does_not_enable_trading(tmp_path):
    now = datetime(2026, 6, 7, 9, 30, tzinfo=UTC)
    engine, Session = _session_factory(tmp_path)
    _insert_collected_rows(Session, timeframe="5Min", latest=now - timedelta(minutes=5), count=40, step_minutes=5)
    _insert_collected_rows(Session, timeframe="15Min", latest=now - timedelta(minutes=15), count=40, step_minutes=15)

    class ShouldNotFetchClient:
        async def fetch_bars(self, *args, **kwargs):
            raise AssertionError("fresh collected_market_data should be used before fetching client bars")

    try:
        summary = asyncio.run(
            run_higher_timeframe_research(
                _settings(trading_enabled=True, auto_trade_enabled=True, allow_fallback_trading=True, min_training_rows=1),
                bar_limit=40,
                client=ShouldNotFetchClient(),
                output_dir=tmp_path,
                session_factory=Session,
                now=now,
                strategy=BUY_THE_DIP_STRATEGY,
                max_buy_dip_configs=8,
            )
        )
    finally:
        engine.dispose()

    assert summary["paper_trading_only"] is True
    assert summary["trading_enabled"] is False
    assert summary["auto_trade_enabled"] is False
    assert summary["fallback_trading_allowed"] is False
    assert summary["auto_apply_best_config"] is False


def test_run_volatility_focus_writes_summary_and_keeps_trading_disabled(tmp_path):
    now = datetime(2026, 6, 7, 9, 0, tzinfo=UTC)
    engine, Session = _session_factory(tmp_path)
    _insert_collected_rows(Session, timeframe="1H", latest=now - timedelta(hours=1), count=80, step_minutes=60)

    class ShouldNotFetchClient:
        async def fetch_bars(self, *args, **kwargs):
            raise AssertionError("fresh collected_market_data should be used before fetching client bars")

    focused_path = tmp_path / "volatility_focus_summary.json"
    try:
        summary = asyncio.run(
            run_higher_timeframe_research(
                _settings(trading_enabled=True, auto_trade_enabled=True, allow_fallback_trading=True, min_training_rows=1),
                bar_limit=80,
                client=ShouldNotFetchClient(),
                output_dir=tmp_path,
                session_factory=Session,
                now=now,
                strategy=VOLATILITY_FOCUS_STRATEGY,
                max_v3_configs=4,
                save_focused_summary=focused_path,
                export_focused_trades=True,
                trade_log_dir=tmp_path / "trade_audits",
                top_n_trade_configs=1,
            )
        )
    finally:
        engine.dispose()

    focused = summary["volatility_focus"]
    assert focused_path.exists()
    assert focused["timeframes_used"] == ["1H"]
    assert focused["orders_placed"] == 0
    assert focused["trading_enabled"] is False
    assert focused["auto_trade_enabled"] is False
    assert focused["synthetic_data_used"] is False
    assert summary["trading_enabled"] is False
    assert summary["auto_trade_enabled"] is False
    assert list((tmp_path / "trade_audits").glob("volatility_focus_top_*.csv"))
    assert list((tmp_path / "trade_audits").glob("volatility_focus_top_*.jsonl"))


def test_run_volatility_focus_uses_v7_output_stem_when_requested(tmp_path):
    now = datetime(2026, 6, 7, 9, 0, tzinfo=UTC)
    engine, Session = _session_factory(tmp_path)
    _insert_collected_rows(Session, timeframe="1H", latest=now - timedelta(hours=1), count=80, step_minutes=60)

    class ShouldNotFetchClient:
        async def fetch_bars(self, *args, **kwargs):
            raise AssertionError("fresh collected_market_data should be used before fetching client bars")

    try:
        summary = asyncio.run(
            run_higher_timeframe_research(
                _settings(min_training_rows=1),
                bar_limit=80,
                client=ShouldNotFetchClient(),
                output_dir=tmp_path,
                session_factory=Session,
                now=now,
                strategy=VOLATILITY_FOCUS_STRATEGY,
                max_v3_configs=2,
                save_focused_summary=tmp_path / "volatility_focus_v7_summary.json",
                export_focused_trades=True,
                trade_log_dir=tmp_path / "trade_audits",
                top_n_trade_configs=1,
            )
        )
    finally:
        engine.dispose()

    assert (tmp_path / "volatility_focus_v7_summary.json").exists()
    assert (tmp_path / "volatility_focus_v7_top_configs.csv").exists()
    assert (tmp_path / "volatility_focus_v7_rejections.json").exists()
    assert summary["volatility_focus_top_configs_csv_path"].endswith("volatility_focus_v7_top_configs.csv")
    assert summary["volatility_focus_rejections_path"].endswith("volatility_focus_v7_rejections.json")
    assert list((tmp_path / "trade_audits").glob("volatility_focus_v7_top_*.csv"))
    assert list((tmp_path / "trade_audits").glob("volatility_focus_v7_top_*.jsonl"))


def test_volatility_focus_derives_1h_from_15min_without_15min_results(tmp_path, monkeypatch):
    now = datetime(2026, 6, 7, 9, 0, tzinfo=UTC)
    engine, Session = _session_factory(tmp_path)
    _insert_collected_rows(Session, timeframe="15Min", latest=now - timedelta(minutes=15), count=32, step_minutes=15)
    requested_market_timeframes = []
    env_path = tmp_path / ".env"
    env_path.write_text(
        "TRADING_ENABLED=false\nAUTO_TRADE_ENABLED=false\nALLOW_FALLBACK_TRADING=false\n",
        encoding="utf-8",
    )
    env_before = env_path.read_text(encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    class EmptyClient:
        async def fetch_bars(self, symbol, *, timeframe=None, limit=None, force_refresh=False):
            requested_market_timeframes.append(timeframe)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    try:
        summary = asyncio.run(
            run_higher_timeframe_research(
                _settings(min_training_rows=1),
                bar_limit=8,
                client=EmptyClient(),
                output_dir=tmp_path,
                session_factory=Session,
                now=now,
                strategy=VOLATILITY_FOCUS_STRATEGY,
                timeframes=("1H",),
                max_v3_configs=6,
                save_focused_summary=tmp_path / "volatility_focus_summary.json",
            )
        )
    finally:
        engine.dispose()

    assert requested_market_timeframes == ["1Hour"]
    focused = summary["volatility_focus"]
    expected_source = "collected_market_data_derived_from_15min"
    assert env_path.read_text(encoding="utf-8") == env_before
    assert summary["timeframes"] == ["1H"]
    assert focused["timeframes_used"] == ["1H"]
    assert summary["source_used"] == {"1H": expected_source}
    assert focused["source_used_by_timeframe"] == {"1H": expected_source}
    assert summary["research_result_valid"] is True
    assert focused["research_result_valid"] is True
    assert summary["row_count"]["1H"] > 0
    assert focused["row_count"]["1H"] > 0
    assert "15Min" not in summary["row_count"]
    assert "15Min" not in focused["row_count"]
    assert summary["timeframe_data"]["1H"]["derived_from_timeframe"] == "15Min"
    assert any(
        source["timeframe"] == "15Min" and source["status"] == "used_for_1H_derivation"
        for source in summary["timeframe_data"]["1H"]["rejected_sources"]
    )
    assert {row["timeframe"] for row in summary["all_results"]} == {"1H"}
    assert {row["timeframe"] for row in focused["top_configs"]} == {"1H"}
    assert all(
        "data_source_not_collected_market_data" not in row["research_rejection_reasons"].split(";")
        for row in summary["all_results"]
    )
    assert all(
        "research_data_source_invalid" not in row["research_rejection_reasons"].split(";")
        for row in summary["all_results"]
    )
    assert focused["synthetic_data_used"] is False
    assert summary["synthetic_data_used"] is False
    assert focused["orders_placed"] == 0
    assert focused["trading_enabled"] is False
    assert focused["auto_trade_enabled"] is False
    assert focused["fallback_trading_allowed"] is False
    assert summary["trading_enabled"] is False
    assert summary["auto_trade_enabled"] is False
    assert summary["fallback_trading_allowed"] is False


def test_default_research_row_limit_remains_backward_compatible(tmp_path):
    now = datetime(2026, 6, 7, 9, 30, tzinfo=UTC)
    engine, Session = _session_factory(tmp_path)
    _insert_collected_rows(Session, timeframe="5Min", latest=now - timedelta(minutes=5), count=80, step_minutes=5)
    _insert_collected_rows(Session, timeframe="15Min", latest=now - timedelta(minutes=15), count=80, step_minutes=15)

    class ShouldNotFetchClient:
        async def fetch_bars(self, *args, **kwargs):
            raise AssertionError("fresh collected_market_data should be used before fetching client bars")

    try:
        summary = asyncio.run(
            run_higher_timeframe_research(
                _settings(min_training_rows=1),
                bar_limit=40,
                client=ShouldNotFetchClient(),
                output_dir=tmp_path,
                session_factory=Session,
                now=now,
                strategy=BUY_THE_DIP_STRATEGY,
                max_buy_dip_configs=8,
            )
        )
    finally:
        engine.dispose()

    assert summary["requested_max_rows_by_timeframe"] == {"5Min": 40, "15Min": 40}
    assert summary["used_rows_by_timeframe"] == {"5Min": 40, "15Min": 40}


def test_larger_research_row_window_is_used_when_requested(tmp_path):
    now = datetime(2026, 6, 7, 9, 30, tzinfo=UTC)
    engine, Session = _session_factory(tmp_path)
    _insert_collected_rows(Session, timeframe="5Min", latest=now - timedelta(minutes=5), count=120, step_minutes=5)
    _insert_collected_rows(Session, timeframe="15Min", latest=now - timedelta(minutes=15), count=90, step_minutes=15)

    class ShouldNotFetchClient:
        async def fetch_bars(self, *args, **kwargs):
            raise AssertionError("fresh collected_market_data should be used before fetching client bars")

    try:
        summary = asyncio.run(
            run_higher_timeframe_research(
                _settings(min_training_rows=1),
                bar_limit=40,
                max_rows_by_timeframe={"5Min": 80, "15Min": 60},
                client=ShouldNotFetchClient(),
                output_dir=tmp_path,
                session_factory=Session,
                now=now,
                strategy=BUY_THE_DIP_STRATEGY,
                max_buy_dip_configs=8,
            )
        )
    finally:
        engine.dispose()

    assert summary["available_rows_by_timeframe"] == {"5Min": 120, "15Min": 90}
    assert summary["requested_max_rows_by_timeframe"] == {"5Min": 80, "15Min": 60}
    assert summary["actual_used_rows_by_timeframe"] == {"5Min": 80, "15Min": 60}


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
