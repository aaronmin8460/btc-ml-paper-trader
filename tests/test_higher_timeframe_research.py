import asyncio
from datetime import UTC, datetime, timedelta

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
    ResearchDataReport,
    ResearchConfig,
    build_buy_the_dip_research_trades,
    build_research_summary,
    _fetch_research_bars,
    generate_buy_the_dip_configs,
    generate_buy_the_dip_signal_profiles,
    generate_research_configs,
    generate_trend_pullback_configs,
    paper_forward_readiness_gate,
    prepare_buy_the_dip_features,
    research_rank_details,
    research_settings,
    run_higher_timeframe_research,
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


def test_research_config_space_matches_requested_values():
    configs = generate_trend_pullback_configs()

    assert {config.timeframe for config in configs} == {"5Min", "15Min"}
    assert {config.take_profit_pct for config in configs} == {0.008, 0.01, 0.015, 0.02}
    assert {config.stop_loss_pct for config in configs} == {0.003, 0.005, 0.008}
    assert {config.max_hold_bars for config in configs} == {6, 12, 24, 48}
    all_configs = generate_research_configs()
    buy_the_dip_configs = generate_buy_the_dip_configs()
    assert len(all_configs) == len(configs) + len(buy_the_dip_configs)
    assert {config.strategy_name for config in configs} == {"trend_pullback"}
    assert {config.strategy_name for config in buy_the_dip_configs} == {BUY_THE_DIP_STRATEGY}
    assert {config.take_profit_pct for config in buy_the_dip_configs} == {0.0086, 0.01, 0.0125, 0.015, 0.02, 0.025}
    assert len(generate_buy_the_dip_signal_profiles()) > len(BUY_THE_DIP_SIGNAL_PROFILES)
    assert any(config.timeframe == "15Min" for config in generate_buy_the_dip_configs(max_configs=120))


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
            )
        )
    finally:
        engine.dispose()

    assert summary["paper_trading_only"] is True
    assert summary["trading_enabled"] is False
    assert summary["auto_trade_enabled"] is False
    assert summary["fallback_trading_allowed"] is False
    assert summary["auto_apply_best_config"] is False


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
