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
    ResearchDataReport,
    build_research_summary,
    _fetch_research_bars,
    generate_research_configs,
    paper_forward_readiness_gate,
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
