import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.database import Base, connect_args_for_database_url, run_sqlite_schema_migrations
from app.db.models import CollectedMarketData, Order
from scripts import auto_research_train as art
from scripts import backfill_market_data as bmd


NOW = datetime(2026, 6, 9, 12, 34, 30, tzinfo=UTC)


class FakeHistoricalProvider:
    def __init__(self, frames: dict[str, pd.DataFrame] | pd.DataFrame) -> None:
        self.frames = frames
        self.calls: list[dict[str, object]] = []

    async def fetch_bars(self, symbol, *, timeframe, start, end, limit_per_request):
        self.calls.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "limit_per_request": limit_per_request,
            }
        )
        if isinstance(self.frames, dict):
            return self.frames.get(timeframe, pd.DataFrame())
        return self.frames.copy()


def _safe_env(tmp_path: Path, **overrides) -> dict[str, str]:
    env = {
        "PAPER_TRADING_ONLY": "true",
        "TRADING_ENABLED": "false",
        "AUTO_TRADE_ENABLED": "false",
        "ALLOW_FALLBACK_TRADING": "false",
        "SYMBOL": "BTC/USD",
        "MAX_OPEN_POSITIONS": "1",
        "ALPACA_PAPER_BASE_URL": "https://paper-api.alpaca.markets",
        "PAPER_EXECUTION_MODE": "alpaca_paper",
        "LOG_DIR": str(tmp_path / "logs"),
    }
    env.update(overrides)
    return env


def _valid_frame(*, timeframe: str = "1Min", count: int = 2, latest: datetime | None = None) -> pd.DataFrame:
    step = {"1Min": 1, "5Min": 5, "15Min": 15}[timeframe]
    latest = latest or NOW - timedelta(minutes=step * 3)
    timestamps = [latest - timedelta(minutes=step * (count - 1 - index)) for index in range(count)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + index for index in range(count)],
            "high": [101.0 + index for index in range(count)],
            "low": [99.0 + index for index in range(count)],
            "close": [100.5 + index for index in range(count)],
            "volume": [1.0 + index for index in range(count)],
        }
    )


def _session_factory(database_path: Path):
    database_url = f"sqlite:///{database_path}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))
    Base.metadata.create_all(bind=engine)
    run_sqlite_schema_migrations(engine)
    return engine, sessionmaker(bind=engine)


def _run_backfill(tmp_path: Path, provider: FakeHistoricalProvider, **overrides):
    kwargs = {
        "symbol": "BTC/USD",
        "timeframes": ["1Min"],
        "start": NOW - timedelta(minutes=10),
        "end": NOW,
        "database": tmp_path / "trading.db",
        "env": _safe_env(tmp_path),
        "env_path": tmp_path / "missing.env",
        "provider": provider,
        "now": NOW,
        "write_reports": True,
    }
    kwargs.update(overrides)
    return asyncio.run(bmd.run_backfill(**kwargs))


def test_dry_run_inserts_nothing_and_reports_no_orders(tmp_path):
    engine, Session = _session_factory(tmp_path / "trading.db")
    provider = FakeHistoricalProvider(_valid_frame())

    try:
        report, exit_code = _run_backfill(tmp_path, provider, dry_run=True)
        with Session() as db:
            row_count = db.query(CollectedMarketData).count()
            order_count = db.query(Order).count()
    finally:
        engine.dispose()

    summary = report["per_timeframe_summary"]["1Min"]
    assert exit_code == 0
    assert row_count == 0
    assert order_count == 0
    assert summary["inserted_rows"] == 0
    assert summary["would_insert_rows"] == 2
    assert report["synthetic_data_used"] is False
    assert report["orders_placed"] == 0


def test_run_inserts_valid_rows_and_records_source_metadata(tmp_path):
    provider = FakeHistoricalProvider(_valid_frame())

    report, exit_code = _run_backfill(tmp_path, provider, dry_run=False)

    engine, Session = _session_factory(tmp_path / "trading.db")
    try:
        with Session() as db:
            rows = db.query(CollectedMarketData).order_by(CollectedMarketData.timestamp).all()
            order_count = db.query(Order).count()
    finally:
        engine.dispose()

    assert exit_code == 0
    assert report["total_inserted_rows"] == 2
    assert len(rows) == 2
    assert order_count == 0
    assert rows[0].symbol == "BTC/USD"
    assert rows[0].source == "alpaca"
    assert rows[0].source_used == "alpaca_historical_backfill"
    assert rows[0].backfilled is True
    assert json.loads(rows[0].provider_metadata)["synthetic_data_used"] is False


def test_duplicate_backfill_is_idempotent_and_creates_no_duplicate_keys(tmp_path):
    provider = FakeHistoricalProvider(_valid_frame())

    first_report, first_code = _run_backfill(tmp_path, provider, dry_run=False)
    second_report, second_code = _run_backfill(tmp_path, provider, dry_run=False)

    engine, Session = _session_factory(tmp_path / "trading.db")
    try:
        with Session() as db:
            total_rows = db.query(CollectedMarketData).count()
            duplicate_groups = (
                db.query(
                    CollectedMarketData.symbol,
                    CollectedMarketData.timeframe,
                    CollectedMarketData.timestamp,
                    func.count(CollectedMarketData.id),
                )
                .group_by(
                    CollectedMarketData.symbol,
                    CollectedMarketData.timeframe,
                    CollectedMarketData.timestamp,
                )
                .having(func.count(CollectedMarketData.id) > 1)
                .count()
            )
    finally:
        engine.dispose()

    assert first_code == 0
    assert second_code == 0
    assert first_report["total_inserted_rows"] == 2
    assert second_report["per_timeframe_summary"]["1Min"]["inserted_rows"] == 0
    assert second_report["per_timeframe_summary"]["1Min"]["skipped_duplicate_rows"] == 2
    assert total_rows == 2
    assert duplicate_groups == 0


def test_only_btc_usd_is_allowed_and_non_btc_usd_fails(tmp_path):
    provider = FakeHistoricalProvider(_valid_frame())

    report, exit_code = _run_backfill(tmp_path, provider, symbol="ETH/USD", dry_run=False)

    assert exit_code == 1
    assert provider.calls == []
    assert "symbol_argument_not_btc_usd" in report["safety_flags"]["fatal_reasons"]


def test_future_incomplete_and_invalid_ohlcv_rows_are_rejected(tmp_path):
    invalid_frame = pd.DataFrame(
        {
            "timestamp": [
                NOW + timedelta(minutes=1),
                NOW.replace(second=0, microsecond=0),
                NOW - timedelta(minutes=5),
                NOW - timedelta(minutes=6),
                NOW - timedelta(minutes=7),
            ],
            "open": [100.0, 100.0, 100.0, 0.0, 100.0],
            "high": [101.0, 101.0, 100.2, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.5, 100.5, 100.8, 100.5, None],
            "volume": [1.0, 1.0, 1.0, -1.0, 1.0],
        }
    )
    provider = FakeHistoricalProvider(invalid_frame)

    report, exit_code = _run_backfill(tmp_path, provider, dry_run=False)

    summary = report["per_timeframe_summary"]["1Min"]
    assert exit_code == 1
    assert summary["inserted_rows"] == 0
    assert summary["invalid_rows"] == 5
    assert summary["validation_errors"]["timestamp_in_future"] == 1
    assert summary["validation_errors"]["candle_incomplete"] == 2
    assert summary["validation_errors"]["high_below_open_or_close"] == 1
    assert summary["validation_errors"]["ohlc_non_positive"] == 1
    assert summary["validation_errors"]["volume_negative"] == 1
    assert summary["validation_errors"]["close_invalid"] == 1
    assert report["synthetic_data_used"] is False
    assert report["orders_placed"] == 0


def test_backfill_never_modifies_env_file_and_trading_flags_remain_disabled(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "PAPER_TRADING_ONLY=true",
                "TRADING_ENABLED=false",
                "AUTO_TRADE_ENABLED=false",
                "ALLOW_FALLBACK_TRADING=false",
                "SYMBOL=BTC/USD",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    before = env_file.read_text(encoding="utf-8")
    provider = FakeHistoricalProvider(_valid_frame())

    report, exit_code = _run_backfill(
        tmp_path,
        provider,
        dry_run=False,
        env=_safe_env(tmp_path),
        env_path=env_file,
    )

    assert exit_code == 0
    assert env_file.read_text(encoding="utf-8") == before
    assert report["safety_flags"]["trading_enabled"] is False
    assert report["safety_flags"]["auto_trade_enabled"] is False
    assert report["trading_remained_disabled"] is True


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"PAPER_TRADING_ONLY": "false"}, "paper_trading_only_not_true"),
        ({"ALLOW_FALLBACK_TRADING": "true"}, "fallback_trading_enabled"),
        ({"SHORT_SELLING_ENABLED": "true"}, "short_selling_setting_detected"),
        ({"MARGIN_ENABLED": "true"}, "margin_setting_detected"),
        ({"MULTI_SYMBOL_ENABLED": "true"}, "multi_symbol_setting_detected"),
        ({"TRADING_ENABLED": "true"}, "trading_enabled_true"),
        ({"AUTO_TRADE_ENABLED": "true"}, "auto_trade_enabled_true"),
    ],
)
def test_unsafe_environment_flags_fail_before_provider_fetch(tmp_path, override, reason):
    provider = FakeHistoricalProvider(_valid_frame())

    report, exit_code = _run_backfill(
        tmp_path,
        provider,
        dry_run=False,
        env=_safe_env(tmp_path, **override),
    )

    assert exit_code == 1
    assert provider.calls == []
    assert reason in report["safety_flags"]["fatal_reasons"]


def test_existing_realtime_row_is_preserved_not_overwritten(tmp_path):
    engine, Session = _session_factory(tmp_path / "trading.db")
    timestamp = NOW - timedelta(minutes=5)
    try:
        with Session() as db:
            db.add(
                CollectedMarketData(
                    symbol="BTC/USD",
                    timeframe="1Min",
                    timestamp=timestamp,
                    open=90.0,
                    high=91.0,
                    low=89.0,
                    close=90.5,
                    volume=5.0,
                    source="live_collector",
                    source_used="market_data_client",
                    backfilled=False,
                    provider_metadata='{"origin":"collector"}',
                    collected_at=timestamp,
                )
            )
            db.commit()
    finally:
        engine.dispose()
    provider = FakeHistoricalProvider(
        pd.DataFrame(
            {
                "timestamp": [timestamp],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1.0],
            }
        )
    )

    report, exit_code = _run_backfill(tmp_path, provider, dry_run=False)

    engine, Session = _session_factory(tmp_path / "trading.db")
    try:
        with Session() as db:
            rows = db.query(CollectedMarketData).all()
    finally:
        engine.dispose()

    assert exit_code == 0
    assert report["per_timeframe_summary"]["1Min"]["skipped_duplicate_rows"] == 1
    assert len(rows) == 1
    assert rows[0].source == "live_collector"
    assert rows[0].close == 90.5


def test_auto_research_train_can_read_backfilled_rows(tmp_path):
    provider = FakeHistoricalProvider({"15Min": _valid_frame(timeframe="15Min", count=4)})

    report, exit_code = _run_backfill(
        tmp_path,
        provider,
        dry_run=False,
        timeframes=["15Min"],
        start=NOW - timedelta(hours=2),
        end=NOW,
    )

    engine, Session = _session_factory(tmp_path / "trading.db")
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path / 'trading.db'}", timeframe="15Min")
    try:
        bars = art.load_collected_bars(settings, timeframe="15Min", limit=10, session_factory=Session)
        readiness = art.build_data_readiness_by_timeframe(
            settings,
            effective_env=art.load_effective_env(
                env=_safe_env(
                    tmp_path,
                    AUTO_RESEARCH_MIN_1MIN_ROWS="1",
                    AUTO_RESEARCH_MIN_5MIN_ROWS="1",
                    AUTO_RESEARCH_MIN_15MIN_ROWS="4",
                    AUTO_RESEARCH_MAX_15MIN_AGE_MINUTES="60",
                ),
                env_path=tmp_path / "missing.env",
            ),
            session_factory=Session,
            now=NOW,
        )
    finally:
        engine.dispose()

    assert exit_code == 0
    assert report["total_inserted_rows"] == 4
    assert len(bars) == 4
    assert readiness["15Min"]["row_count"] == 4
    assert readiness["15Min"]["ready_for_training"] is True


def test_sqlite_backfill_metadata_migration_preserves_legacy_rows(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'legacy.db'}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE collected_market_data ("
                    "id INTEGER PRIMARY KEY, collected_at DATETIME, symbol VARCHAR(16), "
                    "timeframe VARCHAR(16), timestamp DATETIME, open FLOAT, high FLOAT, "
                    "low FLOAT, close FLOAT, volume FLOAT)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO collected_market_data "
                    "(collected_at, symbol, timeframe, timestamp, open, high, low, close, volume) "
                    "VALUES (:collected_at, 'BTC/USD', '1Min', :timestamp, 1, 2, 1, 2, 3)"
                ),
                {"collected_at": NOW, "timestamp": NOW - timedelta(minutes=10)},
            )

        run_sqlite_schema_migrations(engine)

        columns = {column["name"] for column in inspect_engine_columns(engine, "collected_market_data")}
        with engine.connect() as connection:
            row_count = connection.execute(text("SELECT COUNT(*) FROM collected_market_data")).scalar_one()
    finally:
        engine.dispose()

    assert {"source", "source_used", "backfilled", "provider_metadata"} <= columns
    assert row_count == 1


def inspect_engine_columns(engine, table_name):
    from sqlalchemy import inspect

    return inspect(engine).get_columns(table_name)
