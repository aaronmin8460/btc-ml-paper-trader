import asyncio
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.database import Base, connect_args_for_database_url
from app.db.models import CollectedMarketData, Order
from scripts.collect_market_data import collect_market_data


class FakeMarketDataClient:
    def __init__(self) -> None:
        self.order_calls = 0

    async def fetch_bars(self, symbol, *, timeframe=None, limit=None, force_refresh=False):
        assert symbol == "BTC/USD"
        assert force_refresh is True
        return pd.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
                    datetime(2026, 6, 1, 12, 5, tzinfo=UTC),
                ],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1.0, 2.0],
            }
        )

    async def fetch_latest_quote(self, symbol, *, force_refresh=False):
        assert symbol == "BTC/USD"
        return {"bid_price": 101.9, "ask_price": 102.1, "bid_size": 2.0, "ask_size": 1.0}

    async def submit_order(self, *args, **kwargs):
        self.order_calls += 1
        raise AssertionError("collector must never place orders")


def _session_factory(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'collector.db'}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))
    Base.metadata.create_all(bind=engine)
    return engine, sessionmaker(bind=engine)


def test_collector_works_with_trading_disabled_and_places_no_orders(tmp_path):
    engine, Session = _session_factory(tmp_path)
    client = FakeMarketDataClient()
    settings = Settings(_env_file=None, trading_enabled=False, auto_trade_enabled=False)

    try:
        report = asyncio.run(
            collect_market_data(
                settings,
                timeframes=["5Min"],
                limit=2,
                client=client,
                session_factory=Session,
            )
        )
        with Session() as db:
            rows = db.query(CollectedMarketData).order_by(CollectedMarketData.timestamp).all()
            order_count = db.query(Order).count()

        assert report["trading_enabled_required"] is False
        assert report["auto_trade_enabled_required"] is False
        assert report["orders_placed"] == 0
        assert client.order_calls == 0
        assert order_count == 0
        assert len(rows) == 2
        assert rows[0].symbol == "BTC/USD"
        assert rows[0].timeframe == "5Min"
        assert rows[0].bid is None
        assert rows[1].bid == 101.9
        assert rows[1].ask == 102.1
        assert rows[1].spread_bps > 0
        assert rows[1].quote_imbalance > 0
    finally:
        engine.dispose()
