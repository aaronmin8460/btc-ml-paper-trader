from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db.database import Base, connect_args_for_database_url
from app.db.models import Order, Trade
from app.db.repository import Repository


def test_sqlite_engine_creation_uses_sqlite_connect_args(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'trading.db'}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))

    try:
        with engine.connect() as connection:
            assert connection.execute(text("select 1")).scalar_one() == 1
    finally:
        engine.dispose()


def test_non_sqlite_database_url_uses_empty_connect_args():
    database_url = "postgresql+psycopg://user:password@localhost:5432/trading"

    assert connect_args_for_database_url(database_url) == {}


def test_repository_builds_trade_frequency_state(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'trading.db'}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    now = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)

    try:
        with Session() as db:
            db.add_all(
                [
                    Order(symbol="BTC/USD", side="buy", status="submitted", created_at=now - timedelta(minutes=10)),
                    Order(symbol="BTC/USD", side="sell", status="submitted", created_at=now - timedelta(hours=2)),
                    Order(symbol="BTC/USD", side="buy", status="submitted", created_at=now - timedelta(days=1)),
                    Trade(symbol="BTC/USD", side="sell", qty=0.01, price=65000, pnl=-2, created_at=now),
                    Trade(symbol="BTC/USD", side="sell", qty=0.01, price=65000, pnl=-1, created_at=now - timedelta(minutes=5)),
                    Trade(symbol="BTC/USD", side="sell", qty=0.01, price=65000, pnl=3, created_at=now - timedelta(minutes=10)),
                ]
            )
            db.commit()

            state = Repository(db).trade_frequency_state(now=now)

        assert state.trades_last_hour == 1
        assert state.trades_today == 2
        assert state.consecutive_losses == 2
        assert state.last_trade_at == now - timedelta(minutes=10)
    finally:
        engine.dispose()
