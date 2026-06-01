import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.database import Base, connect_args_for_database_url
from app.db.models import Order, Trade
from app.db.repository import Repository
from app.risk.risk_manager import PositionState
from app.strategy.decision_engine import DecisionEngine


def _ioc_limit_raw() -> str:
    return json.dumps({"order_type": "limit", "time_in_force": "ioc"})


def _scalping_feature_row() -> dict:
    return {
        "close": 65000.0,
        "orderbook_spread": 0.0001,
        "quote_imbalance": 0.2,
        "volatility_20": 0.0,
        "high_low_range_pct": 0.0,
        "ema_slow_distance": 0.0,
        "sma_20_distance": -0.002,
        "log_return_3": 0.0,
        "log_return_5": -0.0015,
    }


def _session_factory(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'trading.db'}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))
    Base.metadata.create_all(bind=engine)
    return engine, sessionmaker(bind=engine)


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


def test_init_db_adds_dashboard_observability_columns_to_existing_sqlite_database(tmp_path, monkeypatch):
    from app.db import database

    database_url = f"sqlite:///{tmp_path / 'legacy.db'}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE signals ("
                    "id INTEGER PRIMARY KEY, created_at DATETIME, symbol VARCHAR(16), action VARCHAR(16), "
                    "buy_probability FLOAT, sell_probability FLOAT, reason TEXT)"
                )
            )
            connection.execute(
                text(
                    "CREATE TABLE trades ("
                    "id INTEGER PRIMARY KEY, created_at DATETIME, symbol VARCHAR(16), side VARCHAR(8), "
                    "qty FLOAT, price FLOAT, pnl FLOAT)"
                )
            )
        monkeypatch.setattr(database, "engine", engine)

        database.init_db()

        inspector = inspect(engine)
        signal_columns = {column["name"] for column in inspector.get_columns("signals")}
        trade_columns = {column["name"] for column in inspector.get_columns("trades")}
        assert {"spread_bps", "quote_imbalance", "model_version"} <= signal_columns
        assert {
            "entry_price",
            "exit_price",
            "notional",
            "gross_pnl",
            "net_pnl",
            "fee_amount",
            "slippage_amount",
            "hold_seconds",
            "reason",
        } <= trade_columns
    finally:
        engine.dispose()


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
                    Order(symbol="BTC/USD", side="buy", status="canceled", created_at=now - timedelta(minutes=10)),
                    Order(symbol="BTC/USD", side="buy", status="filled", created_at=now - timedelta(minutes=20)),
                    Order(symbol="BTC/USD", side="buy", status="submitted", created_at=now - timedelta(minutes=30)),
                    Order(symbol="BTC/USD", side="sell", status="filled", created_at=now - timedelta(hours=2)),
                    Order(symbol="BTC/USD", side="buy", status="filled", created_at=now - timedelta(days=1)),
                    Trade(symbol="BTC/USD", side="sell", qty=0.01, price=65000, pnl=-2, created_at=now),
                    Trade(symbol="BTC/USD", side="sell", qty=0.01, price=65000, pnl=-1, created_at=now - timedelta(minutes=5)),
                    Trade(symbol="BTC/USD", side="sell", qty=0.01, price=65000, pnl=3, created_at=now - timedelta(minutes=10)),
                ]
            )
            db.commit()

            repo = Repository(db)
            state = repo.trade_frequency_state(now=now)
            filled_state = repo.filled_trade_frequency_state(now=now)
            attempt_state = repo.order_attempt_frequency_state(now=now)
            recent_attempts = repo.recent_order_attempts_since(now - timedelta(hours=1))
            recent_filled = repo.recent_filled_orders_since(now - timedelta(hours=1))

        assert state.trades_last_hour == 1
        assert state.trades_today == 2
        assert state.consecutive_losses == 2
        assert state.last_trade_at == now - timedelta(minutes=20)
        assert filled_state == state
        assert len(recent_filled) == 1
        assert attempt_state.trades_last_hour == 3
        assert attempt_state.trades_today == 4
        assert attempt_state.last_trade_at == now - timedelta(minutes=10)
        assert len(recent_attempts) == 3
    finally:
        engine.dispose()


def test_repository_builds_scalping_short_window_risk_state(tmp_path):
    engine, Session = _session_factory(tmp_path)
    now = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)

    try:
        with Session() as db:
            db.add_all(
                [
                    Order(symbol="BTC/USD", side="buy", status="filled", created_at=now - timedelta(minutes=1)),
                    Order(symbol="BTC/USD", side="sell", status="partially_filled", created_at=now - timedelta(minutes=2)),
                    Order(symbol="BTC/USD", side="buy", status="canceled", created_at=now - timedelta(minutes=3)),
                    Order(symbol="BTC/USD", side="buy", status="filled", created_at=now - timedelta(minutes=11)),
                    Trade(symbol="BTC/USD", side="sell", qty=0.01, price=65000, pnl=-3, created_at=now - timedelta(minutes=5)),
                    Trade(symbol="BTC/USD", side="sell", qty=0.01, price=65000, pnl=-4, created_at=now - timedelta(minutes=50)),
                    Trade(symbol="BTC/USD", side="sell", qty=0.01, price=65000, pnl=-9, created_at=now - timedelta(minutes=61)),
                ]
            )
            db.commit()

            repo = Repository(db)
            filled_state = repo.filled_trade_frequency_state(now=now)
            attempt_state = repo.order_attempt_frequency_state(now=now)

        assert filled_state.trades_last_10_minutes == 2
        assert attempt_state.trades_last_10_minutes == 3
        assert repo.filled_trade_count_last_10_minutes(now=now) == 2
        assert repo.order_attempt_count_last_10_minutes(now=now) == 3
        assert filled_state.realized_pnl_last_hour == -7
        assert repo.realized_pnl_last_hour(now=now) == -7
    finally:
        engine.dispose()


def test_consecutive_ioc_cancel_count_stops_at_non_cancel_order(tmp_path):
    engine, Session = _session_factory(tmp_path)
    now = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)

    try:
        with Session() as db:
            db.add_all(
                [
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(seconds=1),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(seconds=2),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="filled",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(seconds=3),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(seconds=4),
                    ),
                ]
            )
            db.commit()

            assert Repository(db).consecutive_ioc_canceled_count() == 2
    finally:
        engine.dispose()


def test_recent_ioc_canceled_buy_count_ignores_old_orders_outside_lookback(tmp_path):
    engine, Session = _session_factory(tmp_path)
    now = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)

    try:
        with Session() as db:
            db.add_all(
                [
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(minutes=10),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(minutes=9),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(minutes=8),
                    ),
                ]
            )
            db.commit()

            count = Repository(db).recent_ioc_canceled_buy_count(now=now, lookback_seconds=300)

        assert count == 0
    finally:
        engine.dispose()


def test_repository_account_snapshot_helpers_store_redacted_payload(tmp_path):
    engine, Session = _session_factory(tmp_path)

    try:
        with Session() as db:
            repo = Repository(db)
            snapshot = repo.add_account_snapshot(
                equity="1000.50",
                cash="900.25",
                buying_power="800.75",
                portfolio_value="1001.00",
                currency="USD",
                raw_response={
                    "equity": "1000.50",
                    "api_key": "raw-key",
                    "nested": {"secret_token": "hidden"},
                },
            )
            latest = repo.latest_account_snapshot()
            recent = repo.recent_account_snapshots()
            raw = json.loads(snapshot.raw_response)

        assert snapshot.equity == 1000.50
        assert snapshot.cash == 900.25
        assert snapshot.buying_power == 800.75
        assert snapshot.portfolio_value == 1001.00
        assert snapshot.currency == "USD"
        assert raw["api_key"] == "***"
        assert raw["nested"]["secret_token"] == "***"
        assert latest is not None
        assert latest.id == snapshot.id
        assert recent == [snapshot]
    finally:
        engine.dispose()


def test_recent_ioc_canceled_buy_count_counts_recent_ioc_limit_buys(tmp_path):
    engine, Session = _session_factory(tmp_path)
    now = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)

    try:
        with Session() as db:
            db.add_all(
                [
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(seconds=30),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=json.dumps({"type": "limit", "time_in_force": "ioc"}),
                        created_at=now - timedelta(seconds=60),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="sell",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(seconds=90),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="submitted",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(seconds=120),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=json.dumps({"order_type": "market", "time_in_force": "ioc"}),
                        created_at=now - timedelta(seconds=150),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=json.dumps({"order_type": "limit", "time_in_force": "gtc"}),
                        created_at=now - timedelta(seconds=180),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(minutes=10),
                    ),
                ]
            )
            db.commit()

            repo = Repository(db)
            count = repo.recent_ioc_canceled_buy_count(now=now, lookback_seconds=300)
            buy_count = repo.recent_ioc_canceled_count(side="buy", now=now, lookback_seconds=300)
            all_count = repo.recent_ioc_canceled_count(now=now, lookback_seconds=300)

        assert count == 2
        assert buy_count == 2
        assert all_count == 3
    finally:
        engine.dispose()


def test_latest_ioc_canceled_buy_at_returns_latest_recent_ioc_limit_buy(tmp_path):
    engine, Session = _session_factory(tmp_path)
    now = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)
    expected = now - timedelta(seconds=20)

    try:
        with Session() as db:
            db.add_all(
                [
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=now - timedelta(seconds=60),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=expected,
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=json.dumps({"order_type": "market", "time_in_force": "ioc"}),
                        created_at=now - timedelta(seconds=10),
                    ),
                ]
            )
            db.commit()

            repo = Repository(db)
            latest = repo.latest_ioc_canceled_buy_at(now=now, lookback_seconds=300)
            latest_order = repo.latest_ioc_canceled_order(side="buy", now=now, lookback_seconds=300)

        assert latest == expected
        assert latest_order is not None
        assert latest_order.created_at.replace(tzinfo=UTC) == expected
    finally:
        engine.dispose()


def test_recent_ioc_cancels_guard_clears_after_lookback_window(tmp_path):
    engine, Session = _session_factory(tmp_path)
    settings = Settings(_env_file=None, scalping_mode_enabled=True, trading_enabled=True)
    engine_decisions = DecisionEngine(settings)
    started_at = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)

    try:
        with Session() as db:
            db.add_all(
                [
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=started_at,
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=started_at + timedelta(seconds=1),
                    ),
                    Order(
                        symbol="BTC/USD",
                        side="buy",
                        status="canceled",
                        raw_response=_ioc_limit_raw(),
                        created_at=started_at + timedelta(seconds=2),
                    ),
                ]
            )
            db.commit()

            current_count = Repository(db).recent_ioc_canceled_buy_count(
                now=started_at + timedelta(seconds=10),
                lookback_seconds=settings.ioc_cancel_lookback_seconds,
            )
            cleared_count = Repository(db).recent_ioc_canceled_buy_count(
                now=started_at + timedelta(seconds=settings.ioc_cancel_lookback_seconds + 3),
                lookback_seconds=settings.ioc_cancel_lookback_seconds,
            )

        blocked = engine_decisions.decide(
            prediction={"symbol": "BTC/USD", "buy_probability": 0.9, "sell_probability": 0.1},
            feature_row=_scalping_feature_row(),
            position=PositionState(),
            trading_enabled=True,
            recent_ioc_canceled_buys=current_count,
        )
        cleared = engine_decisions.decide(
            prediction={"symbol": "BTC/USD", "buy_probability": 0.9, "sell_probability": 0.1},
            feature_row=_scalping_feature_row(),
            position=PositionState(),
            trading_enabled=True,
            recent_ioc_canceled_buys=cleared_count,
        )

        assert current_count == settings.max_recent_ioc_cancels
        assert blocked.action == "hold"
        assert blocked.reason == "recent_ioc_cancels_too_high"
        assert cleared_count == 0
        assert cleared.action == "buy"
    finally:
        engine.dispose()
