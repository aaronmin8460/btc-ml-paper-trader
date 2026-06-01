import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.accounting.trade_accounting import local_simulated_position_from_orders, record_filled_order_trade
from app.broker.paper_execution import simulate_limit_ioc_order
from app.config import Settings
from app.db.database import Base, connect_args_for_database_url
from app.db.repository import Repository


def _session_factory(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'trading.db'}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))
    Base.metadata.create_all(bind=engine)
    return engine, sessionmaker(bind=engine)


def _filled_raw(*, side: str, qty: float, price: float, fee: float | None = None) -> dict:
    raw = {
        "id": f"{side}-{price}",
        "status": "filled",
        "symbol": "BTC/USD",
        "side": side,
        "filled_qty": str(qty),
        "filled_avg_price": str(price),
    }
    if fee is not None:
        raw["fee"] = str(fee)
    return raw


def test_filled_buy_alone_does_not_create_realized_trade(tmp_path):
    engine, Session = _session_factory(tmp_path)
    try:
        with Session() as db:
            repo = Repository(db)
            order = repo.add_order(
                side="buy",
                status="filled",
                notional=100,
                raw_response=_filled_raw(side="buy", qty=0.01, price=10_000),
            )

            trade = record_filled_order_trade(repo, order, Settings(_env_file=None))

            assert trade is None
            assert repo.recent_trades() == []
    finally:
        engine.dispose()


def test_filled_sell_after_buy_creates_realized_trade(tmp_path):
    engine, Session = _session_factory(tmp_path)
    try:
        with Session() as db:
            repo = Repository(db)
            buy = repo.add_order(
                side="buy",
                status="filled",
                notional=100,
                raw_response=_filled_raw(side="buy", qty=0.01, price=10_000),
            )
            record_filled_order_trade(repo, buy, Settings(_env_file=None))
            sell = repo.add_order(
                side="sell",
                status="filled",
                qty=0.01,
                raw_response=_filled_raw(side="sell", qty=0.01, price=11_000),
            )

            trade = record_filled_order_trade(repo, sell, Settings(_env_file=None))

            assert trade is not None
            assert trade.side == "sell"
            assert trade.qty == pytest.approx(0.01)
            assert trade.price == pytest.approx(11_000)
            assert trade.pnl == pytest.approx(10.0)
            assert trade.entry_price == pytest.approx(10_000)
            assert trade.exit_price == pytest.approx(11_000)
            assert trade.notional == pytest.approx(110)
            assert trade.gross_pnl == pytest.approx(10)
            assert trade.net_pnl == pytest.approx(10)
            assert trade.fee_amount == pytest.approx(0)
            assert trade.slippage_amount == pytest.approx(0)
            assert trade.hold_seconds is not None
            assert len(repo.recent_trades()) == 1
    finally:
        engine.dispose()


def test_canceled_order_does_not_create_trade_row(tmp_path):
    engine, Session = _session_factory(tmp_path)
    try:
        with Session() as db:
            repo = Repository(db)
            order = repo.add_order(
                side="buy",
                status="canceled",
                notional=100,
                raw_response=_filled_raw(side="buy", qty=0.01, price=10_000),
            )

            trade = record_filled_order_trade(repo, order, Settings(_env_file=None))

            assert trade is None
            assert repo.recent_trades() == []
    finally:
        engine.dispose()


def test_dry_run_order_does_not_create_trade_row(tmp_path):
    engine, Session = _session_factory(tmp_path)
    try:
        with Session() as db:
            repo = Repository(db)
            order = repo.add_order(
                side="buy",
                status="dry_run_trading_disabled",
                notional=100,
                raw_response={
                    "status": "dry_run_trading_disabled",
                    "symbol": "BTC/USD",
                    "side": "buy",
                },
            )

            trade = record_filled_order_trade(repo, order, Settings(_env_file=None))

            assert trade is None
            assert repo.recent_trades() == []
    finally:
        engine.dispose()


def test_rejected_order_does_not_create_trade_row(tmp_path):
    engine, Session = _session_factory(tmp_path)
    try:
        with Session() as db:
            repo = Repository(db)
            order = repo.add_order(
                side="buy",
                status="rejected",
                notional=100,
                raw_response={
                    "status": "rejected",
                    "symbol": "BTC/USD",
                    "side": "buy",
                },
            )

            trade = record_filled_order_trade(repo, order, Settings(_env_file=None))

            assert trade is None
            assert repo.recent_trades() == []
    finally:
        engine.dispose()


def test_paper_fees_and_slippage_reduce_realized_pnl(tmp_path):
    engine, Session = _session_factory(tmp_path)
    settings = Settings(_env_file=None, paper_fee_bps=10, paper_slippage_bps=10)
    try:
        with Session() as db:
            repo = Repository(db)
            buy = repo.add_order(
                side="buy",
                status="filled",
                notional=100,
                raw_response=_filled_raw(side="buy", qty=0.01, price=10_000),
            )
            record_filled_order_trade(repo, buy, settings)
            sell = repo.add_order(
                side="sell",
                status="filled",
                qty=0.01,
                raw_response=_filled_raw(side="sell", qty=0.01, price=11_000),
            )

            trade = record_filled_order_trade(repo, sell, settings)

            assert trade is not None
            assert trade.pnl == pytest.approx(9.58)
            assert trade.pnl < 10.0
    finally:
        engine.dispose()


def test_partial_sell_fill_creates_trade_for_filled_quantity_only(tmp_path):
    engine, Session = _session_factory(tmp_path)
    try:
        with Session() as db:
            repo = Repository(db)
            buy = repo.add_order(
                side="buy",
                status="filled",
                notional=100,
                raw_response=_filled_raw(side="buy", qty=0.01, price=10_000),
            )
            record_filled_order_trade(repo, buy, Settings(_env_file=None))
            sell = repo.add_order(
                side="sell",
                status="partially_filled",
                qty=0.01,
                raw_response=_filled_raw(side="sell", qty=0.004, price=11_000),
            )

            trade = record_filled_order_trade(repo, sell, Settings(_env_file=None))

            assert trade is not None
            assert trade.qty == pytest.approx(0.004)
            assert trade.pnl == pytest.approx(4.0)
    finally:
        engine.dispose()


def test_simulated_fill_price_is_not_charged_slippage_twice(tmp_path):
    engine, Session = _session_factory(tmp_path)
    settings = Settings(_env_file=None, paper_fee_bps=10, paper_slippage_bps=10)
    try:
        with Session() as db:
            repo = Repository(db)
            buy_raw = _filled_raw(side="buy", qty=0.01, price=10_010, fee=1.001)
            buy_raw.update({"execution_source": "local_simulated", "slippage_applied": True})
            buy = repo.add_order(side="buy", status="filled", notional=100.1, raw_response=buy_raw)
            record_filled_order_trade(repo, buy, settings)

            sell_raw = _filled_raw(side="sell", qty=0.01, price=10_989, fee=1.0989)
            sell_raw.update({"execution_source": "local_simulated", "slippage_applied": True})
            sell = repo.add_order(side="sell", status="filled", qty=0.01, raw_response=sell_raw)

            trade = record_filled_order_trade(repo, sell, settings)

            assert trade is not None
            assert trade.price == pytest.approx(10_989)
            assert trade.pnl == pytest.approx(7.6901)
    finally:
        engine.dispose()


def test_simulated_canceled_order_is_stored_but_not_counted_as_trade(tmp_path):
    engine, Session = _session_factory(tmp_path)
    try:
        with Session() as db:
            repo = Repository(db)
            fill = simulate_limit_ioc_order(
                side="buy",
                ask_price=100,
                ask_size=1,
                qty=1,
                limit_price=99,
            )
            order = repo.add_order(
                side="buy",
                status=fill.status,
                notional=100,
                raw_response=fill.to_order_response(),
            )

            trade = record_filled_order_trade(repo, order, Settings(_env_file=None))

            assert order.status == "canceled"
            assert repo.recent_orders() == [order]
            assert trade is None
            assert repo.recent_trades() == []
    finally:
        engine.dispose()


def test_local_simulated_position_can_be_reconstructed_after_restart(tmp_path):
    engine, Session = _session_factory(tmp_path)
    try:
        with Session() as db:
            repo = Repository(db)
            buy_raw = _filled_raw(side="buy", qty=0.01, price=10_010)
            buy_raw["execution_source"] = "local_simulated"
            repo.add_order(side="buy", status="filled", notional=100.1, raw_response=buy_raw)
            sell_raw = _filled_raw(side="sell", qty=0.004, price=10_989)
            sell_raw["execution_source"] = "local_simulated"
            repo.add_order(side="sell", status="partially_filled", qty=0.004, raw_response=sell_raw)

            position = local_simulated_position_from_orders(repo)

            assert position is not None
            assert position["qty"] == pytest.approx(0.006)
            assert position["avg_entry_price"] == pytest.approx(10_010)
            assert position["current_price"] == pytest.approx(10_989)
            assert position["market_value"] == pytest.approx(65.934)
    finally:
        engine.dispose()
