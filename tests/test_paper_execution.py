import pytest

from app.broker.paper_execution import simulate_limit_ioc_order, simulate_market_order


def test_limit_ioc_buy_fills_when_limit_reaches_ask():
    result = simulate_limit_ioc_order(
        side="buy",
        bid_price=99.0,
        ask_price=100.0,
        ask_size=2.0,
        qty=1.0,
        limit_price=100.0,
    )

    assert result.status == "filled"
    assert result.filled_qty == pytest.approx(1.0)
    assert result.filled_avg_price == pytest.approx(100.0)
    assert result.order_type == "limit"
    assert result.time_in_force == "ioc"


def test_limit_ioc_buy_cancels_when_limit_is_below_ask():
    result = simulate_limit_ioc_order(
        side="buy",
        bid_price=99.0,
        ask_price=100.0,
        ask_size=2.0,
        qty=1.0,
        limit_price=99.99,
    )

    assert result.status == "canceled"
    assert result.filled_qty == 0
    assert result.filled_avg_price is None


def test_market_buy_applies_ask_plus_slippage():
    result = simulate_market_order(
        side="buy",
        bid_price=99.0,
        ask_price=100.0,
        ask_size=2.0,
        qty=1.0,
        slippage_bps=10,
    )

    assert result.filled_avg_price == pytest.approx(100.1)
    assert result.slippage_amount == pytest.approx(0.1)
    assert result.spread_amount == pytest.approx(1.0)
    assert result.spread_bps == pytest.approx(100.50251256)
    assert result.spread_cost_amount == pytest.approx(0.5)


def test_market_sell_applies_bid_minus_slippage():
    result = simulate_market_order(
        side="sell",
        bid_price=100.0,
        ask_price=101.0,
        bid_size=2.0,
        qty=1.0,
        slippage_bps=10,
    )

    assert result.filled_avg_price == pytest.approx(99.9)
    assert result.slippage_amount == pytest.approx(0.1)


def test_market_fill_calculates_fee_from_realized_notional():
    result = simulate_market_order(
        side="buy",
        ask_price=100.0,
        ask_size=2.0,
        qty=1.0,
        fee_bps=25,
    )

    assert result.filled_notional == pytest.approx(100.0)
    assert result.fee_amount == pytest.approx(0.25)


def test_quote_size_limits_fill_quantity():
    result = simulate_market_order(
        side="buy",
        ask_price=100.0,
        ask_size=0.25,
        qty=1.0,
    )

    assert result.status == "partially_filled"
    assert result.filled_qty == pytest.approx(0.25)
    assert result.unfilled_qty == pytest.approx(0.75)


def test_limit_ioc_sell_cancels_when_limit_is_above_bid():
    result = simulate_limit_ioc_order(
        side="sell",
        bid_price=100.0,
        ask_price=101.0,
        bid_size=2.0,
        qty=1.0,
        limit_price=100.01,
    )

    assert result.status == "canceled"
    assert result.filled_qty == 0
