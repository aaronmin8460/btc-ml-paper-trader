import pandas as pd
import pytest

from app.backtest.scalping import backtest_assumptions, calculate_fee_aware_metrics, walk_forward_fee_aware_backtest
from app.config import Settings
from app.data.dataset_builder import build_training_dataset
from app.data.market_data import MarketDataClient
from app.ml.validation import walk_forward_validate


def _settings(**overrides):
    defaults = {
        "_env_file": None,
        "order_notional_usd": 100,
        "take_profit_pct": 0.006,
        "stop_loss_pct": 0.0025,
        "taker_fee_bps": 0,
        "maker_fee_bps": 0,
        "slippage_bps": 0,
        "backtest_use_taker_fees": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "buy_quality_label": [1, 0, 1],
            "orderbook_spread": [0.0, 0.0, 0.0],
        }
    )


def test_fees_reduce_net_returns():
    no_fee = calculate_fee_aware_metrics(_trades(), _settings())
    with_fee = calculate_fee_aware_metrics(_trades(), _settings(taker_fee_bps=25))

    assert with_fee["gross_return"] == no_fee["gross_return"]
    assert with_fee["net_return"] < no_fee["net_return"]
    assert with_fee["fees_paid_estimate"] > 0


def test_slippage_reduces_net_returns():
    no_slippage = calculate_fee_aware_metrics(_trades(), _settings())
    with_slippage = calculate_fee_aware_metrics(_trades(), _settings(slippage_bps=10))

    assert with_slippage["gross_return"] == no_slippage["gross_return"]
    assert with_slippage["net_return"] < no_slippage["net_return"]
    assert with_slippage["slippage_paid_estimate"] > 0


def test_backtest_report_metrics_include_gross_and_net_fields():
    settings = _settings(taker_fee_bps=25, slippage_bps=10)
    metrics = calculate_fee_aware_metrics(_trades(), settings)
    assumptions = backtest_assumptions(settings, spread_available=True)

    for key in [
        "number_of_trades",
        "win_rate",
        "average_gross_win",
        "average_gross_loss",
        "average_net_win",
        "average_net_loss",
        "profit_factor_gross",
        "profit_factor_net",
        "max_drawdown",
        "fees_paid_estimate",
        "slippage_paid_estimate",
        "total_fees",
        "total_slippage",
        "total_spread_cost",
        "number_of_canceled_orders",
        "partial_fill_count",
        "average_hold_bars",
        "win_rate_net",
        "ambiguous_candle_count",
    ]:
        assert key in metrics
    assert metrics["gross_return"] != metrics["net_return"]
    assert assumptions["spread_source"] == "orderbook_spread"


def test_insufficient_backtest_data_returns_clear_reason():
    settings = _settings()
    report = walk_forward_fee_aware_backtest(pd.DataFrame(), settings, min_train_rows=100, threshold=0.5)

    assert report["valid"] is False
    assert report["reason"] == "not_enough_rows"


def test_existing_walk_forward_validation_still_works():
    bars = MarketDataClient.synthetic_btc_bars(320)
    dataset = build_training_dataset(bars)

    metrics = walk_forward_validate(dataset, min_train_rows=120, threshold=0.5, folds=2)

    assert metrics["rows"] == len(dataset)
    assert "precision" in metrics
    assert "sell_precision" in metrics
    assert "sell_class_balance" in metrics
    assert "number_of_trades" in metrics


def test_ambiguous_scalping_candle_is_counted_as_stop_loss():
    trades = pd.DataFrame(
        {
            "close": [100.0],
            "buy_quality_label": [1],
            "buy_exit_return_pct": [0.01],
            "buy_exit_reason": ["ambiguous_stop_first"],
        }
    )

    metrics = calculate_fee_aware_metrics(
        trades,
        _settings(scalping_mode_enabled=True, max_spread_bps=0, scalping_label_stop_loss_pct=0.001),
    )

    assert metrics["ambiguous_candle_count"] == 1
    assert metrics["ambiguous_candle_ratio"] == 1.0
    assert metrics["gross_return_pct"] == pytest.approx(-0.001)
    assert metrics["win_rate_net"] == 0.0


def test_scalping_backtest_uses_configured_spread_when_quote_is_missing():
    trades = pd.DataFrame({"close": [100.0], "buy_quality_label": [1], "buy_exit_return_pct": [0.01]})

    metrics = calculate_fee_aware_metrics(
        trades,
        _settings(scalping_mode_enabled=True, max_spread_bps=10),
    )

    assert metrics["total_spread_cost"] > 0
    assert metrics["spread_paid_estimate"] == metrics["total_spread_cost"]
    assert metrics["net_return_pct"] < metrics["gross_return_pct"]


def test_canceled_ioc_entry_is_not_counted_as_a_win():
    trades = pd.DataFrame(
        {
            "close": [100.0],
            "bid_price": [99.9],
            "ask_price": [100.1],
            "entry_limit_price": [100.0],
            "buy_quality_label": [1],
            "buy_exit_return_pct": [0.01],
        }
    )

    metrics = calculate_fee_aware_metrics(trades, _settings(order_type="limit", time_in_force="ioc"))

    assert metrics["valid"] is False
    assert metrics["reason"] == "no_filled_trades"
    assert metrics["number_of_canceled_orders"] == 1
    assert metrics["number_of_trades"] == 0
    assert metrics["win_rate_net"] == 0.0


def test_backtest_tracks_partial_entry_and_exit_fills():
    trades = pd.DataFrame(
        {
            "close": [100.0],
            "bid_size": [0.25],
            "ask_size": [0.5],
            "buy_quality_label": [1],
            "buy_exit_return_pct": [0.01],
        }
    )

    metrics = calculate_fee_aware_metrics(trades, _settings(order_type="limit", time_in_force="ioc"))

    assert metrics["valid"] is True
    assert metrics["number_of_trades"] == 1
    assert metrics["partial_fill_count"] == 2
