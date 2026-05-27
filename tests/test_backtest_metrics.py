import pandas as pd

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
    assert "number_of_trades" in metrics
