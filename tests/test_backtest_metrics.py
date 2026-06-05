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
        "expectancy",
        "trade_details",
        "strategy_level_metrics",
        "regime_level_metrics",
        "blocked_signal_metrics",
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


def test_backtest_counts_gross_winners_that_become_net_losers():
    trades = pd.DataFrame({"close": [100.0], "buy_quality_label": [1], "buy_exit_return_pct": [0.003]})

    metrics = calculate_fee_aware_metrics(
        trades,
        _settings(
            order_type="market",
            scalping_mode_enabled=True,
            taker_fee_bps=25,
            slippage_bps=10,
            max_spread_bps=5,
        ),
    )

    assert metrics["gross_return_pct"] > 0
    assert metrics["net_return_pct"] < 0
    assert metrics["gross_winners_became_net_losers"] == 1
    assert metrics["average_gross_winning_trade"] == pytest.approx(0.003)
    assert metrics["average_net_winning_trade"] == 0.0
    assert metrics["required_gross_return_to_overcome_costs"] == pytest.approx(0.0075)


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


def test_backtest_reports_strategy_level_net_metrics():
    trades = pd.DataFrame(
        {
            "close": [100.0, 100.0],
            "buy_quality_label": [1, 0],
            "buy_exit_return_pct": [0.01, -0.002],
            "strategy_name": ["mean_reversion_scalping", "momentum_breakout"],
            "entry_reason": ["mean_reversion_buy_candidate", "momentum_breakout_buy_candidate"],
        }
    )

    metrics = calculate_fee_aware_metrics(trades, _settings(order_type="market"))

    assert metrics["trade_details"][0]["strategy_name"] == "mean_reversion_scalping"
    assert metrics["trade_details"][0]["entry_reason"] == "mean_reversion_buy_candidate"
    assert metrics["trade_details"][0]["gross_return_pct"] == metrics["trade_details"][0]["gross_return"]
    assert metrics["trade_details"][0]["net_return_pct"] == metrics["trade_details"][0]["net_return"]
    assert "fee_amount" in metrics["trade_details"][0]
    assert "hold_bars" in metrics["trade_details"][0]
    assert "mean_reversion_scalping" in metrics["strategy_level_metrics"]
    assert "momentum_breakout" in metrics["strategy_level_metrics"]
    assert "net_return" in metrics["strategy_level_metrics"]["mean_reversion_scalping"]
    assert "profit_factor" in metrics["strategy_level_metrics"]["momentum_breakout"]
    assert "profit_factor_net" in metrics["strategy_level_metrics"]["momentum_breakout"]


def test_backtest_reports_strategy_regime_and_blocked_signal_metrics():
    signals = pd.DataFrame(
        {
            "close": [100.0, 100.0, 100.0],
            "buy_quality_label": [1, 0, 1],
            "buy_exit_return_pct": [0.01, -0.002, 0.01],
            "_probability": [0.8, 0.7, 0.3],
            "ml_buy_probability": [0.8, 0.7, 0.3],
            "ml_sell_probability": [0.2, 0.3, 0.7],
            "strategy_name": ["mean_reversion_scalping", "momentum_breakout", "mean_reversion_scalping"],
            "regime": ["mean_reverting", "too_volatile", "mean_reverting"],
            "entry_reason": [
                "mean_reversion_buy_candidate",
                "volatility_too_high",
                "mean_reversion_buy_candidate",
            ],
            "blocked_by": [None, "regime_filter", "ml_filter"],
            "block_reason": [None, "volatility_too_high", "ml_buy_probability_below_threshold"],
            "strategy_score": [0.7, 0.0, 0.6],
            "strategy_confidence": [0.8, 0.0, 0.7],
            "entry_allowed": [True, False, False],
        }
    )

    metrics = calculate_fee_aware_metrics(
        signals.loc[signals["entry_allowed"]].copy(),
        _settings(order_type="market"),
        signal_frame=signals,
    )

    assert metrics["trade_details"][0]["strategy_name"] == "mean_reversion_scalping"
    assert metrics["trade_details"][0]["regime"] == "mean_reverting"
    assert metrics["trade_details"][0]["ml_buy_probability"] == 0.8
    assert metrics["trade_details"][0]["ml_sell_probability"] == 0.2
    assert metrics["trade_details"][0]["quant_score"] == 0.7
    assert metrics["trade_details"][0]["quant_confidence"] == 0.8
    assert metrics["blocked_signal_metrics"]["regime_filter"] == 1
    assert metrics["blocked_signal_metrics"]["ml_filter"] == 1
    assert metrics["strategy_level_metrics"]["mean_reversion_scalping"]["number_of_signals"] == 2
    assert metrics["strategy_level_metrics"]["mean_reversion_scalping"]["number_of_entries"] == 1
    assert metrics["strategy_level_metrics"]["mean_reversion_scalping"]["number_of_trades"] == 1
    assert metrics["strategy_level_metrics"]["momentum_breakout"]["number_of_signals"] == 1
    assert metrics["strategy_level_metrics"]["momentum_breakout"]["number_of_trades"] == 0
    assert metrics["regime_level_metrics"]["too_volatile"]["number_of_blocked_signals"] == 1
    assert metrics["regime_level_metrics"]["mean_reverting"]["number_of_allowed_signals"] == 1
    assert "profitable" not in str(metrics["strategy_level_metrics"]).lower()
