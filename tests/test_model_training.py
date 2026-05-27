from datetime import UTC, datetime

import pandas as pd

from app.config import Settings
from app.data.dataset_builder import build_training_dataset
from app.data.feature_engineering import FEATURE_COLUMNS
from app.data.market_data import MarketDataClient
from app.ml.model import MLSignalModel
from app.ml.train import train_model_from_bars
from app.ml.validation import promotion_decision


def test_model_training_produces_probabilities():
    bars = MarketDataClient.synthetic_btc_bars(260)
    dataset = build_training_dataset(bars)
    model = MLSignalModel(feature_columns=FEATURE_COLUMNS).train(dataset)
    probabilities = model.predict_proba(dataset.tail(5))
    assert len(probabilities) == 5
    assert all(0 <= value <= 1 for value in probabilities)


def test_labels_drop_rows_where_future_unavailable():
    bars = MarketDataClient.synthetic_btc_bars(120)
    dataset = build_training_dataset(bars)
    assert len(dataset) < len(bars)
    assert "buy_quality_label" in dataset.columns


def test_training_rejects_single_class_dataset_without_crashing(tmp_path):
    timestamps = pd.date_range(datetime(2026, 5, 27, tzinfo=UTC), periods=160, freq="min")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * len(timestamps),
            "high": [100.0] * len(timestamps),
            "low": [100.0] * len(timestamps),
            "close": [100.0] * len(timestamps),
            "volume": [1.0] * len(timestamps),
        }
    )
    settings = Settings(
        _env_file=None,
        min_training_rows=50,
        model_dir=str(tmp_path),
        scalping_mode_enabled=True,
    )

    result = train_model_from_bars(bars, settings)

    assert result["accepted"] is False
    assert result["reason"] == "target_class_diversity_too_low"
    assert result["model_path"] is None
    assert list(tmp_path.glob("*.joblib")) == []


def _passing_metrics(**overrides):
    metrics = {
        "validation_rows": 200,
        "precision": 0.8,
        "profit_factor": 1.5,
        "max_drawdown": 0.005,
        "number_of_trades": 25,
        "net_return_pct": 0.01,
        "max_drawdown_pct": 0.005,
        "profit_factor_net": 1.2,
        "fee_aware_backtest_valid": True,
    }
    metrics.update(overrides)
    return metrics


def test_model_rejected_when_precision_passes_but_net_return_is_negative():
    accepted, reason = promotion_decision(
        _passing_metrics(net_return_pct=-0.01),
        min_rows=100,
        min_precision=0.52,
        max_drawdown=0.2,
        max_trade_fraction=0.4,
        min_net_return_pct=0.001,
        max_backtest_drawdown_pct=0.01,
        min_backtest_profit_factor=1.05,
        min_backtest_trades=20,
        require_positive_net_return=True,
    )

    assert accepted is False
    assert reason == "model_not_profitable_after_costs"


def test_model_rejected_when_backtest_drawdown_is_too_high():
    accepted, reason = promotion_decision(
        _passing_metrics(max_drawdown_pct=0.02),
        min_rows=100,
        min_precision=0.52,
        max_drawdown=0.2,
        max_trade_fraction=0.4,
        min_net_return_pct=0.001,
        max_backtest_drawdown_pct=0.01,
        min_backtest_profit_factor=1.05,
        min_backtest_trades=20,
        require_positive_net_return=True,
    )

    assert accepted is False
    assert reason == "backtest_drawdown_too_high"


def test_model_accepted_only_when_ml_and_account_style_metrics_pass():
    accepted, reason = promotion_decision(
        _passing_metrics(),
        min_rows=100,
        min_precision=0.52,
        max_drawdown=0.2,
        max_trade_fraction=0.4,
        min_net_return_pct=0.001,
        max_backtest_drawdown_pct=0.01,
        min_backtest_profit_factor=1.05,
        min_backtest_trades=20,
        require_positive_net_return=True,
    )

    assert accepted is True
    assert reason == "accepted"
