import json
from datetime import UTC, datetime

import pandas as pd

from app.config import Settings
from app.data.dataset_builder import build_training_dataset
from app.data.feature_engineering import BAR_FEATURE_COLUMNS, RUNTIME_ONLY_FEATURE_COLUMNS
from app.data.market_data import MarketDataClient
from app.ml.labels import net_profit_scalping_labels
from app.ml.model import MLSignalModel
from app.ml.predict import Predictor
from app.ml.registry import ModelRegistry
from app.ml.train import train_model_from_bars
from app.ml.validation import promotion_decision


def test_model_training_produces_probabilities():
    bars = MarketDataClient.synthetic_btc_bars(260)
    dataset = build_training_dataset(bars)
    model = MLSignalModel(feature_columns=BAR_FEATURE_COLUMNS).train(dataset)
    probabilities = model.predict_proba(dataset.tail(5))
    assert len(probabilities) == 5
    assert all(0 <= value <= 1 for value in probabilities)


def test_labels_drop_rows_where_future_unavailable():
    bars = MarketDataClient.synthetic_btc_bars(120)
    dataset = build_training_dataset(bars)
    assert len(dataset) < len(bars)
    assert "buy_quality_label" in dataset.columns


def test_predictor_marks_fallback_predictions_when_no_model_is_active(tmp_path):
    settings = Settings(_env_file=None, model_dir=str(tmp_path))
    bars = MarketDataClient.synthetic_btc_bars(140)

    prediction = Predictor(settings).predict(bars)

    assert prediction["prediction_source"] == "fallback"
    assert prediction["model_available"] is False
    assert prediction["model_path"] is None
    assert prediction["active_model_status"] == "stale"


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
        "number_of_trades": 35,
        "net_return_pct": 0.01,
        "max_drawdown_pct": 0.005,
        "profit_factor_net": 1.2,
        "fee_aware_backtest_valid": True,
        "promotion_reason": "accepted",
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
        min_backtest_profit_factor=1.2,
        min_backtest_trades=30,
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
        min_backtest_profit_factor=1.2,
        min_backtest_trades=30,
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
        min_backtest_profit_factor=1.2,
        min_backtest_trades=30,
        require_positive_net_return=True,
    )

    assert accepted is True
    assert reason == "accepted"


def test_scalping_labels_require_net_profitable_exit_after_costs():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range(datetime(2026, 5, 27, tzinfo=UTC), periods=6, freq="min"),
            "open": [100.0] * 6,
            "high": [100.0, 100.30, 100.25, 100.20, 100.10, 100.0],
            "low": [100.0] * 6,
            "close": [100.0] * 6,
            "volume": [1.0] * 6,
        }
    )

    labeled = net_profit_scalping_labels(
        bars,
        horizon_bars=3,
        take_profit_pct=0.0015,
        stop_loss_pct=0.001,
        fee_bps_per_side=25,
        slippage_bps_per_side=10,
        spread_cost_pct=0.001,
        min_net_exit_profit_pct=0.002,
        exit_profit_buffer_bps=5,
    )

    assert int(labeled.loc[0, "buy_quality_label"]) == 0
    assert labeled.loc[0, "buy_exit_reason"] == "no_profitable_exit"


def test_scalping_labels_accept_take_profit_that_covers_costs():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range(datetime(2026, 5, 27, tzinfo=UTC), periods=6, freq="min"),
            "open": [100.0] * 6,
            "high": [100.0, 101.20, 100.20, 100.10, 100.05, 100.0],
            "low": [100.0] * 6,
            "close": [100.0] * 6,
            "volume": [1.0] * 6,
        }
    )

    labeled = net_profit_scalping_labels(
        bars,
        horizon_bars=3,
        take_profit_pct=0.0015,
        stop_loss_pct=0.001,
        fee_bps_per_side=25,
        slippage_bps_per_side=10,
        spread_cost_pct=0.001,
        min_net_exit_profit_pct=0.002,
        exit_profit_buffer_bps=5,
    )

    assert int(labeled.loc[0, "buy_quality_label"]) == 1
    assert labeled.loc[0, "buy_exit_reason"] == "scalping_take_profit"
    assert labeled.loc[0, "buy_exit_return_pct"] >= 0.0105


def test_scalping_dataset_does_not_train_on_runtime_quote_filters():
    bars = MarketDataClient.synthetic_btc_bars(140)
    dataset = build_training_dataset(
        bars,
        scalping_mode_enabled=True,
        take_profit_pct=0.0015,
        stop_loss_pct=0.001,
        trailing_stop_pct=0.0008,
        trailing_stop_arm_profit_pct=0.002,
        fee_bps_per_side=25,
        slippage_bps_per_side=10,
        spread_cost_pct=0.001,
        min_net_exit_profit_pct=0.002,
        exit_profit_buffer_bps=5,
    )

    assert "orderbook_spread" not in BAR_FEATURE_COLUMNS
    assert "quote_imbalance" not in BAR_FEATURE_COLUMNS
    assert RUNTIME_ONLY_FEATURE_COLUMNS == ["orderbook_spread", "quote_imbalance"]
    assert {"orderbook_spread", "quote_imbalance"}.issubset(dataset.columns)


def test_model_rejected_when_positive_gross_return_is_negative_after_costs():
    accepted, reason = promotion_decision(
        _passing_metrics(gross_return_pct=0.02, net_return_pct=-0.002, profit_factor_net=1.5),
        min_rows=100,
        min_precision=0.52,
        max_drawdown=0.2,
        max_trade_fraction=0.4,
        min_net_return_pct=0.001,
        max_backtest_drawdown_pct=0.01,
        min_backtest_profit_factor=1.2,
        min_backtest_trades=30,
        require_positive_net_return=False,
    )

    assert accepted is False
    assert reason == "model_not_profitable_after_costs"


def test_model_rejected_when_net_profit_factor_is_too_low():
    accepted, reason = promotion_decision(
        _passing_metrics(profit_factor_net=1.19),
        min_rows=100,
        min_precision=0.52,
        max_drawdown=0.2,
        max_trade_fraction=0.4,
        min_net_return_pct=0.001,
        max_backtest_drawdown_pct=0.01,
        min_backtest_profit_factor=1.2,
        min_backtest_trades=30,
        require_positive_net_return=True,
    )

    assert accepted is False
    assert reason == "profit_factor_net_too_low"


def test_model_rejected_when_trade_count_is_too_low():
    accepted, reason = promotion_decision(
        _passing_metrics(number_of_trades=29),
        min_rows=100,
        min_precision=0.52,
        max_drawdown=0.2,
        max_trade_fraction=0.4,
        min_net_return_pct=0.001,
        max_backtest_drawdown_pct=0.01,
        min_backtest_profit_factor=1.2,
        min_backtest_trades=30,
        require_positive_net_return=True,
    )

    assert accepted is False
    assert reason == "not_enough_backtest_trades"


def test_active_registry_mismatch_marks_model_invalid(tmp_path):
    settings = Settings(_env_file=None, model_dir=str(tmp_path))
    model_path = tmp_path / "btc_model.joblib"
    metadata_metrics = _passing_metrics(net_return_pct=0.02)
    model = MLSignalModel(feature_columns=BAR_FEATURE_COLUMNS)
    model.metadata["validation_metrics"] = metadata_metrics
    model.metadata["promotion_reason"] = "accepted"
    model.save(model_path)

    ModelRegistry(settings).promote(
        model_path=str(model_path),
        feature_columns=BAR_FEATURE_COLUMNS,
        metrics=_passing_metrics(net_return_pct=0.03),
        thresholds={},
        training_start="2026-05-27T00:00:00+00:00",
        training_end="2026-05-27T00:01:00+00:00",
    )

    status = ModelRegistry(settings).validate_active_model()

    assert status.valid is False
    assert status.status == "registry-mismatched"
    assert status.reason == "validation_metrics_mismatch"
    assert status.registry_metadata_matches_joblib is False


def test_predictor_refuses_rejected_active_model_metadata(tmp_path):
    settings = Settings(_env_file=None, model_dir=str(tmp_path))
    model_path = tmp_path / "btc_model.joblib"
    metrics = _passing_metrics(net_return_pct=-0.01, promotion_reason="accepted")
    model = MLSignalModel(feature_columns=BAR_FEATURE_COLUMNS)
    model.metadata["validation_metrics"] = metrics
    model.metadata["promotion_reason"] = "accepted"
    model.save(model_path)
    (tmp_path / "registry.json").write_text(
        json.dumps(
            {
                "active_model_path": str(model_path),
                "feature_columns": BAR_FEATURE_COLUMNS,
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )

    prediction = Predictor(settings).predict(MarketDataClient.synthetic_btc_bars(140))

    assert prediction["prediction_source"] == "fallback_invalid_model"
    assert prediction["model_available"] is False
    assert prediction["active_model_valid"] is False
    assert prediction["active_model_status"] == "rejected"
    assert prediction["active_model_invalid_reason"] == "model_not_profitable_after_costs"
    assert prediction["active_model_reason"] == "model_not_profitable_after_costs"
    assert prediction["active_model_net_return_pct"] == -0.01
    assert prediction["active_model_profit_factor_net"] == 1.2
    assert prediction["active_model_number_of_trades"] == 35
    assert prediction["registry_metadata_matches_joblib"] is True


def test_active_model_with_unaccepted_promotion_reason_is_invalid(tmp_path):
    settings = Settings(_env_file=None, model_dir=str(tmp_path))
    model_path = tmp_path / "btc_model.joblib"
    metrics = _passing_metrics(net_return_pct=0.02, promotion_reason="precision_below_threshold")
    model = MLSignalModel(feature_columns=BAR_FEATURE_COLUMNS)
    model.metadata["validation_metrics"] = metrics
    model.metadata["promotion_reason"] = "precision_below_threshold"
    model.save(model_path)
    (tmp_path / "registry.json").write_text(
        json.dumps(
            {
                "active_model_path": str(model_path),
                "feature_columns": BAR_FEATURE_COLUMNS,
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )

    status = ModelRegistry(settings).validate_active_model()

    assert status.valid is False
    assert status.status == "rejected"
    assert status.reason == "promotion_reason_not_accepted"
    assert status.registry_metadata_matches_joblib is True


def test_active_model_rejected_when_profit_factor_net_is_below_threshold(tmp_path):
    settings = Settings(_env_file=None, model_dir=str(tmp_path), min_backtest_profit_factor=1.2)
    model_path = tmp_path / "btc_model.joblib"
    metrics = _passing_metrics(profit_factor_net=1.19)
    model = MLSignalModel(feature_columns=BAR_FEATURE_COLUMNS)
    model.metadata["validation_metrics"] = metrics
    model.metadata["promotion_reason"] = "accepted"
    model.save(model_path)
    (tmp_path / "registry.json").write_text(
        json.dumps(
            {
                "active_model_path": str(model_path),
                "model_version": model_path.stem,
                "feature_columns": BAR_FEATURE_COLUMNS,
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )

    status = ModelRegistry(settings).validate_active_model()

    assert status.valid is False
    assert status.status == "rejected"
    assert status.reason == "profit_factor_net_too_low"


def test_active_model_rejected_when_trade_count_is_too_low(tmp_path):
    settings = Settings(_env_file=None, model_dir=str(tmp_path), min_backtest_trades=30)
    model_path = tmp_path / "btc_model.joblib"
    metrics = _passing_metrics(number_of_trades=29)
    model = MLSignalModel(feature_columns=BAR_FEATURE_COLUMNS)
    model.metadata["validation_metrics"] = metrics
    model.metadata["promotion_reason"] = "accepted"
    model.save(model_path)
    (tmp_path / "registry.json").write_text(
        json.dumps(
            {
                "active_model_path": str(model_path),
                "model_version": model_path.stem,
                "feature_columns": BAR_FEATURE_COLUMNS,
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )

    status = ModelRegistry(settings).validate_active_model()

    assert status.valid is False
    assert status.status == "rejected"
    assert status.reason == "not_enough_backtest_trades"


def test_active_model_version_mismatch_marks_registry_mismatched(tmp_path):
    settings = Settings(_env_file=None, model_dir=str(tmp_path))
    model_path = tmp_path / "btc_model.joblib"
    metrics = _passing_metrics()
    model = MLSignalModel(feature_columns=BAR_FEATURE_COLUMNS)
    model.metadata["validation_metrics"] = metrics
    model.metadata["promotion_reason"] = "accepted"
    model.metadata["model_version"] = "different_model_version"
    model.save(model_path)
    (tmp_path / "registry.json").write_text(
        json.dumps(
            {
                "active_model_path": str(model_path),
                "model_version": model_path.stem,
                "feature_columns": BAR_FEATURE_COLUMNS,
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )

    status = ModelRegistry(settings).validate_active_model()

    assert status.valid is False
    assert status.status == "registry-mismatched"
    assert status.reason == "model_metadata_version_mismatch"
    assert status.registry_metadata_matches_joblib is False
