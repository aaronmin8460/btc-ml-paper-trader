import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.data.dataset_builder import build_training_dataset, training_feature_columns
from app.data.feature_engineering import BAR_FEATURE_COLUMNS, RUNTIME_ONLY_FEATURE_COLUMNS
from app.data.market_data import MarketDataClient
from app.data.scalping_features import SCALPING_BAR_FEATURE_COLUMNS, SCALPING_FEATURE_COLUMNS, SCALPING_QUOTE_FEATURE_COLUMNS
from app.ml.labels import net_profit_scalping_labels
from app.ml.model import MLSignalModel
from app.ml.predict import Predictor
from app.ml.registry import ActiveModelStatus, ModelRegistry
from app.ml.train import dataset_metadata, train_model_from_bars
from app.ml.validation import promotion_decision


def test_model_training_produces_probabilities():
    bars = MarketDataClient.synthetic_btc_bars(260)
    dataset = build_training_dataset(bars)
    model = MLSignalModel(feature_columns=BAR_FEATURE_COLUMNS).train(dataset)
    probabilities = model.predict_proba(dataset.tail(5))
    assert len(probabilities) == 5
    assert all(0 <= value <= 1 for value in probabilities)


def test_model_training_returns_independent_buy_and_sell_probabilities():
    feature = np.linspace(-1, 1, 120)
    dataset = pd.DataFrame(
        {
            "feature": feature,
            "buy_quality_label": (feature > 0).astype(int),
            "sell_quality_label": (feature > 0).astype(int),
        }
    )

    model = MLSignalModel(feature_columns=["feature"]).train(dataset, model_version="two_head_test")
    rows = dataset.iloc[[10, 110]]
    buy_probabilities = model.predict_buy_proba(rows)
    sell_probabilities = model.predict_sell_proba(rows)

    assert model.supports_independent_sell_probability is True
    assert np.allclose(model.predict_proba(rows), buy_probabilities)
    assert np.allclose(sell_probabilities, buy_probabilities)
    assert not np.allclose(sell_probabilities, 1 - buy_probabilities)
    assert model.metadata["target_class_balances"] == {
        "buy_quality_label": {0: 0.5, 1: 0.5},
        "sell_quality_label": {0: 0.5, 1: 0.5},
    }
    assert model.metadata["feature_columns"] == ["feature"]
    assert model.metadata["model_version"] == "two_head_test"
    assert model.metadata["supports_independent_sell_probability"] is True


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


def test_predictor_uses_scalping_row_for_active_scalping_model(monkeypatch, tmp_path):
    class ScalpingModel:
        feature_columns = SCALPING_BAR_FEATURE_COLUMNS
        supports_independent_sell_probability = True

        def predict_buy_proba(self, row):
            assert row[SCALPING_BAR_FEATURE_COLUMNS].notna().all(axis=None)
            assert row.iloc[-1]["scalping_spread_bps"] > 0
            return np.array([0.7])

        def predict_sell_proba(self, row):
            return np.array([0.2])

    status = ActiveModelStatus("scalping-model.joblib", "accepted", True)
    monkeypatch.setattr(ModelRegistry, "load_valid_active_model", lambda self: (ScalpingModel(), status))
    settings = Settings(_env_file=None, model_dir=str(tmp_path))
    quote = {"bid_price": 99.9, "ask_price": 100.1, "bid_size": 2.0, "ask_size": 1.0}

    prediction = Predictor(settings).predict(MarketDataClient.synthetic_btc_bars(140), quote=quote)

    assert prediction["prediction_source"] == "model"
    assert prediction["sell_probability_source"] == "independent_sell_model"
    assert prediction["supports_independent_sell_probability"] is True
    assert prediction["model_available"] is True
    assert prediction["buy_probability"] == 0.7
    assert prediction["sell_probability"] == 0.2
    assert "scalping_log_return_1" in prediction["features"]


def test_predictor_uses_legacy_sell_probability_fallback_for_old_model(monkeypatch, tmp_path):
    class LegacyModel:
        feature_columns = BAR_FEATURE_COLUMNS

        def predict_proba(self, row):
            return np.array([0.7])

    status = ActiveModelStatus("legacy-model.joblib", "accepted", True)
    monkeypatch.setattr(ModelRegistry, "load_valid_active_model", lambda self: (LegacyModel(), status))
    settings = Settings(_env_file=None, model_dir=str(tmp_path))

    prediction = Predictor(settings).predict(MarketDataClient.synthetic_btc_bars(140))

    assert prediction["prediction_source"] == "legacy_sell_probability_fallback"
    assert prediction["sell_probability_source"] == "buy_probability_complement"
    assert prediction["supports_independent_sell_probability"] is False
    assert prediction["buy_probability"] == 0.7
    assert prediction["sell_probability"] == pytest.approx(0.3)


def test_training_rejects_single_class_dataset_without_crashing(tmp_path):
    bars = MarketDataClient.synthetic_btc_bars(160)
    settings = Settings(
        _env_file=None,
        min_training_rows=50,
        model_dir=str(tmp_path),
        scalping_mode_enabled=True,
        min_buy_positive_labels=0,
        min_buy_positive_label_pct=0.0,
        scalping_label_take_profit_pct=1.0,
        scalping_label_stop_loss_pct=1.0,
        scalping_trailing_stop_pct=0.0,
    )

    result = train_model_from_bars(bars, settings)

    assert result["accepted"] is False
    assert result["reason"] == "target_class_diversity_too_low"
    assert result["model_path"] is None
    assert list(tmp_path.glob("*.joblib")) == []


def test_training_rejects_when_buy_positive_labels_are_too_low(tmp_path):
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
    assert result["reason"] == "buy_positive_labels_too_low"
    assert result["metrics"]["buy_positive_label_count"] == 0
    assert result["metrics"]["fee_aware_backtest_reason"] == "not_run_buy_positive_labels_too_low"
    assert result["model_path"] is None
    assert list(tmp_path.glob("*.joblib")) == []


def test_training_rejects_when_trainable_rows_are_below_min_training_rows(tmp_path):
    bars = MarketDataClient.synthetic_btc_bars(180)
    settings = Settings(
        _env_file=None,
        min_training_rows=500,
        min_buy_positive_labels=1,
        min_buy_positive_label_pct=0.0,
        model_dir=str(tmp_path),
        scalping_mode_enabled=True,
        taker_fee_bps=0,
        slippage_bps=0,
        max_spread_bps=0,
        scalping_label_take_profit_pct=0.0001,
        scalping_label_stop_loss_pct=0.01,
        scalping_label_min_net_profit_pct=0.0,
        exit_profit_buffer_bps=0,
    )

    result = train_model_from_bars(bars, settings)

    assert result["accepted"] is False
    assert result["reason"] == "not_enough_rows"
    assert result["metrics"]["trainable_rows"] < settings.min_training_rows
    assert result["metrics"]["buy_positive_label_count"] >= settings.min_buy_positive_labels
    assert result["metrics"]["fee_aware_backtest_reason"] == "not_run_not_enough_rows"
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
    assert set(SCALPING_FEATURE_COLUMNS).issubset(dataset.columns)
    assert training_feature_columns(True) == SCALPING_BAR_FEATURE_COLUMNS
    assert dataset[SCALPING_QUOTE_FEATURE_COLUMNS].isna().all(axis=None)


def test_scalping_dataset_metadata_records_short_horizon_and_costs():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        scalping_label_horizon_bars=2,
        scalping_label_take_profit_pct=0.0014,
        scalping_label_stop_loss_pct=0.0009,
        scalping_label_min_net_profit_pct=0.0003,
        taker_fee_bps=17,
        slippage_bps=4,
        max_spread_bps=6,
    )

    metadata = dataset_metadata(settings, feature_columns=training_feature_columns(True))

    assert metadata["feature_set_name"] == "scalping_bar_features_v1"
    assert metadata["feature_columns"] == SCALPING_BAR_FEATURE_COLUMNS
    assert metadata["horizon_bars"] == 2
    assert metadata["take_profit_pct"] == 0.0014
    assert metadata["stop_loss_pct"] == 0.0009
    assert metadata["fee_bps_per_side"] == 17
    assert metadata["slippage_bps_per_side"] == 4
    assert metadata["spread_cost_pct"] == 0.0006
    assert metadata["min_net_profit_pct"] == 0.0003


def test_registry_records_independent_sell_probability_support(tmp_path):
    registry = ModelRegistry(Settings(_env_file=None, model_dir=str(tmp_path))).promote(
        model_path=str(tmp_path / "model.joblib"),
        feature_columns=["feature"],
        metrics=_passing_metrics(),
        thresholds={},
        training_start="2026-05-27T00:00:00+00:00",
        training_end="2026-05-27T00:01:00+00:00",
        supports_independent_sell_probability=True,
    )

    assert registry["supports_independent_sell_probability"] is True


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


def test_model_rejected_when_ambiguous_candle_ratio_is_too_high():
    accepted, reason = promotion_decision(
        _passing_metrics(ambiguous_candle_ratio=0.11),
        min_rows=100,
        min_precision=0.52,
        max_drawdown=0.2,
        max_trade_fraction=0.4,
        min_net_return_pct=0.001,
        max_backtest_drawdown_pct=0.01,
        min_backtest_profit_factor=1.2,
        min_backtest_trades=30,
        max_ambiguous_candle_ratio=0.10,
        require_positive_net_return=True,
    )

    assert accepted is False
    assert reason == "ambiguous_candle_ratio_too_high"


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


def test_active_model_rejected_when_ambiguous_candle_ratio_is_too_high(tmp_path):
    settings = Settings(_env_file=None, model_dir=str(tmp_path), max_backtest_ambiguous_candle_ratio=0.10)
    model_path = tmp_path / "btc_model.joblib"
    metrics = _passing_metrics(ambiguous_candle_ratio=0.11)
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
    assert status.reason == "ambiguous_candle_ratio_too_high"


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
