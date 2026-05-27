from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.config import Settings, get_settings
from app.data.dataset_builder import build_training_dataset
from app.data.feature_engineering import FEATURE_COLUMNS
from app.ml.model import MLSignalModel, tune_tree_params
from app.ml.registry import ModelRegistry
from app.ml.validation import promotion_decision, walk_forward_validate
from app.monitoring.logger import get_logger


def train_model_from_bars(bars: pd.DataFrame, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    logger = get_logger()
    training_start = datetime.now(UTC).isoformat()
    dataset = build_training_dataset(
        bars,
        take_profit_pct=settings.take_profit_pct,
        stop_loss_pct=settings.stop_loss_pct,
    )
    metrics = walk_forward_validate(
        dataset,
        min_train_rows=settings.min_training_rows,
        threshold=settings.min_buy_probability,
    )
    accepted, reason = promotion_decision(
        metrics,
        min_rows=settings.min_training_rows,
        min_precision=settings.min_precision_for_promotion,
        max_drawdown=settings.max_validation_drawdown_pct,
        max_trade_fraction=settings.max_trade_fraction,
    )
    version = datetime.now(UTC).strftime("btc_model_%Y%m%dT%H%M%SZ")
    tuned_params = tune_tree_params(dataset, FEATURE_COLUMNS) if settings.optuna_enabled else {}
    model = MLSignalModel(feature_columns=FEATURE_COLUMNS).train(dataset, tuned_params=tuned_params)
    model.metadata["validation_metrics"] = metrics
    model.metadata["promotion_reason"] = reason
    model.metadata["tuned_params"] = tuned_params
    model_path = Path(settings.model_dir) / f"{version}.joblib"
    model.save(model_path)
    training_end = datetime.now(UTC).isoformat()
    registry = None
    if accepted:
        registry = ModelRegistry(settings).promote(
            model_path=str(model_path),
            feature_columns=FEATURE_COLUMNS,
            metrics=metrics,
            thresholds={
                "min_buy_probability": settings.min_buy_probability,
                "min_sell_probability": settings.min_sell_probability,
                "confidence_gap_required": settings.confidence_gap_required,
            },
            training_start=training_start,
            training_end=training_end,
        )
        logger.event("model_promotion", model_path=str(model_path), metrics=metrics, reason=reason)
    else:
        logger.event("model_rejection", model_path=str(model_path), metrics=metrics, reason=reason)
    return {"model_path": str(model_path), "accepted": accepted, "reason": reason, "metrics": metrics, "registry": registry}
