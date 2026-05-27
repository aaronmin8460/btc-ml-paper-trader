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
    take_profit_pct = settings.scalping_take_profit_pct if settings.scalping_mode_enabled else settings.take_profit_pct
    stop_loss_pct = settings.scalping_stop_loss_pct if settings.scalping_mode_enabled else settings.stop_loss_pct
    threshold = settings.scalping_buy_probability_floor if settings.scalping_mode_enabled else settings.min_buy_probability
    dataset = build_training_dataset(
        bars,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )
    class_counts = {
        int(label): int(count)
        for label, count in dataset["buy_quality_label"].astype(int).value_counts().sort_index().items()
    }
    if len(class_counts) < 2:
        metrics = {
            "valid": False,
            "reason": "target_class_diversity_too_low",
            "rows": len(dataset),
            "class_counts": class_counts,
        }
        logger.event("model_rejection", model_path=None, metrics=metrics, reason=metrics["reason"])
        return {"model_path": None, "accepted": False, "reason": metrics["reason"], "metrics": metrics, "registry": None}

    metrics = walk_forward_validate(
        dataset,
        min_train_rows=settings.min_training_rows,
        threshold=threshold,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )
    accepted, reason = promotion_decision(
        metrics,
        min_rows=settings.min_training_rows,
        min_precision=settings.min_precision_for_promotion,
        max_drawdown=settings.max_validation_drawdown_pct,
        max_trade_fraction=settings.max_trade_fraction,
    )
    version = datetime.now(UTC).strftime("btc_model_%Y%m%dT%H%M%SZ")
    if metrics.get("valid") is False and metrics.get("reason") in {
        "target_class_diversity_too_low",
        "no_trainable_validation_folds",
    }:
        logger.event("model_rejection", model_path=None, metrics=metrics, reason=metrics["reason"])
        return {"model_path": None, "accepted": False, "reason": metrics["reason"], "metrics": metrics, "registry": None}

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
                "min_buy_probability": threshold,
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
