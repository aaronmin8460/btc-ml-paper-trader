from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.config import Settings, get_settings
from app.backtest.scalping import walk_forward_fee_aware_backtest
from app.data.dataset_builder import build_training_dataset
from app.data.feature_engineering import FEATURE_COLUMNS
from app.ml.model import MLSignalModel, tune_tree_params
from app.ml.registry import ModelRegistry
from app.ml.validation import promotion_decision, walk_forward_validate
from app.monitoring.logger import get_logger


def train_model_from_bars(
    bars: pd.DataFrame,
    settings: Settings | None = None,
    *,
    starting_equity: float | None = None,
) -> dict:
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
        _record_model_run("none", "rejected", metrics)
        return {"model_path": None, "accepted": False, "reason": metrics["reason"], "metrics": metrics, "registry": None}

    metrics = walk_forward_validate(
        dataset,
        min_train_rows=settings.min_training_rows,
        threshold=threshold,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )
    fee_metrics = walk_forward_fee_aware_backtest(
        dataset,
        settings,
        min_train_rows=settings.min_training_rows,
        threshold=threshold,
    )
    metrics.update(_promotion_performance_metrics(fee_metrics, starting_equity=starting_equity))
    accepted, reason = promotion_decision(
        metrics,
        min_rows=settings.min_training_rows,
        min_precision=settings.min_precision_for_promotion,
        max_drawdown=settings.max_validation_drawdown_pct,
        max_trade_fraction=settings.max_trade_fraction,
        min_net_return_pct=settings.min_backtest_net_return_pct,
        max_backtest_drawdown_pct=settings.max_backtest_drawdown_pct,
        min_backtest_profit_factor=settings.min_backtest_profit_factor,
        min_backtest_trades=settings.min_backtest_trades,
        require_positive_net_return=settings.model_promotion_require_positive_net_return,
    )
    metrics["promotion_reason"] = reason
    version = datetime.now(UTC).strftime("btc_model_%Y%m%dT%H%M%SZ")
    if metrics.get("valid") is False and metrics.get("reason") in {
        "target_class_diversity_too_low",
        "no_trainable_validation_folds",
    }:
        logger.event("model_rejection", model_path=None, metrics=metrics, reason=metrics["reason"])
        _record_model_run(version, "rejected", metrics)
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
    _record_model_run(version, "accepted" if accepted else "rejected", metrics)
    return {"model_path": str(model_path), "accepted": accepted, "reason": reason, "metrics": metrics, "registry": registry}


def _promotion_performance_metrics(fee_metrics: dict, *, starting_equity: float | None = None) -> dict:
    net_return_pct = fee_metrics.get("net_return_pct", fee_metrics.get("net_return"))
    gross_return_pct = fee_metrics.get("gross_return_pct", fee_metrics.get("gross_return"))
    ending_equity = fee_metrics.get("ending_equity")
    if starting_equity is not None and net_return_pct is not None:
        ending_equity = float(starting_equity) * (1 + float(net_return_pct))
    return {
        "starting_equity": starting_equity if starting_equity is not None else fee_metrics.get("starting_equity"),
        "ending_equity": ending_equity,
        "gross_return_pct": gross_return_pct,
        "net_return_pct": net_return_pct,
        "fees_paid_estimate": fee_metrics.get("fees_paid_estimate"),
        "slippage_paid_estimate": fee_metrics.get("slippage_paid_estimate"),
        "max_drawdown_pct": fee_metrics.get("max_drawdown_pct", fee_metrics.get("max_drawdown")),
        "profit_factor_net": fee_metrics.get("profit_factor_net"),
        "average_trade_pnl": fee_metrics.get("average_trade_pnl"),
        "best_trade_pnl": fee_metrics.get("best_trade_pnl"),
        "worst_trade_pnl": fee_metrics.get("worst_trade_pnl"),
        "fee_aware_backtest_valid": fee_metrics.get("valid"),
        "fee_aware_backtest_reason": fee_metrics.get("reason"),
    }


def _record_model_run(model_version: str, status: str, metrics: dict) -> None:
    try:
        from app.db.database import SessionLocal, init_db
        from app.db.repository import Repository

        init_db()
        with SessionLocal() as db:
            Repository(db).add_model_run(model_version=model_version, status=status, metrics=metrics)
    except Exception:
        pass
