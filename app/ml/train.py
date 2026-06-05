from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.config import Settings, get_settings
from app.backtest.scalping import walk_forward_fee_aware_backtest
from app.data.dataset_builder import training_feature_columns
from app.ml.model import MLSignalModel, tune_tree_params
from app.ml.registry import ModelRegistry
from app.ml.training_diagnostics import (
    build_training_dataset_with_diagnostics,
    buy_positive_label_guard_failed,
    conservative_promotion_label_config,
    current_production_scalping_label_config,
    required_exit_return_pct,
    training_scalping_label_config,
)
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
    take_profit_pct = settings.scalping_label_take_profit_pct if settings.scalping_mode_enabled else settings.take_profit_pct
    stop_loss_pct = settings.scalping_label_stop_loss_pct if settings.scalping_mode_enabled else settings.stop_loss_pct
    threshold = settings.scalping_buy_probability_floor if settings.scalping_mode_enabled else settings.min_buy_probability
    feature_columns = training_feature_columns(settings.scalping_mode_enabled)
    dataset, data_diagnostics = build_training_dataset_with_diagnostics(bars, settings)
    class_counts = {
        int(label): int(count)
        for label, count in dataset["buy_quality_label"].astype(int).value_counts().sort_index().items()
    }
    data_metrics = _training_data_metrics(data_diagnostics, class_counts=class_counts)
    if buy_positive_label_guard_failed(data_diagnostics, settings):
        metrics = {
            **data_metrics,
            "valid": False,
            "reason": "buy_positive_labels_too_low",
            "promotion_reason": "buy_positive_labels_too_low",
            "fee_aware_backtest_valid": False,
            "fee_aware_backtest_reason": "not_run_buy_positive_labels_too_low",
        }
        logger.event("model_rejection", model_path=None, metrics=metrics, reason=metrics["reason"])
        _record_model_run("none", "rejected", metrics)
        return {"model_path": None, "accepted": False, "reason": metrics["reason"], "metrics": metrics, "registry": None}

    if len(dataset) < settings.min_training_rows:
        metrics = {
            **data_metrics,
            "valid": False,
            "reason": "not_enough_rows",
            "promotion_reason": "not_enough_rows",
            "min_training_rows": int(settings.min_training_rows),
            "fee_aware_backtest_valid": False,
            "fee_aware_backtest_reason": "not_run_not_enough_rows",
        }
        logger.event("model_rejection", model_path=None, metrics=metrics, reason=metrics["reason"])
        _record_model_run("none", "rejected", metrics)
        return {"model_path": None, "accepted": False, "reason": metrics["reason"], "metrics": metrics, "registry": None}

    if len(class_counts) < 2:
        metrics = {
            **data_metrics,
            "valid": False,
            "reason": "target_class_diversity_too_low",
            "promotion_reason": "target_class_diversity_too_low",
            "fee_aware_backtest_valid": False,
            "fee_aware_backtest_reason": "not_run_target_class_diversity_too_low",
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
        feature_columns=feature_columns,
        sell_threshold=settings.min_sell_probability,
    )
    metrics.update(data_metrics)
    fee_metrics = walk_forward_fee_aware_backtest(
        dataset,
        settings,
        min_train_rows=settings.min_training_rows,
        threshold=threshold,
        feature_columns=feature_columns,
    )
    metrics.update(_promotion_performance_metrics(fee_metrics, starting_equity=starting_equity))
    metrics.update(data_metrics)
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
        max_ambiguous_candle_ratio=settings.max_backtest_ambiguous_candle_ratio,
        require_positive_net_return=settings.model_promotion_require_positive_net_return,
        min_buy_positive_labels=settings.min_buy_positive_labels,
        min_buy_positive_label_pct=settings.min_buy_positive_label_pct,
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

    tuned_params = tune_tree_params(dataset, feature_columns) if settings.optuna_enabled else {}
    model = MLSignalModel(feature_columns=feature_columns).train(dataset, tuned_params=tuned_params, model_version=version)
    model.metadata["validation_metrics"] = metrics
    model.metadata["promotion_reason"] = reason
    model.metadata["model_version"] = version
    model.metadata["tuned_params"] = tuned_params
    model.metadata["dataset_metadata"] = dataset_metadata(settings, feature_columns=feature_columns)
    model_path = Path(settings.model_dir) / f"{version}.joblib"
    model.save(model_path)
    training_end = datetime.now(UTC).isoformat()
    registry = None
    if accepted:
        registry = ModelRegistry(settings).promote(
            model_path=str(model_path),
            feature_columns=feature_columns,
            metrics=metrics,
            thresholds={
                "min_buy_probability": threshold,
                "min_sell_probability": settings.min_sell_probability,
                "confidence_gap_required": settings.confidence_gap_required,
            },
            training_start=training_start,
            training_end=training_end,
            supports_independent_sell_probability=model.supports_independent_sell_probability,
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
        "number_of_trades": fee_metrics.get("number_of_trades"),
        "number_of_canceled_orders": fee_metrics.get("number_of_canceled_orders"),
        "partial_fill_count": fee_metrics.get("partial_fill_count"),
        "evaluated_signal_count": fee_metrics.get("evaluated_signal_count"),
        "ambiguous_candle_count": fee_metrics.get("ambiguous_candle_count"),
        "ambiguous_candle_ratio": fee_metrics.get("ambiguous_candle_ratio"),
        "win_rate_net": fee_metrics.get("win_rate_net"),
        "average_hold_bars": fee_metrics.get("average_hold_bars"),
        "fees_paid_estimate": fee_metrics.get("fees_paid_estimate"),
        "slippage_paid_estimate": fee_metrics.get("slippage_paid_estimate"),
        "spread_paid_estimate": fee_metrics.get("spread_paid_estimate"),
        "total_fees": fee_metrics.get("total_fees"),
        "total_slippage": fee_metrics.get("total_slippage"),
        "total_spread_cost": fee_metrics.get("total_spread_cost"),
        "max_drawdown_pct": fee_metrics.get("max_drawdown_pct", fee_metrics.get("max_drawdown")),
        "profit_factor_net": fee_metrics.get("profit_factor_net"),
        "average_trade_pnl": fee_metrics.get("average_trade_pnl"),
        "best_trade_pnl": fee_metrics.get("best_trade_pnl"),
        "worst_trade_pnl": fee_metrics.get("worst_trade_pnl"),
        "fee_aware_backtest_valid": fee_metrics.get("valid"),
        "fee_aware_backtest_reason": fee_metrics.get("reason"),
    }


def _backtest_fee_bps(settings: Settings) -> float:
    return settings.taker_fee_bps if settings.backtest_use_taker_fees else settings.maker_fee_bps


def dataset_metadata(settings: Settings, *, feature_columns: list[str]) -> dict:
    scalping_mode_enabled = settings.scalping_mode_enabled
    training_label_config = training_scalping_label_config(settings) if scalping_mode_enabled else None
    production_label_config = current_production_scalping_label_config(settings) if scalping_mode_enabled else None
    promotion_label_config = conservative_promotion_label_config(settings) if scalping_mode_enabled else None
    return {
        "feature_set_name": "scalping_bar_features_v1" if scalping_mode_enabled else "legacy_bar_features_v1",
        "feature_columns": list(feature_columns),
        "horizon_bars": settings.label_horizon_bars if scalping_mode_enabled else 12,
        "take_profit_pct": settings.scalping_label_take_profit_pct if scalping_mode_enabled else settings.take_profit_pct,
        "stop_loss_pct": settings.scalping_label_stop_loss_pct if scalping_mode_enabled else settings.stop_loss_pct,
        "fee_bps_per_side": settings.label_fee_bps_per_side if scalping_mode_enabled else 0.0,
        "slippage_bps_per_side": settings.label_slippage_bps_per_side if scalping_mode_enabled else 0.0,
        "spread_cost_pct": (settings.label_spread_bps / 10_000) if scalping_mode_enabled else 0.0,
        "min_net_profit_pct": settings.label_min_net_profit_pct if scalping_mode_enabled else 0.0,
        "entry_target_col": "buy_quality_label",
        "exit_target_col": "exit_quality_label",
        "sell_quality_label_semantics": "compatibility alias for exit_quality_label; closes existing long only",
        "training_label_assumptions": _label_config_metadata(training_label_config),
        "current_production_label_assumptions": _label_config_metadata(production_label_config),
        "conservative_promotion_assumptions": _label_config_metadata(promotion_label_config),
        "promotion_backtest_costs": {
            "fee_bps_per_side": _backtest_fee_bps(settings),
            "slippage_bps_per_side": settings.slippage_bps,
            "max_spread_bps": settings.max_spread_bps,
            "min_net_return_pct": settings.min_backtest_net_return_pct,
            "min_profit_factor_net": settings.min_backtest_profit_factor,
            "min_trades": settings.min_backtest_trades,
        },
    }


def _label_config_metadata(config) -> dict | None:
    if config is None:
        return None
    return {
        "name": config.name,
        "horizon_bars": config.horizon_bars,
        "take_profit_pct": config.take_profit_pct,
        "stop_loss_pct": config.stop_loss_pct,
        "trailing_stop_pct": config.trailing_stop_pct,
        "trailing_stop_arm_profit_pct": config.trailing_stop_arm_profit_pct,
        "fee_bps_per_side": config.fee_bps_per_side,
        "slippage_bps_per_side": config.slippage_bps_per_side,
        "spread_bps": config.spread_cost_pct * 10_000,
        "min_net_exit_profit_pct": config.min_net_exit_profit_pct,
        "exit_profit_buffer_bps": config.exit_profit_buffer_bps,
        "estimated_required_exit_return_pct": required_exit_return_pct(config),
    }


def _training_data_metrics(data_diagnostics: dict, *, class_counts: dict[int, int]) -> dict:
    metrics = dict(data_diagnostics)
    metrics["rows"] = int(data_diagnostics.get("trainable_rows", 0))
    metrics["class_counts"] = class_counts
    return metrics


def _record_model_run(model_version: str, status: str, metrics: dict) -> None:
    try:
        from app.db.database import SessionLocal, init_db
        from app.db.repository import Repository

        init_db()
        with SessionLocal() as db:
            Repository(db).add_model_run(model_version=model_version, status=status, metrics=metrics)
    except Exception:
        pass
