from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import pandas as pd

from app.config import Settings
from app.data.dataset_builder import ML_TARGET_COLUMNS, training_feature_columns
from app.data.feature_engineering import add_features
from app.data.scalping_features import SCALPING_BAR_FEATURE_COLUMNS, build_scalping_features
from app.ml.labels import (
    BUY_QUALITY_LABEL,
    EXIT_QUALITY_LABEL,
    SELL_QUALITY_LABEL,
    net_profit_scalping_labels,
    triple_barrier_labels,
)


@dataclass(frozen=True)
class ScalpingLabelConfig:
    name: str
    horizon_bars: int
    take_profit_pct: float
    stop_loss_pct: float
    trailing_stop_pct: float
    trailing_stop_arm_profit_pct: float
    fee_bps_per_side: float
    slippage_bps_per_side: float
    spread_cost_pct: float
    min_net_exit_profit_pct: float
    exit_profit_buffer_bps: float


def build_training_dataset_with_diagnostics(bars: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict[str, Any]]:
    featured, labeled = build_feature_label_frames(bars, settings)
    feature_columns = training_feature_columns(settings.scalping_mode_enabled)
    required_columns = feature_columns + ML_TARGET_COLUMNS
    trainable = _dropna_required(labeled, required_columns)
    summary = label_diagnosis_summary(
        bars=bars,
        featured=featured,
        labeled=labeled,
        trainable=trainable,
        settings=settings,
        required_columns=required_columns,
    )
    return trainable, summary


def build_feature_label_frames(
    bars: pd.DataFrame,
    settings: Settings,
    *,
    scalping_mode_enabled: bool | None = None,
    scalping_label_config: ScalpingLabelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scalping_mode = settings.scalping_mode_enabled if scalping_mode_enabled is None else scalping_mode_enabled
    if scalping_mode:
        featured = build_scalping_features(bars)
        labeled = apply_scalping_label_config(
            featured,
            scalping_label_config or training_scalping_label_config(settings),
        )
        return featured, labeled

    featured = add_features(bars)
    labeled = triple_barrier_labels(
        featured,
        horizon_bars=12,
        take_profit_pct=settings.take_profit_pct,
        stop_loss_pct=settings.stop_loss_pct,
    )
    return featured, labeled


def label_diagnosis_summary(
    *,
    bars: pd.DataFrame,
    featured: pd.DataFrame,
    labeled: pd.DataFrame,
    trainable: pd.DataFrame,
    settings: Settings,
    required_columns: list[str],
) -> dict[str, Any]:
    buy_distribution = label_distribution(trainable, BUY_QUALITY_LABEL)
    exit_distribution = label_distribution(trainable, EXIT_QUALITY_LABEL)
    sell_distribution = label_distribution(trainable, SELL_QUALITY_LABEL)
    buy_positive_count = buy_distribution.get(1, 0)
    exit_positive_count = exit_distribution.get(1, 0)
    trainable_rows = len(trainable)
    training_config = training_scalping_label_config(settings)
    production_config = current_production_scalping_label_config(settings)
    conservative_config = conservative_promotion_label_config(settings)
    return {
        "raw_bars": int(len(bars)),
        "first_timestamp": _timestamp_value(bars, first=True),
        "latest_timestamp": _timestamp_value(bars, first=False),
        "featured_rows": int(len(featured)),
        "labeled_rows": int(len(labeled)),
        "trainable_rows": int(trainable_rows),
        "required_training_columns": list(required_columns),
        "missing_required_training_columns": [column for column in required_columns if column not in labeled.columns],
        "top_nan_columns": top_nan_columns(labeled, required_columns),
        "buy_quality_label_distribution": buy_distribution,
        "entry_quality_label_distribution": label_distribution(trainable, "entry_quality_label"),
        "exit_quality_label_distribution": exit_distribution,
        "sell_quality_label_distribution": sell_distribution,
        "buy_positive_label_count": int(buy_positive_count),
        "buy_positive_label_pct": _positive_pct(buy_positive_count, trainable_rows),
        "exit_positive_label_count": int(exit_positive_count),
        "exit_positive_label_pct": _positive_pct(exit_positive_count, trainable_rows),
        "sell_positive_label_count": int(exit_positive_count),
        "sell_positive_label_pct": _positive_pct(exit_positive_count, trainable_rows),
        "buy_sell_imbalance_ratio": _positive_ratio(buy_positive_count, exit_positive_count),
        "buy_to_exit_positive_ratio": _positive_ratio(buy_positive_count, exit_positive_count),
        "buy_exit_reason_distribution": reason_distribution(trainable, "buy_exit_reason"),
        "current_production_label_assumptions": _config_payload(production_config),
        "training_label_assumptions": _config_payload(training_config),
        "conservative_promotion_assumptions": _config_payload(conservative_config),
        "estimated_required_exit_return_pct": required_exit_return_pct(training_config),
        "training_estimated_required_exit_return_pct": required_exit_return_pct(training_config),
        "current_production_estimated_required_exit_return_pct": required_exit_return_pct(production_config),
        "conservative_promotion_estimated_required_exit_return_pct": required_exit_return_pct(conservative_config),
        "min_buy_positive_labels": int(settings.min_buy_positive_labels),
        "min_buy_positive_label_pct": float(settings.min_buy_positive_label_pct),
    }


def label_distribution(df: pd.DataFrame, column: str) -> dict[int, int]:
    if column not in df.columns or df.empty:
        return {0: 0, 1: 0}
    counts = df[column].dropna().astype(int).value_counts().sort_index()
    distribution = {0: 0, 1: 0}
    distribution.update({int(label): int(count) for label, count in counts.items()})
    return distribution


def reason_distribution(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns or df.empty:
        return {}
    counts = df[column].fillna("missing").astype(str).value_counts().sort_index()
    return {str(reason): int(count) for reason, count in counts.items()}


def top_nan_columns(df: pd.DataFrame, required_columns: list[str], *, limit: int = 10) -> list[dict[str, Any]]:
    if df.empty:
        return []
    counts: dict[str, int] = {}
    for column in required_columns:
        if column not in df.columns:
            counts[column] = len(df)
        else:
            counts[column] = int(df[column].isna().sum())
    return [
        {"column": column, "nan_count": count}
        for column, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
        if count > 0
    ]


def training_scalping_label_config(settings: Settings) -> ScalpingLabelConfig:
    return ScalpingLabelConfig(
        name="training_label",
        horizon_bars=int(settings.label_horizon_bars),
        take_profit_pct=float(settings.scalping_label_take_profit_pct),
        stop_loss_pct=float(settings.scalping_label_stop_loss_pct),
        trailing_stop_pct=float(settings.scalping_trailing_stop_pct),
        trailing_stop_arm_profit_pct=float(settings.trailing_stop_arm_profit_pct),
        fee_bps_per_side=float(settings.label_fee_bps_per_side),
        slippage_bps_per_side=float(settings.label_slippage_bps_per_side),
        spread_cost_pct=float(settings.label_spread_bps) / 10_000,
        min_net_exit_profit_pct=float(settings.label_min_net_profit_pct),
        exit_profit_buffer_bps=0.0,
    )


def current_production_scalping_label_config(settings: Settings) -> ScalpingLabelConfig:
    return ScalpingLabelConfig(
        name="current_production_label",
        horizon_bars=int(settings.scalping_label_horizon_bars),
        take_profit_pct=float(settings.scalping_label_take_profit_pct),
        stop_loss_pct=float(settings.scalping_label_stop_loss_pct),
        trailing_stop_pct=float(settings.scalping_trailing_stop_pct),
        trailing_stop_arm_profit_pct=float(settings.trailing_stop_arm_profit_pct),
        fee_bps_per_side=backtest_fee_bps(settings),
        slippage_bps_per_side=float(settings.slippage_bps),
        spread_cost_pct=float(settings.max_spread_bps) / 10_000,
        min_net_exit_profit_pct=float(settings.scalping_label_min_net_profit_pct),
        exit_profit_buffer_bps=float(settings.exit_profit_buffer_bps),
    )


def current_scalping_label_config(settings: Settings) -> ScalpingLabelConfig:
    return current_production_scalping_label_config(settings)


def conservative_promotion_label_config(settings: Settings) -> ScalpingLabelConfig:
    return ScalpingLabelConfig(
        name="conservative_promotion_diagnostic",
        horizon_bars=int(settings.label_horizon_bars),
        take_profit_pct=float(settings.scalping_label_take_profit_pct),
        stop_loss_pct=float(settings.scalping_label_stop_loss_pct),
        trailing_stop_pct=float(settings.scalping_trailing_stop_pct),
        trailing_stop_arm_profit_pct=float(settings.trailing_stop_arm_profit_pct),
        fee_bps_per_side=backtest_fee_bps(settings),
        slippage_bps_per_side=float(settings.slippage_bps),
        spread_cost_pct=float(settings.max_spread_bps) / 10_000,
        min_net_exit_profit_pct=float(settings.min_backtest_net_return_pct),
        exit_profit_buffer_bps=0.0,
    )


def label_config_comparison(bars: pd.DataFrame, settings: Settings) -> list[dict[str, Any]]:
    featured = build_scalping_features(bars)
    production = current_production_scalping_label_config(settings)
    training = training_scalping_label_config(settings)
    conservative = conservative_promotion_label_config(settings)
    configs = [
        production,
        training,
        conservative,
        replace(
            training,
            name="loose_diagnostic",
            horizon_bars=max(int(training.horizon_bars), 6),
            take_profit_pct=min(float(training.take_profit_pct), 0.0008),
            stop_loss_pct=max(float(training.stop_loss_pct), 0.0012),
            fee_bps_per_side=float(settings.maker_fee_bps),
            slippage_bps_per_side=0.0,
            spread_cost_pct=0.0,
            min_net_exit_profit_pct=0.0,
            exit_profit_buffer_bps=0.0,
        ),
    ]
    return [_summarize_scalping_config(featured, config) for config in configs]


def apply_scalping_label_config(featured: pd.DataFrame, config: ScalpingLabelConfig) -> pd.DataFrame:
    return net_profit_scalping_labels(
        featured,
        horizon_bars=config.horizon_bars,
        take_profit_pct=config.take_profit_pct,
        stop_loss_pct=config.stop_loss_pct,
        trailing_stop_pct=config.trailing_stop_pct,
        trailing_stop_arm_profit_pct=config.trailing_stop_arm_profit_pct,
        fee_bps_per_side=config.fee_bps_per_side,
        slippage_bps_per_side=config.slippage_bps_per_side,
        spread_cost_pct=config.spread_cost_pct,
        min_net_exit_profit_pct=config.min_net_exit_profit_pct,
        exit_profit_buffer_bps=config.exit_profit_buffer_bps,
    )


def estimated_required_exit_return_pct(settings: Settings) -> float:
    return required_exit_return_pct(training_scalping_label_config(settings))


def required_exit_return_pct(config: ScalpingLabelConfig) -> float:
    return (
        2 * max(0.0, float(config.fee_bps_per_side)) / 10_000
        + 2 * max(0.0, float(config.slippage_bps_per_side)) / 10_000
        + max(0.0, float(config.spread_cost_pct))
        + max(0.0, float(config.min_net_exit_profit_pct))
        + max(0.0, float(config.exit_profit_buffer_bps)) / 10_000
    )


def backtest_fee_bps(settings: Settings) -> float:
    return float(settings.taker_fee_bps if settings.backtest_use_taker_fees else settings.maker_fee_bps)


def buy_positive_label_guard_failed(summary: dict[str, Any], settings: Settings) -> bool:
    return (
        int(summary.get("buy_positive_label_count", 0)) < int(settings.min_buy_positive_labels)
        or float(summary.get("buy_positive_label_pct", 0.0)) < float(settings.min_buy_positive_label_pct)
    )


def buy_positive_label_warning(summary: dict[str, Any], settings: Settings) -> str | None:
    if not buy_positive_label_guard_failed(summary, settings):
        return None
    return (
        "buy positive labels are too low: "
        f"count={summary.get('buy_positive_label_count', 0)} "
        f"min_count={settings.min_buy_positive_labels} "
        f"pct={float(summary.get('buy_positive_label_pct', 0.0)):.4f} "
        f"min_pct={settings.min_buy_positive_label_pct:.4f}"
    )


def next_recommended_action(result: dict[str, Any], settings: Settings) -> str:
    metrics = result.get("metrics") or {}
    reason = result.get("reason") or metrics.get("promotion_reason") or metrics.get("reason")
    if result.get("accepted"):
        return "Accepted into the active paper-trading model registry; continue monitoring paper-only runtime behavior."
    if reason == "buy_positive_labels_too_low":
        return "Do not promote. Collect more BTC/USD bars for the configured timeframe or inspect LABEL_* training assumptions before adding strategies."
    if reason in {"not_enough_rows", "validation_rows_too_low"}:
        return "Do not promote. Increase BTC/USD historical bars or lower MIN_TRAINING_ROWS only for diagnostics, not production."
    if reason == "target_class_diversity_too_low":
        return "Do not promote. Labels are effectively one-class; revisit fee-aware label thresholds and market sample coverage."
    if metrics.get("fee_aware_backtest_valid") is False:
        return "Do not promote. Fix the fee-aware backtest rejection before considering the model active."
    if reason:
        return f"Do not promote. Investigate rejection reason '{reason}' with diagnose_labels before changing strategy logic."
    return "Do not promote. Training did not produce an accepted model; inspect diagnostics before changing strategy logic."


def _summarize_scalping_config(featured: pd.DataFrame, config: ScalpingLabelConfig) -> dict[str, Any]:
    labeled = apply_scalping_label_config(featured, config)
    required_columns = SCALPING_BAR_FEATURE_COLUMNS + ML_TARGET_COLUMNS
    trainable = _dropna_required(labeled, required_columns)
    buy_distribution = label_distribution(trainable, BUY_QUALITY_LABEL)
    exit_distribution = label_distribution(trainable, EXIT_QUALITY_LABEL)
    exit_positive_count = int(exit_distribution.get(1, 0))
    buy_positive_count = int(buy_distribution.get(1, 0))
    return {
        "config": config.name,
        "horizon_bars": int(config.horizon_bars),
        "fee_bps_per_side": float(config.fee_bps_per_side),
        "slippage_bps_per_side": float(config.slippage_bps_per_side),
        "spread_bps": float(config.spread_cost_pct) * 10_000,
        "min_net_profit_pct": float(config.min_net_exit_profit_pct),
        "exit_profit_buffer_bps": float(config.exit_profit_buffer_bps),
        "estimated_required_exit_return_pct": required_exit_return_pct(config),
        "trainable_rows": int(len(trainable)),
        "buy_1": buy_positive_count,
        "buy_0": int(buy_distribution.get(0, 0)),
        "buy_1_pct": _positive_pct(buy_positive_count, len(trainable)),
        "exit_1": exit_positive_count,
        "exit_0": int(exit_distribution.get(0, 0)),
        "exit_1_pct": _positive_pct(exit_positive_count, len(trainable)),
        "sell_1": exit_positive_count,
        "sell_0": int(exit_distribution.get(0, 0)),
        "sell_1_pct": _positive_pct(exit_positive_count, len(trainable)),
        "buy_sell_imbalance_ratio": _positive_ratio(buy_positive_count, exit_positive_count),
        "buy_exit_reason_distribution": reason_distribution(trainable, "buy_exit_reason"),
    }


def _dropna_required(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        return df.iloc[0:0].copy()
    return df.dropna(subset=required_columns).reset_index(drop=True)


def _positive_pct(positive_count: int, total: int) -> float:
    return float(positive_count / total) if total else 0.0


def _positive_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _config_payload(config: ScalpingLabelConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["spread_bps"] = float(config.spread_cost_pct) * 10_000
    payload["estimated_required_exit_return_pct"] = required_exit_return_pct(config)
    return payload


def _timestamp_value(bars: pd.DataFrame, *, first: bool) -> str | None:
    if bars.empty or "timestamp" not in bars.columns:
        return None
    index = 0 if first else -1
    return pd.Timestamp(bars["timestamp"].iloc[index]).isoformat()
