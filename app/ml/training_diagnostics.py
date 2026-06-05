from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from app.config import Settings
from app.data.dataset_builder import ML_TARGET_COLUMNS, training_feature_columns
from app.data.feature_engineering import add_features
from app.data.scalping_features import SCALPING_BAR_FEATURE_COLUMNS, build_scalping_features
from app.ml.labels import net_profit_scalping_labels, triple_barrier_labels


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
            scalping_label_config or current_scalping_label_config(settings),
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
    buy_distribution = label_distribution(trainable, "buy_quality_label")
    sell_distribution = label_distribution(trainable, "sell_quality_label")
    buy_positive_count = buy_distribution.get(1, 0)
    sell_positive_count = sell_distribution.get(1, 0)
    trainable_rows = len(trainable)
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
        "sell_quality_label_distribution": sell_distribution,
        "buy_positive_label_count": int(buy_positive_count),
        "buy_positive_label_pct": _positive_pct(buy_positive_count, trainable_rows),
        "sell_positive_label_count": int(sell_positive_count),
        "sell_positive_label_pct": _positive_pct(sell_positive_count, trainable_rows),
        "estimated_required_exit_return_pct": estimated_required_exit_return_pct(settings),
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


def current_scalping_label_config(settings: Settings) -> ScalpingLabelConfig:
    return ScalpingLabelConfig(
        name="current",
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


def label_config_comparison(bars: pd.DataFrame, settings: Settings) -> list[dict[str, Any]]:
    featured = build_scalping_features(bars)
    current = current_scalping_label_config(settings)
    configs = [
        current,
        replace(current, name="maker_fee_assumption", fee_bps_per_side=float(settings.maker_fee_bps)),
        replace(current, name="lower_slippage", slippage_bps_per_side=max(0.0, float(settings.slippage_bps) * 0.5)),
        replace(current, name="longer_horizon", horizon_bars=max(int(current.horizon_bars) + 3, 6)),
        replace(
            current,
            name="loose_diagnostic",
            horizon_bars=max(int(current.horizon_bars) + 3, 6),
            take_profit_pct=min(float(current.take_profit_pct), 0.0008),
            stop_loss_pct=max(float(current.stop_loss_pct), 0.0012),
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
    return (
        2 * backtest_fee_bps(settings) / 10_000
        + 2 * max(0.0, float(settings.slippage_bps)) / 10_000
        + max(0.0, float(settings.max_spread_bps)) / 10_000
        + max(0.0, float(settings.scalping_label_min_net_profit_pct))
        + max(0.0, float(settings.exit_profit_buffer_bps)) / 10_000
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
        return "Do not promote. Collect more BTC/USD bars for the configured timeframe and inspect label assumptions before adding strategies."
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
    buy_distribution = label_distribution(trainable, "buy_quality_label")
    sell_distribution = label_distribution(trainable, "sell_quality_label")
    return {
        "config": config.name,
        "trainable_rows": int(len(trainable)),
        "buy_1": int(buy_distribution.get(1, 0)),
        "buy_0": int(buy_distribution.get(0, 0)),
        "buy_1_pct": _positive_pct(buy_distribution.get(1, 0), len(trainable)),
        "sell_1": int(sell_distribution.get(1, 0)),
        "sell_0": int(sell_distribution.get(0, 0)),
        "sell_1_pct": _positive_pct(sell_distribution.get(1, 0), len(trainable)),
    }


def _dropna_required(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        return df.iloc[0:0].copy()
    return df.dropna(subset=required_columns).reset_index(drop=True)


def _positive_pct(positive_count: int, total: int) -> float:
    return float(positive_count / total) if total else 0.0


def _timestamp_value(bars: pd.DataFrame, *, first: bool) -> str | None:
    if bars.empty or "timestamp" not in bars.columns:
        return None
    index = 0 if first else -1
    return pd.Timestamp(bars["timestamp"].iloc[index]).isoformat()
