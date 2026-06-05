from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.backtest.scalping import (
    _annotate_strategy_candidates,
    _walk_forward_prediction_frame,
    calculate_fee_aware_metrics,
    estimated_round_trip_execution_cost_pct,
    minimum_take_profit_net_positive_pct,
    promotion_required_return_pct,
    trade_economics_summary,
)
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.data.dataset_builder import build_training_dataset, training_feature_columns
from app.data.market_data import MarketDataClient


STRATEGY_NAMES = ("mean_reversion_scalping", "momentum_breakout")
MIN_SWEEP_TRADES = 20
MAX_SWEEP_DRAWDOWN_PCT = 0.03
MIN_SWEEP_PROFIT_FACTOR_NET = 1.05
MAX_SWEEP_AMBIGUOUS_CANDLE_RATIO = 0.10
MAX_SINGLE_TRADE_RETURN_SHARE = 0.60
MAX_SINGLE_SPLIT_RETURN_SHARE = 0.60

ML_CONFIRMATION_SWEEP = {
    "SCALPING_BUY_PROBABILITY_FLOOR": ("scalping_buy_probability_floor", [0.54, 0.56, 0.58, 0.60]),
    "SCALPING_CONFIDENCE_GAP_REQUIRED": ("scalping_confidence_gap_required", [0.04, 0.06, 0.08, 0.10]),
    "SCALPING_SELL_PROBABILITY_FLOOR": ("scalping_sell_probability_floor", [0.52, 0.55, 0.58]),
    "SCALPING_EXIT_CONFIDENCE_GAP_REQUIRED": ("scalping_exit_confidence_gap_required", [0.03, 0.04, 0.06]),
}

ECONOMIC_VIABILITY_SWEEP = {
    "SCALPING_TAKE_PROFIT_PCT": ("scalping_take_profit_pct", [0.003, 0.005, 0.0075, 0.01]),
    "SCALPING_STOP_LOSS_PCT": ("scalping_stop_loss_pct", [0.002, 0.003, 0.005]),
    "LABEL_HORIZON_BARS": ("label_horizon_bars", [6, 10, 15, 30]),
    "MAX_SPREAD_BPS": ("max_spread_bps", [2, 4, 6]),
    "SLIPPAGE_BPS": ("slippage_bps", [2, 5, 10]),
    "BACKTEST_USE_TAKER_FEES": ("backtest_use_taker_fees", [True, False]),
}

REGIME_FILTER_SWEEP = {
    "REGIME_NO_TRADE_VOLATILITY_THRESHOLD": (
        "regime_no_trade_volatility_threshold",
        [0.012, 0.015, 0.020, 0.025],
    ),
    "REGIME_NO_TRADE_SHORT_RETURN_THRESHOLD": (
        "regime_no_trade_short_return_threshold",
        [0.008, 0.012, 0.015, 0.020],
    ),
    "REGIME_TREND_STRENGTH_THRESHOLD": ("regime_trend_strength_threshold", [0.40, 0.60, 0.80, 1.00]),
    "REGIME_BREAKOUT_THRESHOLD": ("regime_breakout_threshold", [0.0005, 0.0010, 0.0015, 0.0020]),
    "REGIME_MEAN_REVERSION_SHORT_RETURN_THRESHOLD": (
        "regime_mean_reversion_short_return_threshold",
        [0.0015, 0.0025, 0.0030, 0.0040],
    ),
    "REGIME_MEAN_REVERSION_LOW_BREAKOUT_THRESHOLD": (
        "regime_mean_reversion_low_breakout_threshold",
        [0.0030, 0.0045, 0.0060, 0.0080],
    ),
}

PARAMETER_GROUPS = {
    "economic_viability": ECONOMIC_VIABILITY_SWEEP,
}


@dataclass(frozen=True)
class ParameterSet:
    parameter_set_id: str
    group: str
    description: str
    overrides: dict[str, Any]
    env_overrides: dict[str, Any]


async def main() -> None:
    settings = get_settings()
    max_configs = int(os.getenv("SWEEP_MAX_CONFIGS", "32"))
    folds = int(os.getenv("SWEEP_FOLDS", "2"))
    bar_limit = int(os.getenv("SWEEP_BAR_LIMIT", str(max(1500, settings.min_training_rows + 500))))
    bars = await MarketDataClient(settings).fetch_bars(settings.symbol, limit=bar_limit)
    report = run_parameter_sweep(
        bars,
        settings,
        max_configs=max_configs,
        folds=folds,
        output_dir=Path(settings.log_dir),
    )
    print(json.dumps(report, indent=2, default=str))


def run_parameter_sweep(
    bars: pd.DataFrame,
    base_settings: Settings,
    *,
    max_configs: int | None = 32,
    folds: int = 2,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    _assert_sweep_safety(base_settings)
    output_path = output_dir or Path(base_settings.log_dir)
    parameter_sets = generate_parameter_sets(base_settings, max_configs=max_configs)
    rows: list[dict[str, Any]] = []
    configurations_by_id: dict[str, dict[str, Any]] = {}
    dataset_cache: dict[tuple[Any, ...], pd.DataFrame] = {}
    prediction_cache: dict[tuple[Any, ...], pd.DataFrame] = {}

    for parameter_set in parameter_sets:
        candidate_settings = settings_with_overrides(base_settings, parameter_set.overrides)
        configurations_by_id[parameter_set.parameter_set_id] = {
            "parameter_set_id": parameter_set.parameter_set_id,
            "group": parameter_set.group,
            "description": parameter_set.description,
            "env_overrides": _env_safe_overrides(candidate_settings, parameter_set.env_overrides),
        }
        dataset_key = _dataset_cache_key(candidate_settings)
        dataset = dataset_cache.get(dataset_key)
        if dataset is None:
            dataset = _build_sweep_dataset(bars, candidate_settings)
            dataset_cache[dataset_key] = dataset
        prediction_frame = prediction_cache.get(dataset_key)
        if prediction_frame is None:
            prediction_frame = _walk_forward_prediction_frame(
                dataset,
                min_train_rows=candidate_settings.min_training_rows,
                folds=folds,
                feature_columns=training_feature_columns(candidate_settings.scalping_mode_enabled),
            )
            prediction_cache[dataset_key] = prediction_frame
        rows.extend(evaluate_parameter_set(dataset, prediction_frame, candidate_settings, parameter_set, folds=folds))

    csv_path = output_path / "strategy_param_sweep.csv"
    summary_path = output_path / "strategy_param_sweep_summary.json"
    summary = build_sweep_summary(rows, configurations_by_id, base_settings, csv_path=csv_path, summary_path=summary_path)
    write_sweep_outputs(rows, summary, csv_path=csv_path, summary_path=summary_path)
    return summary


def generate_parameter_sets(base_settings: Settings, *, max_configs: int | None = 32) -> list[ParameterSet]:
    baseline_env = _all_env_values(base_settings)
    sets = [
        ParameterSet(
            parameter_set_id="ps_000_baseline",
            group="baseline",
            description="current_settings_safety_locked",
            overrides={},
            env_overrides=baseline_env,
        )
    ]
    ordered_variants = _bounded_parameter_variants(
        _economic_viability_parameter_variants(base_settings, baseline_env),
        slots=None if max_configs is None or max_configs <= 0 else max(0, max_configs - 1),
    )

    seen = {_parameter_signature(baseline_env)}
    next_index = 1
    for variant in ordered_variants:
        signature = _parameter_signature(variant.env_overrides)
        if signature in seen:
            continue
        seen.add(signature)
        sets.append(
            ParameterSet(
                parameter_set_id=f"ps_{next_index:03d}_{variant.group}",
                group=variant.group,
                description=variant.description,
                overrides=variant.overrides,
                env_overrides=variant.env_overrides,
            )
        )
        next_index += 1
        if max_configs is not None and max_configs > 0 and len(sets) >= max_configs:
            break
    return sets


def _economic_viability_parameter_variants(
    base_settings: Settings,
    baseline_env: dict[str, Any],
) -> list[ParameterSet]:
    group_name = "economic_viability"
    group = ECONOMIC_VIABILITY_SWEEP
    env_names = list(group)
    attr_names = [group[env_name][0] for env_name in env_names]
    value_sets = [group[env_name][1] for env_name in env_names]
    variants: list[ParameterSet] = []
    for values in product(*value_sets):
        overrides = dict(zip(attr_names, values, strict=True))
        env_overrides = {**baseline_env, **dict(zip(env_names, values, strict=True))}
        candidate_settings = settings_with_overrides(base_settings, overrides)
        round_trip_cost = estimated_round_trip_execution_cost_pct(candidate_settings)
        description = (
            f"TP={candidate_settings.scalping_take_profit_pct} "
            f"SL={candidate_settings.scalping_stop_loss_pct} "
            f"horizon={candidate_settings.label_horizon_bars} "
            f"spread_bps={candidate_settings.max_spread_bps} "
            f"slippage_bps={candidate_settings.slippage_bps} "
            f"taker_fees={candidate_settings.backtest_use_taker_fees} "
            f"round_trip_cost={round_trip_cost:.6f}"
        )
        variants.append(
            ParameterSet(
                parameter_set_id="",
                group=group_name,
                description=description,
                overrides=overrides,
                env_overrides=env_overrides,
            )
        )
    return variants


def _bounded_parameter_variants(variants: list[ParameterSet], *, slots: int | None) -> list[ParameterSet]:
    if slots is None or slots <= 0 or slots >= len(variants):
        return variants
    if slots == 1:
        return variants[:1]
    step = (len(variants) - 1) / (slots - 1)
    indexes = [round(index * step) for index in range(slots)]
    selected: list[ParameterSet] = []
    seen: set[int] = set()
    for index in indexes:
        if index not in seen:
            selected.append(variants[index])
            seen.add(index)
    next_index = 0
    while len(selected) < slots and next_index < len(variants):
        if next_index not in seen:
            selected.append(variants[next_index])
            seen.add(next_index)
        next_index += 1
    return selected


def evaluate_parameter_set(
    dataset: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    settings: Settings,
    parameter_set: ParameterSet,
    *,
    folds: int,
) -> list[dict[str, Any]]:
    if prediction_frame.empty:
        return [
            _empty_rejected_row(
                settings,
                parameter_set,
                strategy_name,
                "no_walk_forward_validation_folds",
                folds=folds,
            )
            for strategy_name in STRATEGY_NAMES
        ]

    signal_frame = _annotate_strategy_candidates(
        prediction_frame,
        settings,
        threshold=settings.scalping_buy_probability_floor,
    )
    signal_frame["entry_allowed"] = signal_frame["blocked_by"].isna()
    trades = signal_frame.loc[signal_frame["entry_allowed"]].copy()
    aggregate_metrics = calculate_fee_aware_metrics(trades, settings, signal_frame=signal_frame)
    split_metrics = _split_metrics(signal_frame, settings)
    train_period, validation_period = _periods(signal_frame)
    rows = []
    for strategy_name in STRATEGY_NAMES:
        strategy_metric = _strategy_metric(aggregate_metrics, strategy_name)
        strategy_trade_details = [
            trade for trade in aggregate_metrics.get("trade_details", []) if trade.get("strategy_name") == strategy_name
        ]
        strategy_trades = (
            trades.loc[trades["strategy_name"] == strategy_name].copy()
            if "strategy_name" in trades.columns
            else pd.DataFrame()
        )
        strategy_split_metrics = [
            _strategy_metric(split_metric, strategy_name)
            for split_metric in split_metrics
        ]
        fallback_prediction_used = bool(
            "prediction_source" in signal_frame.columns
            and signal_frame["prediction_source"].astype(str).str.lower().str.startswith("fallback").any()
        )
        trade_frequency_violated = _trade_frequency_violates(strategy_trades, settings)
        round_trip_cost = estimated_round_trip_execution_cost_pct(settings)
        minimum_take_profit = minimum_take_profit_net_positive_pct(settings)
        required_return = promotion_required_return_pct(settings)
        strategy_economics = trade_economics_summary(
            strategy_trade_details,
            notional=float(settings.order_notional_usd),
            required_gross_return_to_overcome_costs=round_trip_cost,
        )
        rejection_reasons = reject_config(
            strategy_metric,
            settings=settings,
            split_metrics=strategy_split_metrics,
            trade_details=strategy_trade_details,
            fallback_prediction_used=fallback_prediction_used,
            trade_frequency_violated=trade_frequency_violated,
        )
        split_stability = split_stability_summary(strategy_split_metrics)
        top_block_reason = _top_block_reason(signal_frame, strategy_name)
        row = {
            "parameter_set_id": parameter_set.parameter_set_id,
            "parameter_group": parameter_set.group,
            "description": parameter_set.description,
            "strategy_name": strategy_name,
            **_all_env_values(settings),
            "number_of_signals": int(strategy_metric.get("number_of_signals", 0)),
            "number_of_trades": int(strategy_metric.get("number_of_trades", 0)),
            "net_return_pct": _metric_float(strategy_metric.get("net_return_pct")),
            "gross_return_pct": _metric_float(strategy_metric.get("gross_return_pct")),
            "profit_factor_net": _metric_float(strategy_metric.get("profit_factor_net")),
            "max_drawdown_pct": _metric_float(strategy_metric.get("max_drawdown_pct")),
            "win_rate_net": _metric_float(strategy_metric.get("win_rate_net")),
            "expectancy": _metric_float(strategy_metric.get("expectancy")),
            "average_trade_net_return": _metric_float(strategy_metric.get("average_net_return_pct")),
            "average_hold_bars": _metric_float(strategy_metric.get("average_hold_bars")),
            "round_trip_estimated_cost_pct": round_trip_cost,
            "promotion_required_return_pct": required_return,
            "minimum_take_profit_net_positive_pct": minimum_take_profit,
            "scalping_take_profit_covers_cost": settings.scalping_take_profit_pct > minimum_take_profit,
            "gross_winners_became_net_losers": int(strategy_economics["gross_winners_became_net_losers"]),
            "average_gross_winning_trade": _metric_float(strategy_economics["average_gross_winning_trade"]),
            "average_net_winning_trade": _metric_float(strategy_economics["average_net_winning_trade"]),
            "average_total_execution_cost_pct_per_trade": _metric_float(
                strategy_economics["average_total_execution_cost_pct_per_trade"]
            ),
            "required_gross_return_to_overcome_costs": _metric_float(
                strategy_economics["required_gross_return_to_overcome_costs"]
            ),
            "canceled_orders": int(strategy_metric.get("canceled_orders", 0)),
            "ambiguous_candle_ratio": _metric_float(strategy_metric.get("ambiguous_candle_ratio")),
            "blocked_signal_count": int(strategy_metric.get("number_of_blocked_signals", 0)),
            "top_block_reason": top_block_reason,
            "train_period": train_period,
            "validation_period": validation_period,
            "walk_forward_splits": folds,
            "profitable_split_count": split_stability["profitable_split_count"],
            "split_return_concentration": split_stability["split_return_concentration"],
            "single_trade_return_concentration": single_trade_return_concentration(strategy_trade_details),
            "trade_frequency_violated": trade_frequency_violated,
            "accepted": not rejection_reasons,
            "rejection_reasons": ";".join(rejection_reasons),
            "rank_score": rank_score(strategy_metric, split_stability, rejection_reasons),
        }
        rows.append(row)
    return rows


def reject_config(
    metrics: dict[str, Any],
    *,
    settings: Settings | None = None,
    split_metrics: list[dict[str, Any]] | None = None,
    trade_details: list[dict[str, Any]] | None = None,
    fallback_prediction_used: bool = False,
    trade_frequency_violated: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if fallback_prediction_used:
        reasons.append("fallback_prediction_not_allowed")
    if int(metrics.get("number_of_trades", 0) or 0) < MIN_SWEEP_TRADES:
        reasons.append("number_of_trades_below_min")
    if _metric_float(metrics.get("max_drawdown_pct")) > MAX_SWEEP_DRAWDOWN_PCT:
        reasons.append("max_drawdown_too_high")
    if _profit_factor_value(metrics.get("profit_factor_net")) < MIN_SWEEP_PROFIT_FACTOR_NET:
        reasons.append("profit_factor_net_too_low")
    if _metric_float(metrics.get("net_return_pct")) <= 0:
        reasons.append("net_return_not_positive")
    if _metric_float(metrics.get("expectancy")) <= 0:
        reasons.append("expectancy_not_positive")
    if _metric_float(metrics.get("ambiguous_candle_ratio")) > MAX_SWEEP_AMBIGUOUS_CANDLE_RATIO:
        reasons.append("ambiguous_candle_ratio_too_high")
    if single_trade_return_concentration(trade_details or []) > MAX_SINGLE_TRADE_RETURN_SHARE:
        reasons.append("single_trade_return_concentration_too_high")
    if split_stability_summary(split_metrics or [])["split_return_concentration"] > MAX_SINGLE_SPLIT_RETURN_SHARE:
        reasons.append("one_walk_forward_split_dominates_profit")
    if trade_frequency_violated:
        reasons.append("trade_frequency_violates_risk_limit")
    if settings is not None and settings.allow_fallback_trading:
        reasons.append("fallback_trading_enabled_not_allowed")
    return reasons


def split_stability_summary(split_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    split_returns = [_metric_float(metrics.get("net_return_pct")) for metrics in split_metrics]
    positive_returns = [value for value in split_returns if value > 0]
    total_positive_return = sum(positive_returns)
    concentration = max(positive_returns) / total_positive_return if total_positive_return > 0 else 0.0
    return {
        "split_count": len(split_metrics),
        "profitable_split_count": len(positive_returns),
        "split_return_concentration": concentration,
        "stable_across_splits": len(positive_returns) >= 2 and concentration <= MAX_SINGLE_SPLIT_RETURN_SHARE,
    }


def single_trade_return_concentration(trade_details: list[dict[str, Any]]) -> float:
    returns = [_metric_float(trade.get("net_return_pct", trade.get("net_return"))) for trade in trade_details]
    positive_returns = [value for value in returns if value > 0]
    total_positive_return = sum(positive_returns)
    if total_positive_return <= 0:
        return 0.0
    return max(positive_returns) / total_positive_return


def rank_score(metrics: dict[str, Any], split_stability: dict[str, Any], rejection_reasons: list[str]) -> float:
    if rejection_reasons:
        return -1_000_000.0 + _metric_float(metrics.get("net_return_pct"))
    profit_factor = min(5.0, _profit_factor_value(metrics.get("profit_factor_net")))
    return (
        _metric_float(metrics.get("net_return_pct")) * 10_000
        + profit_factor * 10
        + _metric_float(metrics.get("expectancy")) * 1_000
        + int(metrics.get("number_of_trades", 0) or 0) * 0.1
        + float(split_stability.get("profitable_split_count", 0)) * 2
        - _metric_float(metrics.get("max_drawdown_pct")) * 1_000
        - _metric_float(metrics.get("ambiguous_candle_ratio")) * 100
    )


def build_sweep_summary(
    rows: list[dict[str, Any]],
    configurations_by_id: dict[str, dict[str, Any]],
    settings: Settings,
    *,
    csv_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: float(row.get("rank_score", -1_000_000)), reverse=True)
    accepted = [row for row in ranked if row.get("accepted")]
    stable = [
        row
        for row in accepted
        if float(row.get("split_return_concentration") or 0.0) <= MAX_SINGLE_SPLIT_RETURN_SHARE
        and int(row.get("profitable_split_count") or 0) >= 2
    ]
    rejected = [row for row in ranked if not row.get("accepted")]
    return _json_safe(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "symbol": settings.symbol,
            "paper_trading_only": settings.paper_trading_only,
            "btc_usd_only": settings.symbol == ALLOWED_SYMBOL,
            "long_only": True,
            "fallback_trading_allowed": settings.allow_fallback_trading,
            "auto_apply_best_config": False,
            "diagnostic_only_parameters": ["BACKTEST_USE_TAKER_FEES"],
            "walk_forward_validation": True,
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "parameter_space": parameter_space(),
            "total_parameter_sets": len(configurations_by_id),
            "total_strategy_rows": len(rows),
            "best_candidate_configs": accepted[:10],
            "stable_candidate_configs": stable[:10],
            "rejected_configs": rejected[:50],
            "recommended_paper_forward_test_configs": (stable or accepted)[:3],
            "configurations_by_id": configurations_by_id,
            "notes": [
                "Backtests are historical simulations and do not guarantee future profitability.",
                "No parameters are auto-applied; use scripts/apply_candidate_config.py to write .env.candidate only.",
                "Default sweep is bounded and deterministic to reduce overfit pressure.",
                "BACKTEST_USE_TAKER_FEES is swept only to diagnose economics; conservative promotion remains the default.",
            ],
        }
    )


def write_sweep_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    csv_path: Path,
    summary_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


def settings_with_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    data = settings.model_dump()
    data.update(overrides)
    data.update(
        {
            "paper_trading_only": True,
            "symbol": ALLOWED_SYMBOL,
            "trading_enabled": False,
            "auto_trade_enabled": False,
            "allow_fallback_trading": False,
            "scalping_mode_enabled": True,
        }
    )
    return Settings(_env_file=None, **data)


def parameter_space() -> dict[str, dict[str, list[Any]]]:
    return {
        group_name: {env_name: list(values) for env_name, (_, values) in group.items()}
        for group_name, group in PARAMETER_GROUPS.items()
    }


def _assert_sweep_safety(settings: Settings) -> None:
    if settings.symbol != ALLOWED_SYMBOL:
        raise ValueError("Parameter sweep is BTC/USD-only.")
    if not settings.paper_trading_only:
        raise ValueError("Parameter sweep requires PAPER_TRADING_ONLY=true.")
    if settings.allow_fallback_trading:
        raise ValueError("Parameter sweep refuses ALLOW_FALLBACK_TRADING=true.")


def _build_sweep_dataset(bars: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    return build_training_dataset(
        bars,
        scalping_label_horizon_bars=settings.scalping_label_horizon_bars,
        label_horizon_bars=settings.label_horizon_bars if settings.scalping_mode_enabled else None,
        take_profit_pct=settings.scalping_take_profit_pct if settings.scalping_mode_enabled else settings.take_profit_pct,
        stop_loss_pct=settings.scalping_stop_loss_pct if settings.scalping_mode_enabled else settings.stop_loss_pct,
        scalping_mode_enabled=settings.scalping_mode_enabled,
        trailing_stop_pct=settings.scalping_trailing_stop_pct if settings.scalping_mode_enabled else settings.trailing_stop_pct,
        trailing_stop_arm_profit_pct=settings.trailing_stop_arm_profit_pct,
        fee_bps_per_side=settings.label_fee_bps_per_side if settings.scalping_mode_enabled else 0.0,
        slippage_bps_per_side=settings.label_slippage_bps_per_side if settings.scalping_mode_enabled else 0.0,
        spread_cost_pct=(settings.label_spread_bps / 10_000) if settings.scalping_mode_enabled else 0.0,
        min_net_exit_profit_pct=settings.label_min_net_profit_pct if settings.scalping_mode_enabled else 0.0,
        exit_profit_buffer_bps=0.0,
    )


def _split_metrics(signal_frame: pd.DataFrame, settings: Settings) -> list[dict[str, Any]]:
    if "_walk_forward_split" not in signal_frame.columns:
        return []
    metrics = []
    for split_id in sorted(signal_frame["_walk_forward_split"].dropna().unique()):
        split_frame = signal_frame.loc[signal_frame["_walk_forward_split"] == split_id].copy()
        split_trades = split_frame.loc[split_frame["entry_allowed"]].copy()
        metrics.append(calculate_fee_aware_metrics(split_trades, settings, signal_frame=split_frame))
    return metrics


def _strategy_metric(metrics: dict[str, Any], strategy_name: str) -> dict[str, Any]:
    return dict((metrics.get("strategy_level_metrics") or {}).get(strategy_name, _zero_strategy_metric()))


def _zero_strategy_metric() -> dict[str, Any]:
    return {
        "number_of_signals": 0,
        "number_of_entries": 0,
        "number_of_exits": 0,
        "number_of_trades": 0,
        "win_rate_net": 0.0,
        "average_net_return_pct": 0.0,
        "expectancy": 0.0,
        "gross_return_pct": 0.0,
        "net_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "profit_factor_net": 0.0,
        "average_hold_bars": 0.0,
        "canceled_orders": 0,
        "partial_fills": 0,
        "ambiguous_candle_ratio": 0.0,
        "number_of_allowed_signals": 0,
        "number_of_blocked_signals": 0,
    }


def _empty_rejected_row(
    settings: Settings,
    parameter_set: ParameterSet,
    strategy_name: str,
    reason: str,
    *,
    folds: int,
) -> dict[str, Any]:
    round_trip_cost = estimated_round_trip_execution_cost_pct(settings)
    minimum_take_profit = minimum_take_profit_net_positive_pct(settings)
    return {
        "parameter_set_id": parameter_set.parameter_set_id,
        "parameter_group": parameter_set.group,
        "description": parameter_set.description,
        "strategy_name": strategy_name,
        **_all_env_values(settings),
        **_zero_strategy_metric(),
        "average_trade_net_return": 0.0,
        "round_trip_estimated_cost_pct": round_trip_cost,
        "promotion_required_return_pct": promotion_required_return_pct(settings),
        "minimum_take_profit_net_positive_pct": minimum_take_profit,
        "scalping_take_profit_covers_cost": settings.scalping_take_profit_pct > minimum_take_profit,
        "gross_winners_became_net_losers": 0,
        "average_gross_winning_trade": 0.0,
        "average_net_winning_trade": 0.0,
        "average_total_execution_cost_pct_per_trade": 0.0,
        "required_gross_return_to_overcome_costs": round_trip_cost,
        "blocked_signal_count": 0,
        "top_block_reason": reason,
        "train_period": "unavailable",
        "validation_period": "unavailable",
        "walk_forward_splits": folds,
        "profitable_split_count": 0,
        "split_return_concentration": 0.0,
        "single_trade_return_concentration": 0.0,
        "trade_frequency_violated": False,
        "accepted": False,
        "rejection_reasons": reason,
        "rank_score": -1_000_000.0,
    }


def _trade_frequency_violates(trades: pd.DataFrame, settings: Settings) -> bool:
    if trades.empty or "timestamp" not in trades.columns:
        return False
    timestamps = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce").dropna().sort_values()
    if timestamps.empty:
        return False
    if len(timestamps) > 1:
        min_gap = timestamps.diff().dropna().dt.total_seconds().min()
        if min_gap is not None and min_gap < settings.min_seconds_between_trades:
            return True
    per_hour = timestamps.dt.floor("h").value_counts()
    return bool(not per_hour.empty and int(per_hour.max()) > settings.max_trades_per_hour)


def _top_block_reason(signal_frame: pd.DataFrame, strategy_name: str) -> str | None:
    if signal_frame.empty or "strategy_name" not in signal_frame.columns or "block_reason" not in signal_frame.columns:
        return None
    blocked = signal_frame.loc[
        (signal_frame["strategy_name"] == strategy_name) & signal_frame["block_reason"].notna(),
        "block_reason",
    ]
    if blocked.empty:
        return None
    return Counter(blocked.astype(str)).most_common(1)[0][0]


def _periods(signal_frame: pd.DataFrame) -> tuple[str, str]:
    train_start = _first_non_empty(signal_frame.get("_train_start_timestamp"))
    train_end = _last_non_empty(signal_frame.get("_train_end_timestamp"))
    validation_start = _first_non_empty(signal_frame.get("_validation_start_timestamp"))
    validation_end = _last_non_empty(signal_frame.get("_validation_end_timestamp"))
    return _period_string(train_start, train_end), _period_string(validation_start, validation_end)


def _first_non_empty(series: pd.Series | None) -> Any:
    if series is None:
        return None
    values = [value for value in series.dropna().tolist() if value not in {"", None}]
    return values[0] if values else None


def _last_non_empty(series: pd.Series | None) -> Any:
    if series is None:
        return None
    values = [value for value in series.dropna().tolist() if value not in {"", None}]
    return values[-1] if values else None


def _period_string(start: Any, end: Any) -> str:
    if start is None and end is None:
        return "unavailable"
    return f"{start or 'unknown'}..{end or 'unknown'}"


def _all_env_values(settings: Settings) -> dict[str, Any]:
    names = {}
    for group in PARAMETER_GROUPS.values():
        for env_name, (attr_name, _) in group.items():
            names[env_name] = getattr(settings, attr_name)
    return names


def _env_safe_overrides(settings: Settings, env_overrides: dict[str, Any]) -> dict[str, Any]:
    return {
        "PAPER_TRADING_ONLY": True,
        "SYMBOL": ALLOWED_SYMBOL,
        "TRADING_ENABLED": False,
        "AUTO_TRADE_ENABLED": False,
        "ALLOW_FALLBACK_TRADING": False,
        "SCALPING_MODE_ENABLED": True,
        **{name: getattr(settings, attr_name) for group in PARAMETER_GROUPS.values() for name, (attr_name, _) in group.items()},
        **env_overrides,
    }


def _dataset_cache_key(settings: Settings) -> tuple[Any, ...]:
    return (
        settings.label_horizon_bars,
        settings.scalping_take_profit_pct,
        settings.scalping_stop_loss_pct,
        settings.scalping_trailing_stop_pct,
        settings.trailing_stop_arm_profit_pct,
        settings.label_fee_bps_per_side,
        settings.label_slippage_bps_per_side,
        settings.label_spread_bps,
        settings.label_min_net_profit_pct,
    )


def _parameter_signature(env_values: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, repr(value)) for key, value in env_values.items()))


def _metric_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(parsed):
        return 0.0
    return parsed


def _profit_factor_value(value: Any) -> float:
    parsed = _metric_float(value)
    if math.isinf(parsed):
        return 1_000_000.0
    return parsed


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "strategy_name",
        "parameter_set_id",
        "parameter_group",
        "description",
        *_all_parameter_env_names(),
        "number_of_signals",
        "number_of_trades",
        "net_return_pct",
        "gross_return_pct",
        "profit_factor_net",
        "max_drawdown_pct",
        "win_rate_net",
        "expectancy",
        "average_trade_net_return",
        "average_hold_bars",
        "round_trip_estimated_cost_pct",
        "promotion_required_return_pct",
        "minimum_take_profit_net_positive_pct",
        "scalping_take_profit_covers_cost",
        "gross_winners_became_net_losers",
        "average_gross_winning_trade",
        "average_net_winning_trade",
        "average_total_execution_cost_pct_per_trade",
        "required_gross_return_to_overcome_costs",
        "canceled_orders",
        "ambiguous_candle_ratio",
        "blocked_signal_count",
        "top_block_reason",
        "train_period",
        "validation_period",
        "accepted",
        "rejection_reasons",
        "rank_score",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    return [field for field in preferred if any(field in row for row in rows)] + extras


def _all_parameter_env_names() -> list[str]:
    names: list[str] = []
    for group in PARAMETER_GROUPS.values():
        names.extend(group.keys())
    return names


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(_json_safe(value), sort_keys=True)
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    return value


if __name__ == "__main__":
    asyncio.run(main())
