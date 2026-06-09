from __future__ import annotations

import asyncio
import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.backtest.scalping import (
    backtest_assumptions,
    calculate_fee_aware_metrics,
    estimated_round_trip_execution_cost_pct,
    promotion_required_return_pct as estimated_promotion_required_return_pct,
)
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.data.feature_engineering import add_features
from app.data.market_data import (
    MarketDataClient,
    StaleMarketDataError,
    normalize_ohlcv,
    stale_threshold_for_timeframe,
)
from app.db.database import SessionLocal, init_db
from app.db.models import CollectedMarketData
from app.ml.registry import ModelRegistry
from app.risk.risk_manager import PositionState
from app.strategy.strategies import MarketContext, MarketRegimeFilter, TrendPullbackStrategy


LEGACY_RESEARCH_TIMEFRAMES = ("5Min", "15Min")
V3_RESEARCH_TIMEFRAMES = ("15Min", "1H")
RESEARCH_TIMEFRAMES = LEGACY_RESEARCH_TIMEFRAMES
SUPPORTED_RESEARCH_TIMEFRAMES = ("5Min", "15Min", "1H")
TREND_PULLBACK_STRATEGY = "trend_pullback"
BUY_THE_DIP_STRATEGY = "buy_the_dip_mean_reversion"
UPTREND_PULLBACK_STRATEGY = "uptrend_pullback"
VOLATILITY_BREAKOUT_STRATEGY = "volatility_breakout"
V3_STRATEGIES = (UPTREND_PULLBACK_STRATEGY, VOLATILITY_BREAKOUT_STRATEGY)
STRATEGY_CHOICES = (
    "all",
    TREND_PULLBACK_STRATEGY,
    BUY_THE_DIP_STRATEGY,
    UPTREND_PULLBACK_STRATEGY,
    VOLATILITY_BREAKOUT_STRATEGY,
)
TAKE_PROFIT_VALUES = (0.008, 0.01, 0.015, 0.02)
STOP_LOSS_VALUES = (0.003, 0.005, 0.008)
MAX_HOLD_BARS_VALUES = (6, 12, 24, 48)
BUY_THE_DIP_TAKE_PROFIT_VALUES = (0.0086, 0.01, 0.0125, 0.015, 0.02, 0.025)
BUY_THE_DIP_STOP_LOSS_VALUES = (0.004, 0.006, 0.008, 0.01)
BUY_THE_DIP_MAX_HOLD_BARS_VALUES = (6, 12, 24, 48, 72)
UPTREND_PULLBACK_TAKE_PROFIT_VALUES = (0.015, 0.02, 0.03, 0.04)
UPTREND_PULLBACK_STOP_LOSS_VALUES = (0.008, 0.012, 0.015, 0.02)
V3_MAX_HOLD_BARS_VALUES = (12, 24, 48, 72)
VOLATILITY_BREAKOUT_TAKE_PROFIT_VALUES = (0.02, 0.03, 0.04, 0.05)
VOLATILITY_BREAKOUT_STOP_LOSS_VALUES = (0.01, 0.015, 0.02)
BUY_THE_DIP_SIGNAL_PROFILES = (
    {
        "rsi_threshold": 35.0,
        "zscore_threshold": -1.5,
        "vwap_distance_threshold": -0.003,
        "drawdown_threshold": 0.005,
        "min_volume_zscore": 0.5,
        "reversal_confirmation_required": False,
        "higher_timeframe_regime_filter": False,
    },
    {
        "rsi_threshold": 30.0,
        "zscore_threshold": -2.0,
        "vwap_distance_threshold": -0.005,
        "drawdown_threshold": 0.008,
        "min_volume_zscore": 1.0,
        "reversal_confirmation_required": True,
        "higher_timeframe_regime_filter": False,
    },
    {
        "rsi_threshold": 25.0,
        "zscore_threshold": -2.0,
        "vwap_distance_threshold": -0.008,
        "drawdown_threshold": 0.012,
        "min_volume_zscore": 1.5,
        "reversal_confirmation_required": True,
        "higher_timeframe_regime_filter": True,
    },
    {
        "rsi_threshold": 20.0,
        "zscore_threshold": -2.5,
        "vwap_distance_threshold": -0.010,
        "drawdown_threshold": 0.020,
        "min_volume_zscore": 2.0,
        "reversal_confirmation_required": True,
        "higher_timeframe_regime_filter": True,
    },
)
BUY_THE_DIP_V2_PROFILE_VALUES = {
    "rsi_threshold": (25.0, 30.0, 35.0, 40.0),
    "zscore_threshold": (-1.0, -1.25, -1.5, -2.0),
    "vwap_distance_threshold": (-0.002, -0.003, -0.005, -0.008),
    "drawdown_threshold": (0.003, 0.005, 0.008, 0.012),
    "min_volume_zscore": (-0.5, 0.0, 0.5, 1.0),
    "reversal_confirmation_required": (True, False),
    "higher_timeframe_regime_filter": (True, False),
}
DEFAULT_MAX_BUY_DIP_CONFIGS = 960
DEFAULT_MAX_V3_CONFIGS = 3000
DEFAULT_WALK_FORWARD_SPLITS = 4
PREFERRED_RESEARCH_TRADES = 50
MIN_RESEARCH_TRADES = 20
MIN_RESEARCH_TRADES_PER_SPLIT = 3
MIN_RESEARCH_PROFIT_FACTOR_NET = 1.05
MAX_SINGLE_TRADE_RETURN_SHARE = 0.60


@dataclass(frozen=True)
class ResearchConfig:
    parameter_set_id: str
    strategy_name: str
    timeframe: str
    take_profit_pct: float
    stop_loss_pct: float
    max_hold_bars: int
    rsi_threshold: float | None = None
    zscore_threshold: float | None = None
    vwap_distance_threshold: float | None = None
    drawdown_threshold: float | None = None
    min_volume_zscore: float | None = None
    reversal_confirmation_required: bool | None = None
    higher_timeframe_regime_filter: bool | None = None
    support: str | None = None
    support_distance_pct: float | None = None
    pullback_min_pct: float | None = None
    pullback_max_pct: float | None = None
    rsi_min: float | None = None
    rsi_max: float | None = None
    confirmation: str | None = None
    min_lower_wick_ratio: float | None = None
    min_volume_recovery: float | None = None
    breakout_lookback: int | None = None
    consolidation_lookback: int | None = None
    min_body_vs_avg: float | None = None
    min_recent_return_pct: float | None = None
    min_trend_strength: float | None = None
    max_atr_expansion: float | None = None


@dataclass(frozen=True)
class ResearchDataReport:
    timeframe: str
    source_used: str
    latest_timestamp: str | None
    data_age_minutes: float | None
    row_count: int
    synthetic_data_used: bool
    research_result_valid: bool
    rejection_reason: str | None = None
    rejected_sources: tuple[dict[str, Any], ...] = ()
    available_rows: int | None = None
    used_rows: int | None = None
    first_timestamp: str | None = None
    requested_max_rows: int | None = None
    requested_start: str | None = None
    requested_end: str | None = None
    derived_from_timeframe: str | None = None


@dataclass(frozen=True)
class ResearchBarsResult:
    bars: pd.DataFrame
    report: ResearchDataReport

    def __iter__(self):
        yield self.bars
        yield self.report.source_used


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline BTC/USD higher-timeframe research.")
    parser.add_argument("--json", action="store_true", help="Print JSON output. JSON is also the default output format.")
    parser.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default="all",
        help="Limit research to a single strategy or run all strategies.",
    )
    parser.add_argument("--max-rows-per-timeframe", type=int, default=None, help="Maximum rows per timeframe.")
    parser.add_argument("--max-rows-5min", type=int, default=None, help="Maximum 5Min rows.")
    parser.add_argument("--max-rows-15min", type=int, default=None, help="Maximum 15Min rows.")
    parser.add_argument("--max-rows-1h", type=int, default=None, help="Maximum 1H rows.")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        choices=SUPPORTED_RESEARCH_TIMEFRAMES,
        default=None,
        help="Timeframes to evaluate. Defaults preserve legacy strategy behavior and use 15Min/1H for v3.",
    )
    parser.add_argument("--start", default=None, help="UTC start date or ISO timestamp.")
    parser.add_argument("--end", default=None, help="UTC end date or ISO timestamp.")
    parser.add_argument("--lookback-days", type=int, default=None, help="Look back this many days from end/now.")
    parser.add_argument(
        "--max-buy-dip-configs",
        type=int,
        default=DEFAULT_MAX_BUY_DIP_CONFIGS,
        help="Maximum buy-the-dip configs to evaluate.",
    )
    parser.add_argument(
        "--max-v3-configs",
        type=int,
        default=DEFAULT_MAX_V3_CONFIGS,
        help="Maximum v3 configs to evaluate for uptrend pullback and volatility breakout.",
    )
    parser.add_argument(
        "--walk-forward-splits",
        type=int,
        default=DEFAULT_WALK_FORWARD_SPLITS,
        help="Chronological walk-forward fold count.",
    )
    parser.add_argument("--min-trades", type=int, default=MIN_RESEARCH_TRADES, help="Minimum total trades per config.")
    parser.add_argument(
        "--min-trades-per-split",
        type=int,
        default=MIN_RESEARCH_TRADES_PER_SPLIT,
        help="Minimum trades required in a fold for walk-forward robustness.",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    settings = research_settings(get_settings())
    bar_limit = int(os.getenv("RESEARCH_BAR_LIMIT", str(max(1500, settings.min_training_rows + 500))))
    timeframes = _resolve_requested_timeframes(args.strategy, args.timeframes)
    row_limits = _row_limits_by_timeframe(
        default_limit=bar_limit,
        max_rows_per_timeframe=args.max_rows_per_timeframe,
        max_rows_5min=args.max_rows_5min,
        max_rows_15min=args.max_rows_15min,
        max_rows_1h=args.max_rows_1h,
        timeframes=timeframes,
    )
    report = await run_higher_timeframe_research(
        settings,
        bar_limit=bar_limit,
        max_rows_by_timeframe=row_limits,
        timeframes=timeframes,
        start=args.start,
        end=args.end,
        lookback_days=args.lookback_days,
        strategy=args.strategy,
        max_buy_dip_configs=args.max_buy_dip_configs,
        max_v3_configs=args.max_v3_configs,
        walk_forward_splits=args.walk_forward_splits,
        min_trades=args.min_trades,
        min_trades_per_split=args.min_trades_per_split,
        output_dir=Path(settings.log_dir),
        allow_synthetic_fallback=_synthetic_research_mode_enabled(),
    )
    print(json.dumps(report, indent=2, default=str))


async def run_higher_timeframe_research(
    base_settings: Settings,
    *,
    bar_limit: int = 1500,
    max_rows_by_timeframe: dict[str, int] | None = None,
    timeframes: tuple[str, ...] | list[str] | None = None,
    start: Any | None = None,
    end: Any | None = None,
    lookback_days: int | None = None,
    strategy: str = "all",
    max_buy_dip_configs: int = DEFAULT_MAX_BUY_DIP_CONFIGS,
    max_v3_configs: int = DEFAULT_MAX_V3_CONFIGS,
    walk_forward_splits: int = DEFAULT_WALK_FORWARD_SPLITS,
    min_trades: int = MIN_RESEARCH_TRADES,
    min_trades_per_split: int = MIN_RESEARCH_TRADES_PER_SPLIT,
    client: MarketDataClient | None = None,
    output_dir: Path | None = None,
    session_factory: Any | None = None,
    allow_synthetic_fallback: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    settings = research_settings(base_settings)
    if strategy not in STRATEGY_CHOICES:
        raise ValueError(f"Unsupported research strategy: {strategy}")
    if max_buy_dip_configs <= 0:
        raise ValueError("max_buy_dip_configs must be positive")
    if max_v3_configs <= 0:
        raise ValueError("max_v3_configs must be positive")
    if walk_forward_splits <= 0:
        raise ValueError("walk_forward_splits must be positive")
    if min_trades <= 0:
        raise ValueError("min_trades must be positive")
    if min_trades_per_split <= 0:
        raise ValueError("min_trades_per_split must be positive")
    client = client or MarketDataClient(settings)
    active_model = ModelRegistry(settings).validate_active_model()
    bars_by_timeframe: dict[str, pd.DataFrame] = {}
    data_source_reports: dict[str, ResearchDataReport] = {}
    current_time = _utc_timestamp(now or datetime.now(UTC))
    window_start, window_end = _resolve_research_window(
        start=start,
        end=end,
        lookback_days=lookback_days,
        now=current_time,
    )
    requested_timeframes = _resolve_requested_timeframes(strategy, timeframes)
    row_limits = max_rows_by_timeframe or {
        timeframe: int(bar_limit) for timeframe in requested_timeframes
    }
    for timeframe in requested_timeframes:
        limit = int(row_limits.get(timeframe, bar_limit))
        result = await _fetch_or_derive_research_bars(
            client,
            settings,
            timeframe=timeframe,
            limit=limit,
            existing_bars_by_timeframe=bars_by_timeframe,
            existing_reports_by_timeframe=data_source_reports,
            row_limits=row_limits,
            session_factory=session_factory,
            allow_synthetic_fallback=allow_synthetic_fallback,
            now=current_time,
            start=window_start,
            end=window_end,
        )
        bars_by_timeframe[timeframe] = result.bars
        data_source_reports[timeframe] = result.report
    rows = evaluate_research_configs(
        bars_by_timeframe,
        settings,
        active_model_valid=active_model.valid,
        active_model_status=active_model.to_dict(),
        data_source_reports=data_source_reports,
        strategy=strategy,
        max_buy_dip_configs=max_buy_dip_configs,
        max_v3_configs=max_v3_configs,
        walk_forward_splits=walk_forward_splits,
        min_trades=min_trades,
        min_trades_per_split=min_trades_per_split,
        timeframes=requested_timeframes,
    )
    output_path = output_dir or Path(settings.log_dir)
    csv_path = output_path / "higher_timeframe_research.csv"
    summary_path = output_path / "higher_timeframe_research_summary.json"
    summary = build_research_summary(
        rows,
        settings,
        data_source_reports=data_source_reports,
        csv_path=csv_path,
        summary_path=summary_path,
        active_model_status=active_model.to_dict(),
        requested_max_rows_by_timeframe=row_limits,
        requested_start=window_start.isoformat() if window_start else None,
        requested_end=window_end.isoformat() if window_end else None,
        strategy_filter=strategy,
        max_buy_dip_configs=max_buy_dip_configs,
        max_v3_configs=max_v3_configs,
        walk_forward_splits=walk_forward_splits,
        min_trades=min_trades,
        min_trades_per_split=min_trades_per_split,
        requested_timeframes=requested_timeframes,
    )
    write_research_outputs(rows, summary, csv_path=csv_path, summary_path=summary_path)
    return summary


def research_settings(settings: Settings) -> Settings:
    data = settings.model_dump()
    data.update(
        {
            "symbol": ALLOWED_SYMBOL,
            "paper_trading_only": True,
            "trading_enabled": False,
            "auto_trade_enabled": False,
            "allow_fallback_trading": False,
            "max_open_positions": 1,
        }
    )
    return Settings(_env_file=None, **data)


def _row_limits_by_timeframe(
    *,
    default_limit: int,
    max_rows_per_timeframe: int | None = None,
    max_rows_5min: int | None = None,
    max_rows_15min: int | None = None,
    max_rows_1h: int | None = None,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, int]:
    base = max(1, int(max_rows_per_timeframe or default_limit))
    limits = {
        "5Min": max(1, int(max_rows_5min or base)),
        "15Min": max(1, int(max_rows_15min or base)),
        "1H": max(1, int(max_rows_1h or base)),
    }
    requested = tuple(timeframes or SUPPORTED_RESEARCH_TIMEFRAMES)
    return {timeframe: limits[timeframe] for timeframe in requested}


def _resolve_requested_timeframes(strategy: str, timeframes: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if timeframes:
        requested = tuple(dict.fromkeys(str(timeframe) for timeframe in timeframes))
    elif strategy in V3_STRATEGIES or strategy == "all":
        requested = V3_RESEARCH_TIMEFRAMES
    else:
        requested = LEGACY_RESEARCH_TIMEFRAMES
    unsupported = [timeframe for timeframe in requested if timeframe not in SUPPORTED_RESEARCH_TIMEFRAMES]
    if unsupported:
        raise ValueError(f"Unsupported research timeframe(s): {', '.join(unsupported)}")
    if "1H" in requested:
        requested = tuple(
            [
                *(timeframe for timeframe in requested if timeframe not in {"15Min", "1H"}),
                "15Min",
                "1H",
            ]
        )
    return requested


def _market_data_timeframe(timeframe: str) -> str:
    return "1Hour" if timeframe == "1H" else timeframe


def _resolve_research_window(
    *,
    start: Any | None,
    end: Any | None,
    lookback_days: int | None,
    now: datetime,
) -> tuple[datetime | None, datetime]:
    window_end = _utc_timestamp(end) if end is not None else now
    window_end = min(window_end, now)
    window_start = _utc_timestamp(start) if start is not None else None
    if lookback_days is not None:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        implied_start = window_end - timedelta(days=int(lookback_days))
        window_start = max(window_start, implied_start) if window_start is not None else implied_start
    if window_start is not None and window_start >= window_end:
        raise ValueError("research start must be before end")
    return window_start, window_end


def evaluate_research_configs(
    bars_by_timeframe: dict[str, pd.DataFrame],
    settings: Settings,
    *,
    active_model_valid: bool,
    active_model_status: dict[str, Any] | None = None,
    data_source_reports: dict[str, ResearchDataReport | dict[str, Any]] | None = None,
    strategy: str = "all",
    max_buy_dip_configs: int = DEFAULT_MAX_BUY_DIP_CONFIGS,
    max_v3_configs: int = DEFAULT_MAX_V3_CONFIGS,
    walk_forward_splits: int = DEFAULT_WALK_FORWARD_SPLITS,
    min_trades: int = MIN_RESEARCH_TRADES,
    min_trades_per_split: int = MIN_RESEARCH_TRADES_PER_SPLIT,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    if settings.symbol != ALLOWED_SYMBOL:
        raise ValueError("Higher-timeframe research is BTC/USD-only.")
    rows: list[dict[str, Any]] = []
    source_reports = {
        timeframe: _coerce_data_report(timeframe, report)
        for timeframe, report in (data_source_reports or {}).items()
    }
    buy_the_dip_features = {
        timeframe: prepare_buy_the_dip_features(bars)
        for timeframe, bars in bars_by_timeframe.items()
        if not bars.empty
    }
    v3_features = {
        timeframe: prepare_v3_features(bars)
        for timeframe, bars in bars_by_timeframe.items()
        if not bars.empty
    }
    walk_forward_inputs = build_walk_forward_inputs(
        bars_by_timeframe,
        buy_the_dip_features=buy_the_dip_features,
        v3_features=v3_features,
        splits=walk_forward_splits,
    )
    requested_timeframes = _resolve_requested_timeframes(strategy, timeframes)
    for config in generate_research_configs(
        strategy=strategy,
        max_buy_dip_configs=max_buy_dip_configs,
        max_v3_configs=max_v3_configs,
        timeframes=requested_timeframes,
    ):
        candidate_settings = research_settings(
            Settings(
                _env_file=None,
                **{
                    **settings.model_dump(),
                    "take_profit_pct": config.take_profit_pct,
                    "stop_loss_pct": config.stop_loss_pct,
                    "scalping_take_profit_pct": config.take_profit_pct,
                    "scalping_stop_loss_pct": config.stop_loss_pct,
                    "label_horizon_bars": config.max_hold_bars,
                },
            )
        )
        bars = bars_by_timeframe.get(config.timeframe, pd.DataFrame())
        source_report = source_reports.get(config.timeframe)
        source_valid = source_report.research_result_valid if source_report is not None else True
        synthetic_data_used = source_report.synthetic_data_used if source_report is not None else False
        if bars.empty or not source_valid:
            metrics = _empty_research_metrics("invalid_research_data_source" if not source_valid else "no_bars")
            walk_forward = empty_walk_forward_result(walk_forward_splits)
        else:
            trades, signal_frame = build_strategy_research_trades(
                config,
                bars=bars,
                settings=candidate_settings,
                buy_the_dip_features=buy_the_dip_features.get(config.timeframe, pd.DataFrame()),
                v3_features=v3_features.get(config.timeframe, pd.DataFrame()),
            )
            metrics = calculate_fee_aware_metrics(trades, candidate_settings, signal_frame=signal_frame)
            walk_forward = calculate_walk_forward_metrics(
                config,
                settings=candidate_settings,
                fold_inputs=walk_forward_inputs.get(config.timeframe, {}),
                min_trades_per_split=min_trades_per_split,
            )
        round_trip_cost = _metric_float(metrics.get("round_trip_estimated_cost_pct"))
        if round_trip_cost <= 0:
            round_trip_cost = estimated_round_trip_execution_cost_pct(candidate_settings)
        promotion_required = _metric_float(metrics.get("promotion_required_return_pct"))
        if promotion_required <= 0:
            promotion_required = estimated_promotion_required_return_pct(candidate_settings)
        take_profit_vs_cost_safe = config.take_profit_pct > round_trip_cost
        take_profit_vs_promotion_safe = config.take_profit_pct >= promotion_required
        readiness = paper_forward_readiness_gate(
            metrics,
            candidate_settings,
            fallback_prediction_used=synthetic_data_used,
            active_model_valid=active_model_valid,
            research_result_valid=source_valid,
            take_profit_pct=config.take_profit_pct,
            round_trip_estimated_cost_pct=round_trip_cost,
            promotion_required_return_pct=promotion_required,
            source_used=source_report.source_used if source_report is not None else None,
            walk_forward_passed=bool(walk_forward["walk_forward_passed"]),
            min_trades=min_trades,
        )
        concentration = single_trade_return_concentration(metrics.get("trade_details", []))
        rank_details = research_rank_details(metrics, readiness, concentration=concentration, walk_forward=walk_forward)
        rows.append(
            {
                "parameter_set_id": config.parameter_set_id,
                "strategy_name": config.strategy_name,
                "timeframe": config.timeframe,
                "take_profit_pct": config.take_profit_pct,
                "stop_loss_pct": config.stop_loss_pct,
                "max_hold_bars": config.max_hold_bars,
                "rsi_threshold": config.rsi_threshold,
                "zscore_threshold": config.zscore_threshold,
                "vwap_distance_threshold": config.vwap_distance_threshold,
                "drawdown_threshold": config.drawdown_threshold,
                "min_volume_zscore": config.min_volume_zscore,
                "reversal_confirmation_required": config.reversal_confirmation_required,
                "higher_timeframe_regime_filter": config.higher_timeframe_regime_filter,
                "support": config.support,
                "support_distance_pct": config.support_distance_pct,
                "pullback_min_pct": config.pullback_min_pct,
                "pullback_max_pct": config.pullback_max_pct,
                "rsi_min": config.rsi_min,
                "rsi_max": config.rsi_max,
                "confirmation": config.confirmation,
                "min_lower_wick_ratio": config.min_lower_wick_ratio,
                "min_volume_recovery": config.min_volume_recovery,
                "breakout_lookback": config.breakout_lookback,
                "consolidation_lookback": config.consolidation_lookback,
                "min_body_vs_avg": config.min_body_vs_avg,
                "min_recent_return_pct": config.min_recent_return_pct,
                "min_trend_strength": config.min_trend_strength,
                "max_atr_expansion": config.max_atr_expansion,
                "number_of_trades": int(metrics.get("number_of_trades", 0) or 0),
                "gross_return_pct": _metric_float(metrics.get("gross_return_pct")),
                "net_return_pct": _metric_float(metrics.get("net_return_pct")),
                "profit_factor_net": _profit_factor_value(metrics.get("profit_factor_net")),
                "max_drawdown_pct": _metric_float(metrics.get("max_drawdown_pct")),
                "win_rate_net": _metric_float(metrics.get("win_rate_net")),
                "expectancy": _metric_float(metrics.get("expectancy")),
                "round_trip_estimated_cost_pct": round_trip_cost,
                "promotion_required_return_pct": promotion_required,
                "take_profit_vs_cost_safe": take_profit_vs_cost_safe,
                "take_profit_vs_promotion_safe": take_profit_vs_promotion_safe,
                "gross_winners_became_net_losers": int(metrics.get("gross_winners_became_net_losers", 0) or 0),
                "single_trade_return_concentration": concentration,
                "fold_count": int(walk_forward["fold_count"]),
                "per_fold_net_return_pct": walk_forward["per_fold_net_return_pct"],
                "per_fold_profit_factor_net": walk_forward["per_fold_profit_factor_net"],
                "per_fold_number_of_trades": walk_forward["per_fold_number_of_trades"],
                "per_fold_max_drawdown_pct": walk_forward["per_fold_max_drawdown_pct"],
                "folds_profitable_count": int(walk_forward["folds_profitable_count"]),
                "folds_with_min_trades_count": int(walk_forward["folds_with_min_trades_count"]),
                "worst_fold_net_return_pct": _metric_float(walk_forward["worst_fold_net_return_pct"]),
                "median_fold_net_return_pct": _metric_float(walk_forward["median_fold_net_return_pct"]),
                "walk_forward_passed": bool(walk_forward["walk_forward_passed"]),
                "statistically_weak": rank_details["statistically_weak"],
                "trade_count_score": rank_details["trade_count_score"],
                "concentration_penalty": rank_details["concentration_penalty"],
                "reliability_score": rank_details["reliability_score"],
                "profit_factor_reliable": rank_details["profit_factor_reliable"],
                "adjusted_rank_score": rank_details["adjusted_rank_score"],
                "reason_ranked_lower_if_any": rank_details["reason_ranked_lower_if_any"],
                "source_used": source_report.source_used if source_report is not None else "unknown",
                "latest_timestamp": source_report.latest_timestamp if source_report is not None else None,
                "data_age_minutes": source_report.data_age_minutes if source_report is not None else None,
                "row_count": source_report.row_count if source_report is not None else int(len(bars)),
                "synthetic_data_used": synthetic_data_used,
                "research_result_valid": source_valid,
                "data_rejection_reason": source_report.rejection_reason if source_report is not None else None,
                "rejected_sources": list(source_report.rejected_sources) if source_report is not None else [],
                "fallback_prediction_used": synthetic_data_used,
                "active_model_valid": bool(active_model_valid),
                "active_model_status": (active_model_status or {}).get("active_model_status"),
                "economically_viable": readiness["economically_viable"],
                "research_promising": readiness["economically_viable"],
                "paper_forward_eligible": readiness["paper_forward_eligible"],
                "rejection_reasons": ";".join(readiness["rejection_reasons"]),
                "rank_score": rank_details["raw_rank_score"],
            }
        )
    return rows


def generate_research_configs(
    *,
    strategy: str = "all",
    max_buy_dip_configs: int = DEFAULT_MAX_BUY_DIP_CONFIGS,
    max_v3_configs: int = DEFAULT_MAX_V3_CONFIGS,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> list[ResearchConfig]:
    requested_timeframes = _resolve_requested_timeframes(strategy, timeframes)
    if strategy == TREND_PULLBACK_STRATEGY:
        return generate_trend_pullback_configs(timeframes=requested_timeframes)
    if strategy == BUY_THE_DIP_STRATEGY:
        return generate_buy_the_dip_configs(max_configs=max_buy_dip_configs, timeframes=requested_timeframes)
    if strategy == UPTREND_PULLBACK_STRATEGY:
        return generate_uptrend_pullback_configs(max_configs=max_v3_configs, timeframes=requested_timeframes)
    if strategy == VOLATILITY_BREAKOUT_STRATEGY:
        return generate_volatility_breakout_configs(max_configs=max_v3_configs, timeframes=requested_timeframes)
    legacy = (
        generate_trend_pullback_configs(timeframes=requested_timeframes)
        + generate_buy_the_dip_configs(max_configs=max_buy_dip_configs, timeframes=requested_timeframes)
    )
    v3 = generate_v3_configs(max_configs=max_v3_configs, timeframes=requested_timeframes)
    return legacy + v3


def generate_trend_pullback_configs(*, timeframes: tuple[str, ...] | list[str] | None = None) -> list[ResearchConfig]:
    configs: list[ResearchConfig] = []
    index = 0
    requested_timeframes = tuple(
        timeframe for timeframe in (timeframes or LEGACY_RESEARCH_TIMEFRAMES) if timeframe in LEGACY_RESEARCH_TIMEFRAMES
    )
    for timeframe, take_profit_pct, stop_loss_pct, max_hold_bars in product(
        requested_timeframes,
        TAKE_PROFIT_VALUES,
        STOP_LOSS_VALUES,
        MAX_HOLD_BARS_VALUES,
    ):
        configs.append(
            ResearchConfig(
                parameter_set_id=f"htf_{index:03d}",
                strategy_name=TREND_PULLBACK_STRATEGY,
                timeframe=timeframe,
                take_profit_pct=float(take_profit_pct),
                stop_loss_pct=float(stop_loss_pct),
                max_hold_bars=int(max_hold_bars),
            )
        )
        index += 1
    return configs


def generate_buy_the_dip_configs(
    *,
    max_configs: int = DEFAULT_MAX_BUY_DIP_CONFIGS,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> list[ResearchConfig]:
    raw_configs: list[ResearchConfig] = []
    requested_timeframes = tuple(
        timeframe for timeframe in (timeframes or LEGACY_RESEARCH_TIMEFRAMES) if timeframe in LEGACY_RESEARCH_TIMEFRAMES
    )
    for timeframe, take_profit_pct, stop_loss_pct, max_hold_bars, profile in product(
        requested_timeframes,
        BUY_THE_DIP_TAKE_PROFIT_VALUES,
        BUY_THE_DIP_STOP_LOSS_VALUES,
        BUY_THE_DIP_MAX_HOLD_BARS_VALUES,
        generate_buy_the_dip_signal_profiles(),
    ):
        raw_index = len(raw_configs)
        raw_configs.append(
            ResearchConfig(
                parameter_set_id=f"btd_{raw_index:05d}",
                strategy_name=BUY_THE_DIP_STRATEGY,
                timeframe=timeframe,
                take_profit_pct=float(take_profit_pct),
                stop_loss_pct=float(stop_loss_pct),
                max_hold_bars=int(max_hold_bars),
                rsi_threshold=float(profile["rsi_threshold"]),
                zscore_threshold=float(profile["zscore_threshold"]),
                vwap_distance_threshold=float(profile["vwap_distance_threshold"]),
                drawdown_threshold=float(profile["drawdown_threshold"]),
                min_volume_zscore=float(profile["min_volume_zscore"]),
                reversal_confirmation_required=bool(profile["reversal_confirmation_required"]),
                higher_timeframe_regime_filter=bool(profile["higher_timeframe_regime_filter"]),
            )
        )
    if len(raw_configs) <= max_configs:
        return raw_configs
    return [raw_configs[index] for index in _evenly_spaced_indexes(len(raw_configs), max_configs)]


def generate_v3_configs(
    *,
    max_configs: int = DEFAULT_MAX_V3_CONFIGS,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> list[ResearchConfig]:
    configs = (
        generate_uptrend_pullback_configs(max_configs=max_configs, timeframes=timeframes)
        + generate_volatility_breakout_configs(max_configs=max_configs, timeframes=timeframes)
    )
    if len(configs) <= max_configs:
        return configs
    return [configs[index] for index in _evenly_spaced_indexes(len(configs), max_configs)]


def generate_uptrend_pullback_configs(
    *,
    max_configs: int = DEFAULT_MAX_V3_CONFIGS,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> list[ResearchConfig]:
    profiles = (
        {
            "support": "ema20",
            "support_distance_pct": 0.006,
            "pullback_min_pct": 0.01,
            "pullback_max_pct": 0.05,
            "rsi_min": 35.0,
            "rsi_max": 55.0,
            "confirmation": "bullish_close",
            "min_lower_wick_ratio": 0.20,
            "min_volume_recovery": -0.75,
            "min_trend_strength": 0.0,
            "max_atr_expansion": 2.2,
        },
        {
            "support": "ema20",
            "support_distance_pct": 0.010,
            "pullback_min_pct": 0.015,
            "pullback_max_pct": 0.08,
            "rsi_min": 38.0,
            "rsi_max": 58.0,
            "confirmation": "recover_prior_high",
            "min_lower_wick_ratio": 0.15,
            "min_volume_recovery": -0.50,
            "min_trend_strength": 0.0,
            "max_atr_expansion": 2.5,
        },
        {
            "support": "ema50",
            "support_distance_pct": 0.010,
            "pullback_min_pct": 0.02,
            "pullback_max_pct": 0.08,
            "rsi_min": 35.0,
            "rsi_max": 55.0,
            "confirmation": "lower_wick",
            "min_lower_wick_ratio": 0.30,
            "min_volume_recovery": -0.25,
            "min_trend_strength": 0.0,
            "max_atr_expansion": 2.5,
        },
        {
            "support": "ema50",
            "support_distance_pct": 0.014,
            "pullback_min_pct": 0.015,
            "pullback_max_pct": 0.06,
            "rsi_min": 40.0,
            "rsi_max": 58.0,
            "confirmation": "bullish_close",
            "min_lower_wick_ratio": 0.20,
            "min_volume_recovery": -0.75,
            "min_trend_strength": 0.0,
            "max_atr_expansion": 2.0,
        },
        {
            "support": "vwap",
            "support_distance_pct": 0.008,
            "pullback_min_pct": 0.01,
            "pullback_max_pct": 0.05,
            "rsi_min": 38.0,
            "rsi_max": 55.0,
            "confirmation": "recover_prior_high",
            "min_lower_wick_ratio": 0.15,
            "min_volume_recovery": -0.50,
            "min_trend_strength": 0.0,
            "max_atr_expansion": 2.3,
        },
        {
            "support": "vwap",
            "support_distance_pct": 0.012,
            "pullback_min_pct": 0.02,
            "pullback_max_pct": 0.08,
            "rsi_min": 35.0,
            "rsi_max": 52.0,
            "confirmation": "lower_wick",
            "min_lower_wick_ratio": 0.30,
            "min_volume_recovery": -0.25,
            "min_trend_strength": 0.0,
            "max_atr_expansion": 2.5,
        },
    )
    raw_configs: list[ResearchConfig] = []
    requested_timeframes = tuple(
        timeframe for timeframe in (timeframes or V3_RESEARCH_TIMEFRAMES) if timeframe in V3_RESEARCH_TIMEFRAMES
    )
    for timeframe, take_profit_pct, stop_loss_pct, max_hold_bars, profile in product(
        requested_timeframes,
        UPTREND_PULLBACK_TAKE_PROFIT_VALUES,
        UPTREND_PULLBACK_STOP_LOSS_VALUES,
        V3_MAX_HOLD_BARS_VALUES,
        profiles,
    ):
        raw_index = len(raw_configs)
        raw_configs.append(
            ResearchConfig(
                parameter_set_id=f"utp_{raw_index:05d}",
                strategy_name=UPTREND_PULLBACK_STRATEGY,
                timeframe=timeframe,
                take_profit_pct=float(take_profit_pct),
                stop_loss_pct=float(stop_loss_pct),
                max_hold_bars=int(max_hold_bars),
                support=str(profile["support"]),
                support_distance_pct=float(profile["support_distance_pct"]),
                pullback_min_pct=float(profile["pullback_min_pct"]),
                pullback_max_pct=float(profile["pullback_max_pct"]),
                rsi_min=float(profile["rsi_min"]),
                rsi_max=float(profile["rsi_max"]),
                confirmation=str(profile["confirmation"]),
                min_lower_wick_ratio=float(profile["min_lower_wick_ratio"]),
                min_volume_recovery=float(profile["min_volume_recovery"]),
                min_trend_strength=float(profile["min_trend_strength"]),
                max_atr_expansion=float(profile["max_atr_expansion"]),
            )
        )
    if len(raw_configs) <= max_configs:
        return raw_configs
    return [raw_configs[index] for index in _evenly_spaced_indexes(len(raw_configs), max_configs)]


def generate_volatility_breakout_configs(
    *,
    max_configs: int = DEFAULT_MAX_V3_CONFIGS,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> list[ResearchConfig]:
    profiles = (
        {
            "breakout_lookback": 20,
            "consolidation_lookback": 12,
            "min_volume_zscore": 0.5,
            "min_body_vs_avg": 1.0,
            "min_recent_return_pct": 0.002,
            "min_trend_strength": 0.0,
            "max_atr_expansion": 2.6,
        },
        {
            "breakout_lookback": 24,
            "consolidation_lookback": 16,
            "min_volume_zscore": 0.75,
            "min_body_vs_avg": 1.1,
            "min_recent_return_pct": 0.003,
            "min_trend_strength": 0.0,
            "max_atr_expansion": 2.5,
        },
        {
            "breakout_lookback": 32,
            "consolidation_lookback": 20,
            "min_volume_zscore": 1.0,
            "min_body_vs_avg": 1.2,
            "min_recent_return_pct": 0.004,
            "min_trend_strength": 0.1,
            "max_atr_expansion": 2.3,
        },
        {
            "breakout_lookback": 48,
            "consolidation_lookback": 24,
            "min_volume_zscore": 0.5,
            "min_body_vs_avg": 1.0,
            "min_recent_return_pct": 0.002,
            "min_trend_strength": 0.0,
            "max_atr_expansion": 2.8,
        },
        {
            "breakout_lookback": 48,
            "consolidation_lookback": 32,
            "min_volume_zscore": 1.0,
            "min_body_vs_avg": 1.25,
            "min_recent_return_pct": 0.003,
            "min_trend_strength": 0.1,
            "max_atr_expansion": 2.4,
        },
        {
            "breakout_lookback": 64,
            "consolidation_lookback": 32,
            "min_volume_zscore": 1.25,
            "min_body_vs_avg": 1.35,
            "min_recent_return_pct": 0.004,
            "min_trend_strength": 0.15,
            "max_atr_expansion": 2.2,
        },
    )
    raw_configs: list[ResearchConfig] = []
    requested_timeframes = tuple(
        timeframe for timeframe in (timeframes or V3_RESEARCH_TIMEFRAMES) if timeframe in V3_RESEARCH_TIMEFRAMES
    )
    for timeframe, take_profit_pct, stop_loss_pct, max_hold_bars, profile in product(
        requested_timeframes,
        VOLATILITY_BREAKOUT_TAKE_PROFIT_VALUES,
        VOLATILITY_BREAKOUT_STOP_LOSS_VALUES,
        V3_MAX_HOLD_BARS_VALUES,
        profiles,
    ):
        raw_index = len(raw_configs)
        raw_configs.append(
            ResearchConfig(
                parameter_set_id=f"vbo_{raw_index:05d}",
                strategy_name=VOLATILITY_BREAKOUT_STRATEGY,
                timeframe=timeframe,
                take_profit_pct=float(take_profit_pct),
                stop_loss_pct=float(stop_loss_pct),
                max_hold_bars=int(max_hold_bars),
                min_volume_zscore=float(profile["min_volume_zscore"]),
                breakout_lookback=int(profile["breakout_lookback"]),
                consolidation_lookback=int(profile["consolidation_lookback"]),
                min_body_vs_avg=float(profile["min_body_vs_avg"]),
                min_recent_return_pct=float(profile["min_recent_return_pct"]),
                min_trend_strength=float(profile["min_trend_strength"]),
                max_atr_expansion=float(profile["max_atr_expansion"]),
            )
        )
    if len(raw_configs) <= max_configs:
        return raw_configs
    return [raw_configs[index] for index in _evenly_spaced_indexes(len(raw_configs), max_configs)]


def generate_buy_the_dip_signal_profiles() -> tuple[dict[str, Any], ...]:
    profiles: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add_profile(profile: dict[str, Any]) -> None:
        key = tuple(sorted(profile.items()))
        if key not in seen:
            seen.add(key)
            profiles.append(profile)

    for profile in BUY_THE_DIP_SIGNAL_PROFILES:
        add_profile(dict(profile))

    severity_profiles = zip(
        reversed(BUY_THE_DIP_V2_PROFILE_VALUES["rsi_threshold"]),
        BUY_THE_DIP_V2_PROFILE_VALUES["zscore_threshold"],
        BUY_THE_DIP_V2_PROFILE_VALUES["vwap_distance_threshold"],
        BUY_THE_DIP_V2_PROFILE_VALUES["drawdown_threshold"],
        strict=True,
    )
    volume_values = BUY_THE_DIP_V2_PROFILE_VALUES["min_volume_zscore"]
    reversal_values = BUY_THE_DIP_V2_PROFILE_VALUES["reversal_confirmation_required"]
    regime_values = BUY_THE_DIP_V2_PROFILE_VALUES["higher_timeframe_regime_filter"]
    for rsi, zscore, vwap, drawdown in severity_profiles:
        for volume, reversal_required, regime_filter in product(volume_values, reversal_values, regime_values):
            add_profile(
                {
                    "rsi_threshold": float(rsi),
                    "zscore_threshold": float(zscore),
                    "vwap_distance_threshold": float(vwap),
                    "drawdown_threshold": float(drawdown),
                    "min_volume_zscore": float(volume),
                    "reversal_confirmation_required": bool(reversal_required),
                    "higher_timeframe_regime_filter": bool(regime_filter),
                }
            )
    return tuple(profiles)


def _evenly_spaced_indexes(total: int, desired: int) -> list[int]:
    if desired <= 0:
        raise ValueError("desired config count must be positive")
    if total <= desired:
        return list(range(total))
    if desired == 1:
        return [0]
    indexes = {0, total - 1}
    step = (total - 1) / (desired - 1)
    for offset in range(desired):
        indexes.add(int(round(offset * step)))
    if len(indexes) > desired:
        return sorted(indexes)[:desired]
    candidate = 0
    while len(indexes) < desired:
        indexes.add(candidate)
        candidate += 1
    return sorted(indexes)


def build_trend_pullback_research_trades(
    bars: pd.DataFrame,
    settings: Settings,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = add_features(bars).dropna().reset_index(drop=True)
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    features["orderbook_spread"] = max(0.0, float(settings.max_spread_bps)) / 10_000
    features["quote_imbalance"] = 0.0
    features["scalping_spread_bps"] = max(0.0, float(settings.max_spread_bps))
    features["scalping_quote_imbalance"] = 0.0
    regime_filter = MarketRegimeFilter(settings)
    strategy = TrendPullbackStrategy(settings)
    signal_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    index = 0
    while index < len(features) - config.max_hold_bars - 1:
        row = features.iloc[index]
        regime = regime_filter.detect(row)
        allowed_by_regime, regime_reason = regime_filter.allows(regime, strategy.name)
        context = MarketContext(regime=regime, risk_permits_evaluation=allowed_by_regime, risk_reason=regime_reason)
        signal = strategy.generate_signal(
            feature_row=row,
            prediction=None,
            position=PositionState(),
            quote=None,
            market_context=context,
        )
        signal_row = _signal_row(row, signal=signal, regime=regime.regime, config=config)
        if signal.action == "buy":
            exit_result = resolve_research_exit(features, index, config)
            trade_row = {
                **signal_row,
                "buy_quality_label": int(exit_result["gross_return"] > 0),
                "buy_exit_return_pct": float(exit_result["gross_return"]),
                "buy_exit_reason": exit_result["exit_reason"],
                "buy_hold_bars": float(exit_result["hold_bars"]),
                "backtest_exit_high": float(exit_result["exit_high"]),
                "backtest_exit_low": float(exit_result["exit_low"]),
            }
            trade_rows.append(trade_row)
            signal_row["entry_allowed"] = True
            signal_rows.append(signal_row)
            index += max(1, int(exit_result["hold_bars"]))
            continue
        signal_row["entry_allowed"] = False
        signal_rows.append(signal_row)
        index += 1
    return pd.DataFrame(trade_rows), pd.DataFrame(signal_rows)


build_research_trades = build_trend_pullback_research_trades


def prepare_v3_features(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    features = add_features(bars)
    close = features["close"]
    ema_20 = features["ema_20"]
    ema_50 = features["ema_50"]
    ema_200 = close.ewm(span=200, adjust=False).mean()
    candle_range = (features["high"] - features["low"]).replace(0, np.nan)
    lower_body = pd.concat([features["open"], features["close"]], axis=1).min(axis=1)
    upper_body = pd.concat([features["open"], features["close"]], axis=1).max(axis=1)
    body_pct = (features["close"] - features["open"]).abs() / features["close"].replace(0, np.nan)
    body_mean_20 = body_pct.rolling(20).mean()
    rolling_high_20 = close.rolling(20).max()
    rolling_high_50 = close.rolling(50).max()
    rolling_low_20 = close.rolling(20).min()
    atr_mean_20 = features["atr_14"].rolling(20).mean()
    features["ema_200"] = ema_200
    features["ema_20_slope_5"] = ema_20.pct_change(5)
    features["ema_50_slope_5"] = ema_50.pct_change(5)
    features["ema_200_distance"] = (close - ema_200) / close.replace(0, np.nan)
    features["ema_20_above_50"] = ema_20 > ema_50
    features["ema_50_above_200"] = ema_50 > ema_200
    features["close_above_ema_200"] = close > ema_200
    features["recent_high_50"] = rolling_high_50
    features["pullback_from_high_50"] = (rolling_high_50 - close) / rolling_high_50.replace(0, np.nan)
    features["support_distance_ema20_abs"] = ((close - ema_20) / close.replace(0, np.nan)).abs()
    features["support_distance_ema50_abs"] = ((close - ema_50) / close.replace(0, np.nan)).abs()
    features["support_distance_vwap_abs"] = ((close - features["vwap"]) / close.replace(0, np.nan)).abs()
    features["lower_wick_ratio"] = (lower_body - features["low"]) / candle_range
    features["upper_wick_ratio"] = (features["high"] - upper_body) / candle_range
    features["close_position_in_candle"] = (features["close"] - features["low"]) / candle_range
    features["bullish_close"] = features["close"] > features["open"]
    features["recovers_prior_high"] = features["close"] > features["prev_high"]
    features["body_pct"] = body_pct
    features["body_vs_avg_20"] = body_pct / body_mean_20.replace(0, np.nan)
    features["prior_rolling_high_20"] = rolling_high_20.shift(1)
    features["prior_rolling_high_24"] = close.rolling(24).max().shift(1)
    features["prior_rolling_high_32"] = close.rolling(32).max().shift(1)
    features["prior_rolling_high_48"] = close.rolling(48).max().shift(1)
    features["prior_rolling_high_64"] = close.rolling(64).max().shift(1)
    features["rolling_low_20"] = rolling_low_20
    features["range_width_12"] = (close.rolling(12).max() - close.rolling(12).min()) / close.replace(0, np.nan)
    features["range_width_16"] = (close.rolling(16).max() - close.rolling(16).min()) / close.replace(0, np.nan)
    features["range_width_20"] = (close.rolling(20).max() - close.rolling(20).min()) / close.replace(0, np.nan)
    features["range_width_24"] = (close.rolling(24).max() - close.rolling(24).min()) / close.replace(0, np.nan)
    features["range_width_32"] = (close.rolling(32).max() - close.rolling(32).min()) / close.replace(0, np.nan)
    features["atr_expansion_20"] = features["atr_14"] / atr_mean_20.replace(0, np.nan)
    features["atr_downside_explosion"] = (features["log_return_3"] < -0.025) & (features["atr_expansion_20"] > 2.0)
    features["extreme_crash_candle"] = (
        (features["close_open_pct"] <= -0.03)
        | (features["high_low_range_pct"] >= 0.08)
        | (features["log_return_5"] <= -0.05)
    )
    features["abnormal_spread"] = False
    if "scalping_spread_bps" in features:
        features["abnormal_spread"] = features["scalping_spread_bps"] > 0
    return features.replace([np.inf, -np.inf], np.nan)


def build_strategy_research_trades(
    config: ResearchConfig,
    *,
    bars: pd.DataFrame,
    settings: Settings,
    buy_the_dip_features: pd.DataFrame | None = None,
    v3_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.strategy_name == BUY_THE_DIP_STRATEGY:
        return build_buy_the_dip_research_trades(
            buy_the_dip_features if buy_the_dip_features is not None else prepare_buy_the_dip_features(bars),
            settings,
            config,
        )
    if config.strategy_name == UPTREND_PULLBACK_STRATEGY:
        return build_uptrend_pullback_research_trades(
            v3_features if v3_features is not None else prepare_v3_features(bars),
            settings,
            config,
        )
    if config.strategy_name == VOLATILITY_BREAKOUT_STRATEGY:
        return build_volatility_breakout_research_trades(
            v3_features if v3_features is not None else prepare_v3_features(bars),
            settings,
            config,
        )
    return build_trend_pullback_research_trades(bars, settings, config)


def build_uptrend_pullback_research_trades(
    features: pd.DataFrame,
    settings: Settings,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    support_column = {
        "ema20": "support_distance_ema20_abs",
        "ema50": "support_distance_ema50_abs",
        "vwap": "support_distance_vwap_abs",
    }.get(str(config.support or "ema20"))
    if support_column is None:
        return pd.DataFrame(), pd.DataFrame()
    required = [
        "timestamp",
        "close",
        "open",
        "high",
        "low",
        "ema_20_slope_5",
        "ema_50_above_200",
        "close_above_ema_200",
        "pullback_from_high_50",
        support_column,
        "rsi_14",
        "bullish_close",
        "recovers_prior_high",
        "lower_wick_ratio",
        "volume_zscore_20",
        "atr_expansion_20",
        "atr_downside_explosion",
        "extreme_crash_candle",
    ]
    data = features.dropna(subset=required).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    uptrend = (
        data["close_above_ema_200"].astype(bool)
        & data["ema_50_above_200"].astype(bool)
        & (data["ema_20_slope_5"] > 0)
    )
    pullback = (
        (data[support_column] <= float(config.support_distance_pct or 0.01))
        & (data["pullback_from_high_50"] >= float(config.pullback_min_pct or 0.01))
        & (data["pullback_from_high_50"] <= float(config.pullback_max_pct or 0.08))
        & (data["rsi_14"] >= float(config.rsi_min or 35.0))
        & (data["rsi_14"] <= float(config.rsi_max or 55.0))
    )
    confirmation_mode = str(config.confirmation or "bullish_close")
    if confirmation_mode == "recover_prior_high":
        confirmation = data["recovers_prior_high"].astype(bool) | (
            data["bullish_close"].astype(bool)
            & (data["lower_wick_ratio"] >= float(config.min_lower_wick_ratio or 0.20))
        )
    elif confirmation_mode == "lower_wick":
        confirmation = (
            (data["lower_wick_ratio"] >= float(config.min_lower_wick_ratio or 0.30))
            & (data["close_position_in_candle"] >= 0.45)
        )
    else:
        confirmation = data["bullish_close"].astype(bool)
    risk_ok = (
        ~data["extreme_crash_candle"].astype(bool)
        & ~data["atr_downside_explosion"].astype(bool)
        & (data["atr_expansion_20"] <= float(config.max_atr_expansion or 2.5))
        & (data["volume_zscore_20"] >= float(config.min_volume_recovery or -0.75))
    )
    mask = uptrend & pullback & confirmation & risk_ok
    return _build_research_trades_from_mask(
        data,
        mask,
        config,
        strategy_name=UPTREND_PULLBACK_STRATEGY,
        entry_reason="uptrend_pullback_support_reclaim_candidate",
        regime="higher_timeframe_uptrend_pullback",
    )


def build_volatility_breakout_research_trades(
    features: pd.DataFrame,
    settings: Settings,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    breakout_lookback = int(config.breakout_lookback or 20)
    consolidation_lookback = int(config.consolidation_lookback or 20)
    high_column = f"prior_rolling_high_{breakout_lookback}"
    range_column = f"range_width_{consolidation_lookback}"
    if high_column not in features or range_column not in features:
        return pd.DataFrame(), pd.DataFrame()
    required = [
        "timestamp",
        "close",
        "open",
        "high",
        "low",
        "ema_20",
        "ema_50",
        "ema_50_slope_5",
        "log_return_3",
        "log_return_5",
        "volume_zscore_20",
        "atr_expansion_20",
        "body_vs_avg_20",
        "trend_strength_20",
        high_column,
        range_column,
        "extreme_crash_candle",
    ]
    data = features.dropna(subset=required).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    breakout = data["close"] > data[high_column]
    momentum = (
        (data["ema_20"] > data["ema_50"])
        & (data["ema_50_slope_5"] > 0)
        & (data["log_return_3"] >= float(config.min_recent_return_pct or 0.002))
        & (data["log_return_5"] > 0)
    )
    volume_volatility = (
        (data["volume_zscore_20"] >= float(config.min_volume_zscore or 0.5))
        & (data["atr_expansion_20"] >= 0.85)
        & (data["atr_expansion_20"] <= float(config.max_atr_expansion or 2.5))
        & (data["body_vs_avg_20"] >= float(config.min_body_vs_avg or 1.0))
    )
    trend_strength = data["trend_strength_20"].abs() >= float(config.min_trend_strength or 0.0)
    not_choppy = data[range_column] <= max(0.12, float(config.take_profit_pct) * 5)
    mask = breakout & momentum & volume_volatility & trend_strength & not_choppy & ~data["extreme_crash_candle"].astype(bool)
    return _build_research_trades_from_mask(
        data,
        mask,
        config,
        strategy_name=VOLATILITY_BREAKOUT_STRATEGY,
        entry_reason="volatility_breakout_momentum_continuation_candidate",
        regime="higher_timeframe_momentum_breakout",
    )


def _build_research_trades_from_mask(
    data: pd.DataFrame,
    mask: pd.Series,
    config: ResearchConfig,
    *,
    strategy_name: str,
    entry_reason: str,
    regime: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    last_exit_index = -1
    for entry_index in data.index[mask].tolist():
        if entry_index <= last_exit_index:
            continue
        if entry_index >= len(data) - int(config.max_hold_bars) - 1:
            continue
        row = data.iloc[entry_index]
        signal_row = _v3_signal_row(
            row,
            config=config,
            strategy_name=strategy_name,
            entry_reason=entry_reason,
            regime=regime,
        )
        signal_rows.append(signal_row)
        exit_result = resolve_research_exit(data, entry_index, config)
        trade_rows.append(
            {
                **signal_row,
                "buy_quality_label": int(exit_result["gross_return"] > 0),
                "buy_exit_return_pct": float(exit_result["gross_return"]),
                "buy_exit_reason": exit_result["exit_reason"],
                "buy_hold_bars": float(exit_result["hold_bars"]),
                "backtest_exit_high": float(exit_result["exit_high"]),
                "backtest_exit_low": float(exit_result["exit_low"]),
            }
        )
        last_exit_index = entry_index + max(1, int(exit_result["hold_bars"]))
    return pd.DataFrame(trade_rows), pd.DataFrame(signal_rows)


def prepare_buy_the_dip_features(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    features = add_features(bars)
    close = features["close"]
    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    rolling_high = close.rolling(50).max()
    rolling_low = close.rolling(20).min()
    candle_range = (features["high"] - features["low"]).replace(0, np.nan)
    lower_body = pd.concat([features["open"], features["close"]], axis=1).min(axis=1)
    features["rolling_zscore_20"] = (close - rolling_mean) / rolling_std.replace(0, np.nan)
    features["recent_drawdown_50"] = (rolling_high - close) / rolling_high.replace(0, np.nan)
    features["distance_from_rolling_low_20"] = (close - rolling_low) / rolling_low.replace(0, np.nan)
    features["lower_wick_ratio"] = (lower_body - features["low"]) / candle_range
    features["close_position_in_candle"] = (features["close"] - features["low"]) / candle_range
    features["rsi_not_collapsing"] = features["rsi_14"] >= (features["prev_rsi_14"] - 8.0)
    features["reversal_confirmation"] = (
        (features["close_open_pct"] > 0)
        | (features["log_return_1"] > 0)
        | ((features["lower_wick_ratio"] >= 0.30) & (features["close_position_in_candle"] >= 0.45))
    )
    features["severe_breakdown_mode"] = (
        (features["log_return_20"] <= -0.035)
        | (features["rolling_max_drawdown_50"] <= -0.08)
        | (features["volatility_20"] >= 0.05)
    )
    return features.replace([np.inf, -np.inf], np.nan)


def build_buy_the_dip_research_trades(
    features: pd.DataFrame,
    settings: Settings,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = [
        "timestamp",
        "close",
        "open",
        "high",
        "low",
        "vwap_distance",
        "ema_20_distance",
        "rsi_14",
        "prev_rsi_14",
        "rolling_zscore_20",
        "recent_drawdown_50",
        "volume_zscore_20",
        "lower_wick_ratio",
        "close_position_in_candle",
        "reversal_confirmation",
        "rsi_not_collapsing",
        "severe_breakdown_mode",
    ]
    data = features.dropna(subset=required).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    stretch = (
        (data["vwap_distance"] <= float(config.vwap_distance_threshold or 0.0))
        | (data["ema_20_distance"] <= float(config.vwap_distance_threshold or 0.0))
        | (data["rolling_zscore_20"] <= float(config.zscore_threshold or -2.0))
        | (data["bb_close_position"] <= 0.15)
    )
    mask = (
        stretch
        & (data["rsi_14"] <= float(config.rsi_threshold or 30.0))
        & data["rsi_not_collapsing"].astype(bool)
        & (data["recent_drawdown_50"] >= float(config.drawdown_threshold or 0.01))
        & (data["volume_zscore_20"] >= float(config.min_volume_zscore or 1.0))
    )
    if config.reversal_confirmation_required:
        mask &= data["reversal_confirmation"].astype(bool)
    if config.higher_timeframe_regime_filter:
        mask &= ~data["severe_breakdown_mode"].astype(bool)

    signal_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    last_exit_index = -1
    for entry_index in data.index[mask].tolist():
        if entry_index <= last_exit_index:
            continue
        if entry_index >= len(data) - int(config.max_hold_bars) - 1:
            continue
        row = data.iloc[entry_index]
        signal_row = _buy_the_dip_signal_row(row, config=config)
        signal_rows.append(signal_row)
        exit_result = resolve_research_exit(data, entry_index, config)
        trade_rows.append(
            {
                **signal_row,
                "buy_quality_label": int(exit_result["gross_return"] > 0),
                "buy_exit_return_pct": float(exit_result["gross_return"]),
                "buy_exit_reason": exit_result["exit_reason"],
                "buy_hold_bars": float(exit_result["hold_bars"]),
                "backtest_exit_high": float(exit_result["exit_high"]),
                "backtest_exit_low": float(exit_result["exit_low"]),
            }
        )
        last_exit_index = entry_index + max(1, int(exit_result["hold_bars"]))
    return pd.DataFrame(trade_rows), pd.DataFrame(signal_rows)


def resolve_research_exit(features: pd.DataFrame, entry_index: int, config: ResearchConfig) -> dict[str, Any]:
    entry_close = float(features.iloc[entry_index]["close"])
    take_profit_price = entry_close * (1 + config.take_profit_pct)
    stop_loss_price = entry_close * (1 - config.stop_loss_pct)
    max_exit_index = min(len(features) - 1, entry_index + config.max_hold_bars)
    for offset, row_index in enumerate(range(entry_index + 1, max_exit_index + 1), start=1):
        row = features.iloc[row_index]
        high = float(row["high"])
        low = float(row["low"])
        hit_take_profit = high >= take_profit_price
        hit_stop_loss = low <= stop_loss_price
        if hit_take_profit and hit_stop_loss:
            return _exit_result(-config.stop_loss_pct, "ambiguous_stop_first", offset, high, low)
        if hit_stop_loss:
            return _exit_result(-config.stop_loss_pct, "research_stop_loss", offset, high, low)
        if hit_take_profit:
            return _exit_result(config.take_profit_pct, "research_take_profit", offset, high, low)
    exit_row = features.iloc[max_exit_index]
    gross_return = float(exit_row["close"] / entry_close - 1)
    return _exit_result(
        gross_return,
        "research_max_hold",
        max(1, max_exit_index - entry_index),
        float(exit_row["high"]),
        float(exit_row["low"]),
    )


def build_walk_forward_inputs(
    bars_by_timeframe: dict[str, pd.DataFrame],
    *,
    buy_the_dip_features: dict[str, pd.DataFrame],
    v3_features: dict[str, pd.DataFrame],
    splits: int,
) -> dict[str, dict[str, list[pd.DataFrame]]]:
    out: dict[str, dict[str, list[pd.DataFrame]]] = {}
    for timeframe, bars in bars_by_timeframe.items():
        out[timeframe] = {
            "raw": chronological_walk_forward_splits(bars, splits=splits),
            "buy_the_dip": chronological_walk_forward_splits(
                buy_the_dip_features.get(timeframe, pd.DataFrame()), splits=splits
            ),
            "v3": chronological_walk_forward_splits(v3_features.get(timeframe, pd.DataFrame()), splits=splits),
        }
    return out


def chronological_walk_forward_splits(data: pd.DataFrame, *, splits: int) -> list[pd.DataFrame]:
    if splits <= 0:
        raise ValueError("walk-forward splits must be positive")
    if data.empty:
        return []
    ordered = data.sort_values("timestamp").reset_index(drop=True) if "timestamp" in data else data.reset_index(drop=True)
    indexes = np.array_split(np.arange(len(ordered)), int(splits))
    return [ordered.iloc[index].reset_index(drop=True) for index in indexes if len(index)]


def calculate_walk_forward_metrics(
    config: ResearchConfig,
    *,
    settings: Settings,
    fold_inputs: dict[str, list[pd.DataFrame]],
    min_trades_per_split: int,
) -> dict[str, Any]:
    if not fold_inputs:
        return empty_walk_forward_result(0)
    fold_metrics: list[dict[str, Any]] = []
    if config.strategy_name == BUY_THE_DIP_STRATEGY:
        folds = fold_inputs.get("buy_the_dip", [])
    elif config.strategy_name in V3_STRATEGIES:
        folds = fold_inputs.get("v3", [])
    else:
        folds = fold_inputs.get("raw", [])
    for fold in folds:
        if fold.empty:
            continue
        if config.strategy_name == BUY_THE_DIP_STRATEGY:
            trades, signals = build_buy_the_dip_research_trades(fold, settings, config)
        elif config.strategy_name == UPTREND_PULLBACK_STRATEGY:
            trades, signals = build_uptrend_pullback_research_trades(fold, settings, config)
        elif config.strategy_name == VOLATILITY_BREAKOUT_STRATEGY:
            trades, signals = build_volatility_breakout_research_trades(fold, settings, config)
        else:
            trades, signals = build_trend_pullback_research_trades(fold, settings, config)
        fold_metrics.append(calculate_fee_aware_metrics(trades, settings, signal_frame=signals))
    return summarize_walk_forward_metrics(fold_metrics, min_trades_per_split=min_trades_per_split)


def summarize_walk_forward_metrics(
    fold_metrics: list[dict[str, Any]],
    *,
    min_trades_per_split: int,
) -> dict[str, Any]:
    if not fold_metrics:
        return empty_walk_forward_result(0)
    net_returns = [_metric_float(metrics.get("net_return_pct")) for metrics in fold_metrics]
    profit_factors = [_profit_factor_value(metrics.get("profit_factor_net")) for metrics in fold_metrics]
    trades = [int(metrics.get("number_of_trades", 0) or 0) for metrics in fold_metrics]
    drawdowns = [_metric_float(metrics.get("max_drawdown_pct")) for metrics in fold_metrics]
    fold_count = len(fold_metrics)
    folds_profitable = sum(1 for value in net_returns if value > 0)
    folds_with_min_trades = sum(1 for value in trades if value >= int(min_trades_per_split))
    required_robust_folds = max(2, math.ceil(fold_count / 2))
    walk_forward_passed = (
        fold_count >= 2
        and folds_profitable >= required_robust_folds
        and folds_with_min_trades >= required_robust_folds
        and sum(trades) >= MIN_RESEARCH_TRADES
    )
    return {
        "fold_count": fold_count,
        "per_fold_net_return_pct": net_returns,
        "per_fold_profit_factor_net": profit_factors,
        "per_fold_number_of_trades": trades,
        "per_fold_max_drawdown_pct": drawdowns,
        "folds_profitable_count": folds_profitable,
        "folds_with_min_trades_count": folds_with_min_trades,
        "worst_fold_net_return_pct": min(net_returns) if net_returns else 0.0,
        "median_fold_net_return_pct": float(np.median(net_returns)) if net_returns else 0.0,
        "walk_forward_passed": bool(walk_forward_passed),
    }


def empty_walk_forward_result(fold_count: int) -> dict[str, Any]:
    return {
        "fold_count": int(fold_count),
        "per_fold_net_return_pct": [],
        "per_fold_profit_factor_net": [],
        "per_fold_number_of_trades": [],
        "per_fold_max_drawdown_pct": [],
        "folds_profitable_count": 0,
        "folds_with_min_trades_count": 0,
        "worst_fold_net_return_pct": 0.0,
        "median_fold_net_return_pct": 0.0,
        "walk_forward_passed": False,
    }


def paper_forward_readiness_gate(
    metrics: dict[str, Any],
    settings: Settings,
    *,
    fallback_prediction_used: bool,
    active_model_valid: bool,
    research_result_valid: bool = True,
    take_profit_pct: float | None = None,
    round_trip_estimated_cost_pct: float | None = None,
    promotion_required_return_pct: float | None = None,
    source_used: str | None = None,
    walk_forward_passed: bool = True,
    min_trades: int = MIN_RESEARCH_TRADES,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not research_result_valid:
        reasons.append("research_data_source_invalid")
    if source_used is not None and source_used != "collected_market_data":
        reasons.append("data_source_not_collected_market_data")
    if _metric_float(metrics.get("net_return_pct")) <= 0:
        reasons.append("net_return_not_positive")
    if _profit_factor_value(metrics.get("profit_factor_net")) < MIN_RESEARCH_PROFIT_FACTOR_NET:
        reasons.append("profit_factor_net_below_1_05")
    if int(metrics.get("number_of_trades", 0) or 0) < int(min_trades):
        reasons.append(f"number_of_trades_below_{int(min_trades)}")
    if _metric_float(metrics.get("max_drawdown_pct")) > float(settings.max_backtest_drawdown_pct):
        reasons.append("max_drawdown_above_configured_limit")
    if single_trade_return_concentration(metrics.get("trade_details", [])) > MAX_SINGLE_TRADE_RETURN_SHARE:
        reasons.append("single_trade_return_concentration_too_high")
    if take_profit_pct is not None and round_trip_estimated_cost_pct is not None:
        if float(take_profit_pct) <= float(round_trip_estimated_cost_pct):
            reasons.append("take_profit_not_above_round_trip_cost")
    if take_profit_pct is not None and promotion_required_return_pct is not None:
        if float(take_profit_pct) < float(promotion_required_return_pct):
            reasons.append("take_profit_below_promotion_required_return")
    if not walk_forward_passed:
        reasons.append("walk_forward_not_passed")
    if fallback_prediction_used:
        reasons.append("fallback_prediction_not_allowed")
    economically_viable = not reasons
    if not active_model_valid:
        reasons.append("active_model_invalid")
    return {
        "economically_viable": economically_viable,
        "paper_forward_eligible": not reasons,
        "rejection_reasons": reasons,
    }


def build_research_summary(
    rows: list[dict[str, Any]],
    settings: Settings,
    *,
    data_sources: dict[str, str] | None = None,
    data_source_reports: dict[str, ResearchDataReport | dict[str, Any]] | None = None,
    csv_path: Path,
    summary_path: Path,
    active_model_status: dict[str, Any],
    requested_max_rows_by_timeframe: dict[str, int] | None = None,
    requested_start: str | None = None,
    requested_end: str | None = None,
    strategy_filter: str = "all",
    max_buy_dip_configs: int = DEFAULT_MAX_BUY_DIP_CONFIGS,
    max_v3_configs: int = DEFAULT_MAX_V3_CONFIGS,
    walk_forward_splits: int = DEFAULT_WALK_FORWARD_SPLITS,
    min_trades: int = MIN_RESEARCH_TRADES,
    min_trades_per_split: int = MIN_RESEARCH_TRADES_PER_SPLIT,
    requested_timeframes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    ranked = sorted(rows, key=_rank_sort_key, reverse=True)
    raw_economically_viable = [row for row in ranked if row.get("economically_viable")]
    source_reports = _summary_source_reports(data_source_reports=data_source_reports, data_sources=data_sources)
    source_report_payload = {timeframe: _data_report_to_dict(report) for timeframe, report in source_reports.items()}
    data_source_names = {timeframe: report.source_used for timeframe, report in source_reports.items()}
    synthetic_data_used = any(report.synthetic_data_used for report in source_reports.values()) or any(
        bool(row.get("synthetic_data_used")) for row in ranked
    )
    research_result_valid = (
        bool(source_reports)
        and all(report.research_result_valid for report in source_reports.values())
        and not synthetic_data_used
    )
    economically_viable = [] if synthetic_data_used or not research_result_valid else raw_economically_viable
    eligible = [] if synthetic_data_used or not research_result_valid else [
        row for row in ranked if row.get("paper_forward_eligible")
    ]
    strategy_breakdown = build_strategy_breakdown(ranked, economically_viable=economically_viable, eligible=eligible)
    best_by_strategy = best_configs_by_strategy(ranked)
    buy_the_dip_rows = [row for row in ranked if row.get("strategy_name") == BUY_THE_DIP_STRATEGY]
    buy_the_dip_economic = [row for row in economically_viable if row.get("strategy_name") == BUY_THE_DIP_STRATEGY]
    buy_the_dip_eligible = [row for row in eligible if row.get("strategy_name") == BUY_THE_DIP_STRATEGY]
    buy_the_dip_trade_summary = buy_the_dip_trade_count_summary(
        buy_the_dip_rows,
        economically_viable=buy_the_dip_economic,
        eligible=buy_the_dip_eligible,
    )
    if BUY_THE_DIP_STRATEGY in strategy_breakdown:
        strategy_breakdown[BUY_THE_DIP_STRATEGY].update(buy_the_dip_trade_summary)
    concise_summary = build_concise_research_summary(
        ranked,
        research_result_valid=research_result_valid,
        economically_viable=economically_viable,
        eligible=eligible,
        buy_the_dip_rows=buy_the_dip_rows,
        buy_the_dip_economic=buy_the_dip_economic,
        buy_the_dip_eligible=buy_the_dip_eligible,
        buy_the_dip_trade_summary=buy_the_dip_trade_summary,
        target_vs_cost_unsafe=(
            float(settings.scalping_take_profit_pct) <= estimated_round_trip_execution_cost_pct(settings)
            or float(settings.scalping_label_take_profit_pct) <= estimated_round_trip_execution_cost_pct(settings)
        ),
    )
    notes = [
        "Higher-timeframe research is offline analysis only and never enables trading.",
        "No configuration is auto-applied or promoted from this report.",
        "Paper-forward eligibility also requires the existing active model registry to validate.",
        "Research validity requires fresh non-synthetic data for every timeframe.",
    ]
    if synthetic_data_used:
        notes.append(
            "Synthetic bars were used only because explicit test/demo mode allowed it; "
            "this report is invalid for trading decisions."
        )
    return _json_safe(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "symbol": settings.symbol,
            "paper_trading_only": settings.paper_trading_only,
            "btc_usd_only": settings.symbol == ALLOWED_SYMBOL,
            "long_only": True,
            "trading_enabled": settings.trading_enabled,
            "auto_trade_enabled": settings.auto_trade_enabled,
            "fallback_trading_allowed": settings.allow_fallback_trading,
            "auto_apply_best_config": False,
            "paper_forward_eligible_config_count": len(eligible),
            "economically_viable_config_count": len(economically_viable),
            "source_used": data_source_names,
            "latest_timestamp": {
                timeframe: report.latest_timestamp for timeframe, report in source_reports.items()
            },
            "data_age_minutes": {
                timeframe: report.data_age_minutes for timeframe, report in source_reports.items()
            },
            "row_count": {
                timeframe: report.row_count for timeframe, report in source_reports.items()
            },
            "available_rows_by_timeframe": {
                timeframe: report.available_rows if report.available_rows is not None else report.row_count
                for timeframe, report in source_reports.items()
            },
            "used_rows_by_timeframe": {
                timeframe: report.used_rows if report.used_rows is not None else report.row_count
                for timeframe, report in source_reports.items()
            },
            "first_timestamp_by_timeframe": {
                timeframe: report.first_timestamp for timeframe, report in source_reports.items()
            },
            "latest_timestamp_by_timeframe": {
                timeframe: report.latest_timestamp for timeframe, report in source_reports.items()
            },
            "requested_max_rows_by_timeframe": {
                timeframe: int((requested_max_rows_by_timeframe or {}).get(timeframe, report.requested_max_rows or report.row_count))
                for timeframe, report in source_reports.items()
            },
            "actual_used_rows_by_timeframe": {
                timeframe: report.used_rows if report.used_rows is not None else report.row_count
                for timeframe, report in source_reports.items()
            },
            "requested_start": requested_start,
            "requested_end": requested_end,
            "strategy_filter": strategy_filter,
            "timeframes": list(requested_timeframes or source_reports.keys()),
            "max_buy_dip_configs": max_buy_dip_configs,
            "max_v3_configs": max_v3_configs,
            "walk_forward_splits": walk_forward_splits,
            "min_trades": min_trades,
            "min_trades_per_split": min_trades_per_split,
            "synthetic_data_used": synthetic_data_used,
            "research_result_valid": research_result_valid,
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "data_sources": data_source_names,
            "timeframe_data": source_report_payload,
            "strategy_breakdown": strategy_breakdown,
            "best_configs_by_strategy": best_by_strategy,
            "buy_the_dip_mean_reversion_best_configs": best_by_strategy.get(BUY_THE_DIP_STRATEGY, []),
            "buy_the_dip_mean_reversion_trade_summary": buy_the_dip_trade_summary,
            "rejection_reason_counts": rejection_reason_counts(ranked),
            "take_profit_vs_cost_safe_config_count": sum(
                1 for row in ranked if bool(row.get("take_profit_vs_cost_safe"))
            ),
            "economically_viable_by_strategy": {
                strategy: int(values.get("economically_viable_count", 0))
                for strategy, values in strategy_breakdown.items()
            },
            "paper_forward_eligible_by_strategy": {
                strategy: int(values.get("paper_forward_eligible_count", 0))
                for strategy, values in strategy_breakdown.items()
            },
            "concise_summary": concise_summary,
            **concise_summary,
            "parameter_space": {
                "timeframes": list(RESEARCH_TIMEFRAMES),
                "take_profit_pct": list(TAKE_PROFIT_VALUES),
                "stop_loss_pct": list(STOP_LOSS_VALUES),
                "max_hold_bars": list(MAX_HOLD_BARS_VALUES),
                "buy_the_dip_mean_reversion": {
                    "timeframes": list(RESEARCH_TIMEFRAMES),
                    "take_profit_pct": list(BUY_THE_DIP_TAKE_PROFIT_VALUES),
                    "stop_loss_pct": list(BUY_THE_DIP_STOP_LOSS_VALUES),
                    "max_hold_bars": list(BUY_THE_DIP_MAX_HOLD_BARS_VALUES),
                    "signal_profiles": list(BUY_THE_DIP_SIGNAL_PROFILES),
                    "v2_profile_values": BUY_THE_DIP_V2_PROFILE_VALUES,
                    "max_buy_dip_configs": max_buy_dip_configs,
                    "grid_note": "Bounded deterministic profiles use wider v2 thresholds without an unbounded Cartesian explosion.",
                },
                UPTREND_PULLBACK_STRATEGY: {
                    "timeframes": list(V3_RESEARCH_TIMEFRAMES),
                    "take_profit_pct": list(UPTREND_PULLBACK_TAKE_PROFIT_VALUES),
                    "stop_loss_pct": list(UPTREND_PULLBACK_STOP_LOSS_VALUES),
                    "max_hold_bars": list(V3_MAX_HOLD_BARS_VALUES),
                },
                VOLATILITY_BREAKOUT_STRATEGY: {
                    "timeframes": list(V3_RESEARCH_TIMEFRAMES),
                    "take_profit_pct": list(VOLATILITY_BREAKOUT_TAKE_PROFIT_VALUES),
                    "stop_loss_pct": list(VOLATILITY_BREAKOUT_STOP_LOSS_VALUES),
                    "max_hold_bars": list(V3_MAX_HOLD_BARS_VALUES),
                },
            },
            "active_model_status": active_model_status,
            "conservative_backtest_assumptions": backtest_assumptions(settings, spread_available=True),
            "best_economically_viable_configs": economically_viable[:10],
            "paper_forward_eligible_configs": eligible[:10],
            "rejected_configs": [row for row in ranked if not row.get("paper_forward_eligible")][:50],
            "all_results": ranked,
            "notes": notes,
        }
    )


def build_strategy_breakdown(
    rows: list[dict[str, Any]],
    *,
    economically_viable: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    economic_ids = {row.get("parameter_set_id") for row in economically_viable}
    eligible_ids = {row.get("parameter_set_id") for row in eligible}
    strategies = sorted({str(row.get("strategy_name") or "unknown") for row in rows})
    breakdown: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        strategy_rows = [row for row in rows if str(row.get("strategy_name") or "unknown") == strategy]
        best = best_ranked_config(strategy_rows)
        trade_summary = strategy_trade_count_summary(
            strategy_rows,
            economically_viable=[row for row in economically_viable if row.get("strategy_name") == strategy],
            eligible=[row for row in eligible if row.get("strategy_name") == strategy],
        )
        breakdown[strategy] = {
            "configs_tested": len(strategy_rows),
            "take_profit_vs_cost_safe_count": sum(
                1 for row in strategy_rows if bool(row.get("take_profit_vs_cost_safe"))
            ),
            "economically_viable_count": sum(1 for row in strategy_rows if row.get("parameter_set_id") in economic_ids),
            "paper_forward_eligible_count": sum(1 for row in strategy_rows if row.get("parameter_set_id") in eligible_ids),
            "best_net_return_pct": best.get("net_return_pct") if best else None,
            "best_profit_factor_net": best.get("profit_factor_net") if best else None,
            "best_max_drawdown_pct": best.get("max_drawdown_pct") if best else None,
            "rejection_reason_counts": rejection_reason_counts(strategy_rows),
            **trade_summary,
        }
    return breakdown


def best_configs_by_strategy(rows: list[dict[str, Any]], *, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for strategy in sorted({str(row.get("strategy_name") or "unknown") for row in rows}):
        strategy_rows = [row for row in rows if str(row.get("strategy_name") or "unknown") == strategy]
        out[strategy] = sorted(strategy_rows, key=_rank_sort_key, reverse=True)[:limit]
    return out


def best_ranked_config(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=_rank_sort_key)


def _rank_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row.get("economically_viable")),
        not bool(row.get("statistically_weak")),
        _metric_float(row.get("adjusted_rank_score", row.get("rank_score"))),
        int(row.get("number_of_trades", 0) or 0),
        _metric_float(row.get("net_return_pct")),
        min(5.0, _profit_factor_value(row.get("profit_factor_net"))),
    )


def rejection_reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        raw = str(row.get("rejection_reasons") or "")
        for reason in [part for part in raw.split(";") if part]:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def buy_the_dip_trade_count_summary(
    rows: list[dict[str, Any]],
    *,
    economically_viable: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
) -> dict[str, Any]:
    return strategy_trade_count_summary(rows, economically_viable=economically_viable, eligible=eligible)


def strategy_trade_count_summary(
    rows: list[dict[str, Any]],
    *,
    economically_viable: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_1_to_4 = [row for row in rows if 1 <= int(row.get("number_of_trades", 0) or 0) <= 4]
    rows_5_to_19 = [row for row in rows if 5 <= int(row.get("number_of_trades", 0) or 0) <= 19]
    rows_20_plus = [row for row in rows if int(row.get("number_of_trades", 0) or 0) >= MIN_RESEARCH_TRADES]
    rows_50_plus = [row for row in rows if int(row.get("number_of_trades", 0) or 0) >= PREFERRED_RESEARCH_TRADES]
    profitable = [row for row in rows if _metric_float(row.get("net_return_pct")) > 0]
    profitable_20_plus = [row for row in rows_20_plus if _metric_float(row.get("net_return_pct")) > 0]
    promising = [row for row in rows if row.get("research_promising")]
    walk_forward_rows = [row for row in rows if row.get("walk_forward_passed")]
    return {
        "configs_tested": len(rows),
        "configs_with_0_trades": sum(1 for row in rows if int(row.get("number_of_trades", 0) or 0) == 0),
        "configs_with_1_to_4_trades": len(rows_1_to_4),
        "configs_with_5_to_19_trades": len(rows_5_to_19),
        "configs_with_20_plus_trades": len(rows_20_plus),
        "configs_with_50_plus_trades": len(rows_50_plus),
        "profitable_configs": len(profitable),
        "profitable_configs_with_20_plus_trades": len(profitable_20_plus),
        "profitable_20_plus_trade_configs": len(profitable_20_plus),
        "economically_viable_count": len(economically_viable),
        "research_promising_count": len(promising),
        "paper_forward_eligible_count": len(eligible),
        "best_config_any_trade_count": best_ranked_config(rows),
        "best_config_20_plus_trades": best_ranked_config(rows_20_plus),
        "best_config_50_plus_trades": best_ranked_config(rows_50_plus),
        "best_walk_forward_config": best_ranked_config(walk_forward_rows),
        "rejection_reason_counts": rejection_reason_counts(rows),
    }


def build_concise_research_summary(
    rows: list[dict[str, Any]],
    *,
    research_result_valid: bool,
    economically_viable: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    buy_the_dip_rows: list[dict[str, Any]],
    buy_the_dip_economic: list[dict[str, Any]],
    buy_the_dip_eligible: list[dict[str, Any]],
    buy_the_dip_trade_summary: dict[str, Any],
    target_vs_cost_unsafe: bool,
) -> dict[str, Any]:
    best = buy_the_dip_trade_summary.get("best_config_any_trade_count") or best_ranked_config(buy_the_dip_rows)
    best_20_plus = buy_the_dip_trade_summary.get("best_config_20_plus_trades")
    best_50_plus = buy_the_dip_trade_summary.get("best_config_50_plus_trades")
    buy_the_dip_rejected = bool(
        buy_the_dip_rows
        and int(buy_the_dip_trade_summary.get("configs_with_20_plus_trades", 0) or 0) > 0
        and not buy_the_dip_economic
    )
    recommendation = "improve_strategy"
    if not research_result_valid:
        recommendation = "collect_more_data"
    elif eligible:
        recommendation = "run_paper_forward_research_only"
    elif economically_viable:
        recommendation = "candidate_found_keep_trading_disabled"
    elif buy_the_dip_rejected and not any(row.get("strategy_name") in V3_STRATEGIES for row in rows):
        recommendation = "rejected_strategy"
    return {
        "data_ready": research_result_valid,
        "target_vs_cost_unsafe": target_vs_cost_unsafe,
        "buy_the_dip_rejected": buy_the_dip_rejected,
        "buy_the_dip_v2_rejected_historical": True,
        "buy_the_dip_configs_tested": len(buy_the_dip_rows),
        "buy_the_dip_20_plus_trade_configs": int(
            buy_the_dip_trade_summary.get("configs_with_20_plus_trades", 0) or 0
        ),
        "buy_the_dip_50_plus_trade_configs": int(
            buy_the_dip_trade_summary.get("configs_with_50_plus_trades", 0) or 0
        ),
        "buy_the_dip_profitable_configs": int(buy_the_dip_trade_summary.get("profitable_configs", 0) or 0),
        "buy_the_dip_profitable_20_plus_trade_configs": int(
            buy_the_dip_trade_summary.get("profitable_configs_with_20_plus_trades", 0) or 0
        ),
        "buy_the_dip_economically_viable_count": len(buy_the_dip_economic),
        "buy_the_dip_paper_forward_eligible_count": len(buy_the_dip_eligible),
        "buy_the_dip_best_net_return_pct": best.get("net_return_pct") if best else None,
        "buy_the_dip_best_profit_factor_net": best.get("profit_factor_net") if best else None,
        "buy_the_dip_best_max_drawdown_pct": best.get("max_drawdown_pct") if best else None,
        "buy_the_dip_best_config_20_plus_trades": best_20_plus,
        "buy_the_dip_best_config_50_plus_trades": best_50_plus,
        "uptrend_pullback_promising_count": sum(
            1 for row in rows if row.get("strategy_name") == UPTREND_PULLBACK_STRATEGY and row.get("research_promising")
        ),
        "volatility_breakout_promising_count": sum(
            1 for row in rows if row.get("strategy_name") == VOLATILITY_BREAKOUT_STRATEGY and row.get("research_promising")
        ),
        "recommendation": recommendation,
    }


def write_research_outputs(
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


async def _fetch_research_bars(
    client: Any,
    settings: Settings,
    *,
    timeframe: str,
    limit: int,
    session_factory: Any | None = None,
    allow_synthetic_fallback: bool = False,
    now: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> ResearchBarsResult:
    current_time = _utc_timestamp(now or datetime.now(UTC))
    effective_end = min(_utc_timestamp(end or current_time), current_time)
    min_rows = _minimum_research_rows(settings, limit)
    rejected_sources: list[dict[str, Any]] = []

    sqlite_bars = _load_collected_market_data(
        settings,
        timeframe=timeframe,
        limit=limit,
        session_factory=session_factory,
        start=start,
        end=effective_end,
    )
    sqlite_stats = _collected_market_data_window_stats(
        settings,
        timeframe=timeframe,
        session_factory=session_factory,
        start=start,
        end=effective_end,
    )
    sqlite_report = _assess_research_bars(
        sqlite_bars,
        source_used="collected_market_data",
        timeframe=timeframe,
        min_rows=min_rows,
        now=current_time,
        synthetic_data_used=False,
        available_rows=int(sqlite_stats["available_rows"]),
        requested_max_rows=limit,
        requested_start=start.isoformat() if start else None,
        requested_end=effective_end.isoformat(),
    )
    if sqlite_report.research_result_valid:
        return ResearchBarsResult(sqlite_bars, sqlite_report)
    rejected_sources.append(_data_report_to_rejected_source(sqlite_report))

    if isinstance(client, MarketDataClient) and not (settings.alpaca_api_key and settings.alpaca_secret_key):
        rejected_sources.append(
            {
                "source": "market_data_client",
                "status": "unavailable",
                "reason": "alpaca_credentials_missing",
                "latest_timestamp": None,
                "data_age_minutes": None,
                "row_count": 0,
            }
        )
    else:
        try:
            client_bars = await client.fetch_bars(
                settings.symbol,
                timeframe=_market_data_timeframe(timeframe),
                limit=limit,
                force_refresh=True,
            )
            client_bars = _filter_bars_to_window(client_bars, start=start, end=effective_end)
            client_report = _assess_research_bars(
                client_bars,
                source_used="market_data_client",
                timeframe=timeframe,
                min_rows=min_rows,
                now=current_time,
                synthetic_data_used=False,
                requested_max_rows=limit,
                requested_start=start.isoformat() if start else None,
                requested_end=effective_end.isoformat(),
            )
            if client_report.research_result_valid:
                return ResearchBarsResult(client_bars, client_report)
            rejected_sources.append(_data_report_to_rejected_source(client_report))
        except Exception as exc:
            latest_timestamp = _latest_timestamp_from_error(exc)
            rejected_sources.append(
                {
                    "source": "market_data_client",
                    "status": "stale" if isinstance(exc, StaleMarketDataError) else "fetch_error",
                    "reason": "stale_latest_timestamp" if isinstance(exc, StaleMarketDataError) else type(exc).__name__,
                    "latest_timestamp": latest_timestamp,
                    "data_age_minutes": _data_age_minutes_from_iso(latest_timestamp, current_time),
                    "row_count": 0,
                }
            )

    if allow_synthetic_fallback:
        synthetic_bars = MarketDataClient.synthetic_btc_bars(limit=limit, timeframe=_market_data_timeframe(timeframe))
        synthetic_bars = _filter_bars_to_window(synthetic_bars, start=start, end=effective_end)
        synthetic_report = _assess_research_bars(
            synthetic_bars,
            source_used="synthetic_explicit_test_demo_mode",
            timeframe=timeframe,
            min_rows=min_rows,
            now=current_time,
            synthetic_data_used=True,
            force_invalid_reason="synthetic_data_not_valid_for_research_decisions",
            requested_max_rows=limit,
            requested_start=start.isoformat() if start else None,
            requested_end=effective_end.isoformat(),
        )
        return ResearchBarsResult(
            synthetic_bars,
            _with_rejected_sources(synthetic_report, rejected_sources),
        )

    report = ResearchDataReport(
        timeframe=timeframe,
        source_used="no_valid_real_data_source",
        latest_timestamp=None,
        data_age_minutes=None,
        row_count=0,
        synthetic_data_used=False,
        research_result_valid=False,
        rejection_reason="no_fresh_real_bars_available",
        rejected_sources=tuple(rejected_sources),
        available_rows=int(sqlite_stats["available_rows"]),
        used_rows=0,
        first_timestamp=None,
        requested_max_rows=limit,
        requested_start=start.isoformat() if start else None,
        requested_end=effective_end.isoformat(),
    )
    return ResearchBarsResult(pd.DataFrame(), report)


async def _fetch_or_derive_research_bars(
    client: Any,
    settings: Settings,
    *,
    timeframe: str,
    limit: int,
    existing_bars_by_timeframe: dict[str, pd.DataFrame],
    existing_reports_by_timeframe: dict[str, ResearchDataReport],
    row_limits: dict[str, int],
    session_factory: Any | None = None,
    allow_synthetic_fallback: bool = False,
    now: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> ResearchBarsResult:
    result = await _fetch_research_bars(
        client,
        settings,
        timeframe=timeframe,
        limit=limit,
        session_factory=session_factory,
        allow_synthetic_fallback=allow_synthetic_fallback,
        now=now,
        start=start,
        end=end,
    )
    if timeframe != "1H" or result.report.research_result_valid:
        return result

    source_15min = existing_bars_by_timeframe.get("15Min")
    source_report = existing_reports_by_timeframe.get("15Min")
    if source_15min is None or source_report is None:
        source_limit = max(int(row_limits.get("15Min", limit * 4)), limit * 4)
        source_result = await _fetch_research_bars(
            client,
            settings,
            timeframe="15Min",
            limit=source_limit,
            session_factory=session_factory,
            allow_synthetic_fallback=allow_synthetic_fallback,
            now=now,
            start=start,
            end=end,
        )
        source_15min = source_result.bars
        source_report = source_result.report
        existing_bars_by_timeframe["15Min"] = source_15min
        existing_reports_by_timeframe["15Min"] = source_report

    derived = derive_1h_bars_from_15min(source_15min, limit=limit)
    current_time = _utc_timestamp(now or datetime.now(UTC))
    min_rows = _minimum_research_rows(settings, limit)
    derived_report = _assess_research_bars(
        derived,
        source_used="collected_market_data",
        timeframe="1H",
        min_rows=min_rows,
        now=current_time,
        synthetic_data_used=bool(source_report.synthetic_data_used),
        force_invalid_reason=(
            None
            if source_report.research_result_valid and not source_report.synthetic_data_used
            else source_report.rejection_reason or "source_15min_invalid_for_1h_derivation"
        ),
        available_rows=int(len(derived)),
        requested_max_rows=limit,
        requested_start=start.isoformat() if start else None,
        requested_end=_utc_timestamp(end or current_time).isoformat(),
        derived_from_timeframe="15Min",
    )
    return ResearchBarsResult(
        derived,
        _with_rejected_sources(
            derived_report,
            [
                *result.report.rejected_sources,
                _data_report_to_rejected_source(result.report),
                {
                    "source": "collected_market_data",
                    "timeframe": "15Min",
                    "status": "used_for_1h_derivation" if source_report.research_result_valid else "rejected",
                    "reason": source_report.rejection_reason,
                    "latest_timestamp": source_report.latest_timestamp,
                    "data_age_minutes": source_report.data_age_minutes,
                    "row_count": source_report.row_count,
                },
            ],
        ),
    )


def derive_1h_bars_from_15min(bars: pd.DataFrame, *, limit: int | None = None) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    data = normalize_ohlcv(bars)
    if data.empty:
        return data
    grouped = data.assign(hour=data["timestamp"].dt.floor("1h")).groupby("hour", sort=True)
    rows: list[dict[str, Any]] = []
    for hour, group in grouped:
        ordered = group.sort_values("timestamp")
        if len(ordered) < 4:
            continue
        rows.append(
            {
                "timestamp": hour,
                "open": float(ordered["open"].iloc[0]),
                "high": float(ordered["high"].max()),
                "low": float(ordered["low"].min()),
                "close": float(ordered["close"].iloc[-1]),
                "volume": float(ordered["volume"].sum()),
            }
        )
    derived = normalize_ohlcv(pd.DataFrame(rows)) if rows else pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    if limit is not None and len(derived) > int(limit):
        derived = derived.tail(int(limit)).reset_index(drop=True)
    return derived


def _load_collected_market_data(
    settings: Settings,
    *,
    timeframe: str,
    limit: int,
    session_factory: Any | None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    if session_factory is None:
        init_db()
        session_factory = SessionLocal
    with session_factory() as db:
        query = db.query(CollectedMarketData).filter(
            CollectedMarketData.symbol == settings.symbol,
            CollectedMarketData.timeframe == timeframe,
        )
        if start is not None:
            query = query.filter(CollectedMarketData.timestamp >= _sqlite_filter_timestamp(start))
        if end is not None:
            query = query.filter(CollectedMarketData.timestamp <= _sqlite_filter_timestamp(end))
        rows = query.order_by(CollectedMarketData.timestamp.desc()).limit(limit).all()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    records = [
        {
            "timestamp": row.timestamp,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in reversed(rows)
    ]
    return normalize_ohlcv(pd.DataFrame(records))


def _filter_bars_to_window(
    bars: pd.DataFrame,
    *,
    start: datetime | None,
    end: datetime | None,
) -> pd.DataFrame:
    if bars.empty or "timestamp" not in bars:
        return bars
    normalized = normalize_ohlcv(bars)
    timestamps = pd.to_datetime(normalized["timestamp"], utc=True)
    mask = pd.Series(True, index=normalized.index)
    if start is not None:
        mask &= timestamps >= pd.Timestamp(start)
    if end is not None:
        mask &= timestamps <= pd.Timestamp(end)
    return normalized.loc[mask].reset_index(drop=True)


def _collected_market_data_window_stats(
    settings: Settings,
    *,
    timeframe: str,
    session_factory: Any | None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    if session_factory is None:
        init_db()
        session_factory = SessionLocal
    from sqlalchemy import func

    with session_factory() as db:
        query = db.query(
            func.count(CollectedMarketData.id),
            func.min(CollectedMarketData.timestamp),
            func.max(CollectedMarketData.timestamp),
        ).filter(
            CollectedMarketData.symbol == settings.symbol,
            CollectedMarketData.timeframe == timeframe,
        )
        if start is not None:
            query = query.filter(CollectedMarketData.timestamp >= _sqlite_filter_timestamp(start))
        if end is not None:
            query = query.filter(CollectedMarketData.timestamp <= _sqlite_filter_timestamp(end))
        row = query.one()
    return {
        "available_rows": int(row[0] or 0),
        "first_timestamp": row[1],
        "latest_timestamp": row[2],
    }


def _assess_research_bars(
    bars: pd.DataFrame,
    *,
    source_used: str,
    timeframe: str,
    min_rows: int,
    now: datetime,
    synthetic_data_used: bool,
    force_invalid_reason: str | None = None,
    available_rows: int | None = None,
    requested_max_rows: int | None = None,
    requested_start: str | None = None,
    requested_end: str | None = None,
    derived_from_timeframe: str | None = None,
) -> ResearchDataReport:
    normalized = normalize_ohlcv(bars) if not bars.empty else bars
    first = _first_timestamp(normalized)
    latest = _latest_timestamp(normalized)
    age_minutes = _data_age_minutes(latest, now)
    rejection_reasons: list[str] = []
    if len(normalized) < min_rows:
        rejection_reasons.append(f"row_count_below_required_{min_rows}")
    if latest is None:
        rejection_reasons.append("latest_timestamp_missing")
    elif latest > pd.Timestamp(_utc_timestamp(now)):
        rejection_reasons.append("future_timestamp_detected")
    elif age_minutes is not None:
        max_age_minutes = stale_threshold_for_timeframe(_market_data_timeframe(timeframe)).total_seconds() / 60
        if age_minutes > max_age_minutes:
            rejection_reasons.append(f"stale_latest_timestamp_age_minutes_gt_{max_age_minutes:g}")
    if synthetic_data_used:
        rejection_reasons.append(force_invalid_reason or "synthetic_data_not_valid_for_research_decisions")
    elif force_invalid_reason:
        rejection_reasons.append(force_invalid_reason)
    return ResearchDataReport(
        timeframe=timeframe,
        source_used=source_used,
        latest_timestamp=latest.isoformat() if latest is not None else None,
        data_age_minutes=age_minutes,
        row_count=int(len(normalized)),
        synthetic_data_used=synthetic_data_used,
        research_result_valid=not rejection_reasons,
        rejection_reason=";".join(rejection_reasons) if rejection_reasons else None,
        available_rows=available_rows if available_rows is not None else int(len(normalized)),
        used_rows=int(len(normalized)),
        first_timestamp=first.isoformat() if first is not None else None,
        requested_max_rows=requested_max_rows,
        requested_start=requested_start,
        requested_end=requested_end,
        derived_from_timeframe=derived_from_timeframe,
    )


def _minimum_research_rows(settings: Settings, limit: int) -> int:
    return max(1, min(int(limit), max(MIN_RESEARCH_TRADES * 10, int(settings.min_training_rows))))


def _latest_timestamp(bars: pd.DataFrame) -> pd.Timestamp | None:
    if bars.empty or "timestamp" not in bars:
        return None
    latest = pd.Timestamp(bars["timestamp"].iloc[-1])
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    else:
        latest = latest.tz_convert("UTC")
    return latest


def _first_timestamp(bars: pd.DataFrame) -> pd.Timestamp | None:
    if bars.empty or "timestamp" not in bars:
        return None
    first = pd.Timestamp(bars["timestamp"].iloc[0])
    if first.tzinfo is None:
        first = first.tz_localize("UTC")
    else:
        first = first.tz_convert("UTC")
    return first


def _data_age_minutes(latest: pd.Timestamp | None, now: datetime) -> float | None:
    if latest is None:
        return None
    current_time = _utc_timestamp(now)
    return max(0.0, (pd.Timestamp(current_time) - latest).total_seconds() / 60)


def _utc_timestamp(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _sqlite_filter_timestamp(value: Any) -> datetime:
    return _utc_timestamp(value).replace(tzinfo=None)


def _data_report_to_rejected_source(report: ResearchDataReport) -> dict[str, Any]:
    return {
        "source": report.source_used,
        "timeframe": report.timeframe,
        "status": "stale" if _report_is_stale(report) else "rejected",
        "reason": report.rejection_reason,
        "latest_timestamp": report.latest_timestamp,
        "data_age_minutes": report.data_age_minutes,
        "row_count": report.row_count,
        "available_rows": report.available_rows,
        "used_rows": report.used_rows,
        "first_timestamp": report.first_timestamp,
        "requested_max_rows": report.requested_max_rows,
        "derived_from_timeframe": report.derived_from_timeframe,
    }


def _report_is_stale(report: ResearchDataReport) -> bool:
    return "stale_latest_timestamp" in (report.rejection_reason or "")


def _with_rejected_sources(
    report: ResearchDataReport,
    rejected_sources: list[dict[str, Any]],
) -> ResearchDataReport:
    return ResearchDataReport(
        timeframe=report.timeframe,
        source_used=report.source_used,
        latest_timestamp=report.latest_timestamp,
        data_age_minutes=report.data_age_minutes,
        row_count=report.row_count,
        synthetic_data_used=report.synthetic_data_used,
        research_result_valid=report.research_result_valid,
        rejection_reason=report.rejection_reason,
        rejected_sources=tuple(rejected_sources),
        available_rows=report.available_rows,
        used_rows=report.used_rows,
        first_timestamp=report.first_timestamp,
        requested_max_rows=report.requested_max_rows,
        requested_start=report.requested_start,
        requested_end=report.requested_end,
        derived_from_timeframe=report.derived_from_timeframe,
    )


def _coerce_data_report(timeframe: str, value: ResearchDataReport | dict[str, Any]) -> ResearchDataReport:
    if isinstance(value, ResearchDataReport):
        return value
    source_used = str(value.get("source_used", value.get("source", "unknown")))
    synthetic_data_used = bool(value.get("synthetic_data_used", source_used.startswith("synthetic")))
    return ResearchDataReport(
        timeframe=timeframe,
        source_used=source_used,
        latest_timestamp=value.get("latest_timestamp"),
        data_age_minutes=value.get("data_age_minutes"),
        row_count=int(value.get("row_count", 0) or 0),
        synthetic_data_used=synthetic_data_used,
        research_result_valid=bool(value.get("research_result_valid", not synthetic_data_used)),
        rejection_reason=value.get("rejection_reason"),
        rejected_sources=tuple(value.get("rejected_sources", ())),
        available_rows=value.get("available_rows"),
        used_rows=value.get("used_rows", value.get("row_count")),
        first_timestamp=value.get("first_timestamp"),
        requested_max_rows=value.get("requested_max_rows"),
        requested_start=value.get("requested_start"),
        requested_end=value.get("requested_end"),
        derived_from_timeframe=value.get("derived_from_timeframe"),
    )


def _summary_source_reports(
    *,
    data_source_reports: dict[str, ResearchDataReport | dict[str, Any]] | None,
    data_sources: dict[str, str] | None,
) -> dict[str, ResearchDataReport]:
    if data_source_reports:
        return {
            timeframe: _coerce_data_report(timeframe, report)
            for timeframe, report in data_source_reports.items()
        }
    return {
        timeframe: ResearchDataReport(
            timeframe=timeframe,
            source_used=source,
            latest_timestamp=None,
            data_age_minutes=None,
            row_count=0,
            synthetic_data_used=source.startswith("synthetic"),
            research_result_valid=not source.startswith("synthetic"),
            available_rows=0,
            used_rows=0,
        )
        for timeframe, source in (data_sources or {}).items()
    }


def _data_report_to_dict(report: ResearchDataReport) -> dict[str, Any]:
    return {
        "source_used": report.source_used,
        "latest_timestamp": report.latest_timestamp,
        "data_age_minutes": report.data_age_minutes,
        "row_count": report.row_count,
        "synthetic_data_used": report.synthetic_data_used,
        "research_result_valid": report.research_result_valid,
        "rejection_reason": report.rejection_reason,
        "rejected_sources": list(report.rejected_sources),
        "available_rows": report.available_rows,
        "used_rows": report.used_rows,
        "first_timestamp": report.first_timestamp,
        "requested_max_rows": report.requested_max_rows,
        "requested_start": report.requested_start,
        "requested_end": report.requested_end,
        "derived_from_timeframe": report.derived_from_timeframe,
    }


def _latest_timestamp_from_error(exc: Exception) -> str | None:
    marker = "latest_timestamp="
    message = str(exc)
    if marker not in message:
        return None
    value = message.split(marker, 1)[1].split(" ", 1)[0]
    return value or None


def _data_age_minutes_from_iso(value: str | None, now: datetime) -> float | None:
    if value is None:
        return None
    return _data_age_minutes(pd.Timestamp(value), now)


def _synthetic_research_mode_enabled() -> bool:
    return _env_flag("RESEARCH_ALLOW_SYNTHETIC_FALLBACK") or _env_flag("RESEARCH_DEMO_MODE")


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _signal_row(
    row: pd.Series,
    *,
    signal: Any,
    regime: str,
    config: ResearchConfig,
) -> dict[str, Any]:
    return {
        "timestamp": row["timestamp"],
        "close": float(row["close"]),
        "orderbook_spread": float(row.get("orderbook_spread", 0.0)),
        "quote_imbalance": float(row.get("quote_imbalance", 0.0)),
        "scalping_spread_bps": float(row.get("scalping_spread_bps", 0.0)),
        "scalping_quote_imbalance": float(row.get("scalping_quote_imbalance", 0.0)),
        "strategy_name": signal.strategy_name,
        "entry_reason": signal.reason,
        "strategy_score": float(signal.score),
        "strategy_confidence": float(signal.confidence),
        "quant_score": float(signal.score),
        "quant_confidence": float(signal.confidence),
        "regime": regime,
        "blocked_by": None if signal.action == "buy" else _research_block_bucket(signal.reason),
        "block_reason": None if signal.action == "buy" else signal.reason,
        "ml_buy_probability": 1.0 if signal.action == "buy" else 0.0,
        "ml_sell_probability": 0.0 if signal.action == "buy" else 1.0,
        "_probability": 1.0 if signal.action == "buy" else 0.0,
        "research_timeframe": config.timeframe,
        "research_take_profit_pct": config.take_profit_pct,
        "research_stop_loss_pct": config.stop_loss_pct,
        "research_max_hold_bars": config.max_hold_bars,
    }


def _buy_the_dip_signal_row(row: pd.Series, *, config: ResearchConfig) -> dict[str, Any]:
    score = _buy_the_dip_score(row, config=config)
    return {
        "timestamp": row["timestamp"],
        "close": float(row["close"]),
        "orderbook_spread": float(row.get("orderbook_spread", 0.0)),
        "quote_imbalance": float(row.get("quote_imbalance", 0.0)),
        "scalping_spread_bps": float(row.get("scalping_spread_bps", 0.0)),
        "scalping_quote_imbalance": float(row.get("scalping_quote_imbalance", 0.0)),
        "strategy_name": BUY_THE_DIP_STRATEGY,
        "entry_reason": "buy_the_dip_oversold_reversal_candidate",
        "strategy_score": score,
        "strategy_confidence": max(0.45, min(0.95, 0.45 + score * 0.45)),
        "quant_score": score,
        "quant_confidence": max(0.45, min(0.95, 0.45 + score * 0.45)),
        "regime": "mean_reversion_research",
        "blocked_by": None,
        "block_reason": None,
        "ml_buy_probability": 1.0,
        "ml_sell_probability": 0.0,
        "_probability": 1.0,
        "research_timeframe": config.timeframe,
        "research_take_profit_pct": config.take_profit_pct,
        "research_stop_loss_pct": config.stop_loss_pct,
        "research_max_hold_bars": config.max_hold_bars,
        "rsi_14": float(row.get("rsi_14", 0.0)),
        "rolling_zscore_20": float(row.get("rolling_zscore_20", 0.0)),
        "vwap_distance": float(row.get("vwap_distance", 0.0)),
        "ema_20_distance": float(row.get("ema_20_distance", 0.0)),
        "recent_drawdown_50": float(row.get("recent_drawdown_50", 0.0)),
        "volume_zscore_20": float(row.get("volume_zscore_20", 0.0)),
        "lower_wick_ratio": float(row.get("lower_wick_ratio", 0.0)),
        "close_position_in_candle": float(row.get("close_position_in_candle", 0.0)),
        "reversal_confirmation": bool(row.get("reversal_confirmation", False)),
        "higher_timeframe_regime_filter": bool(config.higher_timeframe_regime_filter),
    }


def _v3_signal_row(
    row: pd.Series,
    *,
    config: ResearchConfig,
    strategy_name: str,
    entry_reason: str,
    regime: str,
) -> dict[str, Any]:
    score = _v3_score(row, config=config, strategy_name=strategy_name)
    return {
        "timestamp": row["timestamp"],
        "close": float(row["close"]),
        "orderbook_spread": float(row.get("orderbook_spread", 0.0)),
        "quote_imbalance": float(row.get("quote_imbalance", 0.0)),
        "scalping_spread_bps": float(row.get("scalping_spread_bps", 0.0)),
        "scalping_quote_imbalance": float(row.get("scalping_quote_imbalance", 0.0)),
        "strategy_name": strategy_name,
        "entry_reason": entry_reason,
        "strategy_score": score,
        "strategy_confidence": max(0.45, min(0.95, 0.45 + score * 0.45)),
        "quant_score": score,
        "quant_confidence": max(0.45, min(0.95, 0.45 + score * 0.45)),
        "regime": regime,
        "blocked_by": None,
        "block_reason": None,
        "ml_buy_probability": 1.0,
        "ml_sell_probability": 0.0,
        "_probability": 1.0,
        "research_timeframe": config.timeframe,
        "research_take_profit_pct": config.take_profit_pct,
        "research_stop_loss_pct": config.stop_loss_pct,
        "research_max_hold_bars": config.max_hold_bars,
        "rsi_14": float(row.get("rsi_14", 0.0)),
        "volume_zscore_20": float(row.get("volume_zscore_20", 0.0)),
        "atr_expansion_20": float(row.get("atr_expansion_20", 0.0)),
        "pullback_from_high_50": float(row.get("pullback_from_high_50", 0.0)),
        "body_vs_avg_20": float(row.get("body_vs_avg_20", 0.0)),
        "ema_20_slope_5": float(row.get("ema_20_slope_5", 0.0)),
        "ema_50_slope_5": float(row.get("ema_50_slope_5", 0.0)),
        "support": config.support,
        "confirmation": config.confirmation,
        "breakout_lookback": config.breakout_lookback,
        "consolidation_lookback": config.consolidation_lookback,
    }


def _v3_score(row: pd.Series, *, config: ResearchConfig, strategy_name: str) -> float:
    if strategy_name == UPTREND_PULLBACK_STRATEGY:
        pullback_max = max(0.0001, float(config.pullback_max_pct or 0.08))
        rsi_mid = ((config.rsi_min or 35.0) + (config.rsi_max or 55.0)) / 2
        support_key = f"support_distance_{config.support or 'ema20'}_abs"
        support_distance = _metric_float(row.get(support_key))
        components = [
            max(0.0, _metric_float(row.get("pullback_from_high_50")) / pullback_max),
            max(0.0, 1.0 - support_distance / max(0.0001, float(config.support_distance_pct or 0.01))),
            max(0.0, 1.0 - abs(_metric_float(row.get("rsi_14")) - rsi_mid) / 25.0),
            max(0.0, _metric_float(row.get("lower_wick_ratio"))),
            max(0.0, min(1.0, _metric_float(row.get("ema_20_slope_5")) * 500)),
        ]
    else:
        components = [
            max(0.0, min(1.0, _metric_float(row.get("log_return_3")) / max(0.0001, float(config.min_recent_return_pct or 0.002)))),
            max(0.0, min(1.0, _metric_float(row.get("volume_zscore_20")) / max(0.0001, float(config.min_volume_zscore or 0.5)))),
            max(0.0, min(1.0, _metric_float(row.get("body_vs_avg_20")) / max(0.0001, float(config.min_body_vs_avg or 1.0)))),
            max(0.0, min(1.0, _metric_float(row.get("atr_expansion_20")) / max(0.0001, float(config.max_atr_expansion or 2.5)))),
            max(0.0, min(1.0, _metric_float(row.get("ema_50_slope_5")) * 500)),
        ]
    return float(max(0.0, min(1.0, sum(min(1.0, value) for value in components) / len(components))))


def _buy_the_dip_score(row: pd.Series, *, config: ResearchConfig) -> float:
    rsi_threshold = max(1.0, float(config.rsi_threshold or 30.0))
    zscore_threshold = abs(float(config.zscore_threshold or -2.0))
    vwap_threshold = abs(float(config.vwap_distance_threshold or -0.005))
    drawdown_threshold = max(0.0001, float(config.drawdown_threshold or 0.01))
    volume_threshold = max(0.0001, float(config.min_volume_zscore or 1.0))
    components = [
        max(0.0, (rsi_threshold - _metric_float(row.get("rsi_14"))) / rsi_threshold),
        max(0.0, abs(min(0.0, _metric_float(row.get("rolling_zscore_20")))) / zscore_threshold),
        max(0.0, abs(min(0.0, _metric_float(row.get("vwap_distance")))) / vwap_threshold),
        max(0.0, _metric_float(row.get("recent_drawdown_50")) / drawdown_threshold),
        max(0.0, _metric_float(row.get("volume_zscore_20")) / volume_threshold),
        max(0.0, _metric_float(row.get("lower_wick_ratio"))),
        max(0.0, _metric_float(row.get("close_position_in_candle"))),
    ]
    return float(max(0.0, min(1.0, sum(min(1.0, value) for value in components) / len(components))))


def _research_block_bucket(reason: str) -> str:
    if reason.startswith("regime") or reason in {"trend_not_confirmed", "volatility_not_tradeable"}:
        return "regime_filter"
    if "spread" in reason:
        return "spread"
    return "quant_strategy"


def _exit_result(gross_return: float, exit_reason: str, hold_bars: int, exit_high: float, exit_low: float) -> dict[str, Any]:
    return {
        "gross_return": float(gross_return),
        "exit_reason": exit_reason,
        "hold_bars": int(hold_bars),
        "exit_high": float(exit_high),
        "exit_low": float(exit_low),
    }


def _empty_research_metrics(reason: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "number_of_trades": 0,
        "net_return_pct": 0.0,
        "gross_return_pct": 0.0,
        "profit_factor_net": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_net": 0.0,
        "expectancy": 0.0,
        "trade_details": [],
    }


def single_trade_return_concentration(trade_details: list[dict[str, Any]]) -> float:
    returns = [_metric_float(trade.get("net_return_pct", trade.get("net_return"))) for trade in trade_details]
    positive_returns = [value for value in returns if value > 0]
    total_positive_return = sum(positive_returns)
    if total_positive_return <= 0:
        return 0.0
    return max(positive_returns) / total_positive_return


def research_rank_score(metrics: dict[str, Any], readiness: dict[str, Any]) -> float:
    base = _metric_float(metrics.get("net_return_pct")) * 10_000
    base += min(5.0, _profit_factor_value(metrics.get("profit_factor_net"))) * 10
    base += int(metrics.get("number_of_trades", 0) or 0) * 0.1
    base -= _metric_float(metrics.get("max_drawdown_pct")) * 1_000
    if not readiness.get("economically_viable"):
        base -= 1_000_000
    if not readiness.get("paper_forward_eligible"):
        base -= 10_000
    return base


def research_rank_details(
    metrics: dict[str, Any],
    readiness: dict[str, Any],
    *,
    concentration: float | None = None,
    walk_forward: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trades = int(metrics.get("number_of_trades", 0) or 0)
    concentration_value = _metric_float(concentration)
    walk_forward = walk_forward or empty_walk_forward_result(0)
    statistically_weak = trades < MIN_RESEARCH_TRADES
    trade_count_score = max(0.0, min(1.0, trades / PREFERRED_RESEARCH_TRADES))
    concentration_penalty = max(0.0, min(1.0, concentration_value))
    profit_factor = _profit_factor_value(metrics.get("profit_factor_net"))
    profit_factor_reliable = not (statistically_weak and profit_factor >= MIN_RESEARCH_PROFIT_FACTOR_NET)
    reliability_score = trade_count_score * max(0.0, 1.0 - concentration_penalty)
    if statistically_weak:
        reliability_score *= max(0.05, trades / max(1, MIN_RESEARCH_TRADES)) * 0.25
    if not profit_factor_reliable:
        reliability_score *= 0.50

    raw_rank_score = research_rank_score(metrics, readiness)
    adjusted_rank_score = raw_rank_score
    adjusted_rank_score += reliability_score * 10_000
    adjusted_rank_score -= (1.0 - trade_count_score) * 20_000
    adjusted_rank_score -= concentration_penalty * 25_000
    if not walk_forward.get("walk_forward_passed"):
        adjusted_rank_score -= 125_000
    adjusted_rank_score += int(walk_forward.get("folds_profitable_count", 0) or 0) * 2_500
    adjusted_rank_score += _metric_float(walk_forward.get("median_fold_net_return_pct")) * 50_000

    reasons: list[str] = []
    if statistically_weak:
        adjusted_rank_score -= 100_000 + (MIN_RESEARCH_TRADES - trades) * 2_000
        reasons.append("number_of_trades_below_20")
    if trades == 1:
        adjusted_rank_score -= 100_000
        reasons.append("one_trade_result_not_reliable")
    elif 1 < trades < 5:
        adjusted_rank_score -= 50_000
        reasons.append("very_low_trade_count")
    if concentration_value > MAX_SINGLE_TRADE_RETURN_SHARE:
        adjusted_rank_score -= 75_000
        reasons.append("single_trade_return_concentration_too_high")
    if not walk_forward.get("walk_forward_passed"):
        reasons.append("walk_forward_not_passed")
    if not profit_factor_reliable:
        adjusted_rank_score -= 25_000
        reasons.append("profit_factor_not_reliable_at_low_trade_count")
    if not readiness.get("economically_viable"):
        reasons.append("not_economically_viable")

    return {
        "raw_rank_score": raw_rank_score,
        "statistically_weak": statistically_weak,
        "trade_count_score": trade_count_score,
        "concentration_penalty": concentration_penalty,
        "reliability_score": reliability_score,
        "profit_factor_reliable": profit_factor_reliable,
        "adjusted_rank_score": adjusted_rank_score,
        "reason_ranked_lower_if_any": ";".join(dict.fromkeys(reasons)) if reasons else None,
    }


def _metric_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _profit_factor_value(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(parsed):
        return 1_000_000.0 if parsed > 0 else 0.0
    if math.isnan(parsed):
        return 0.0
    return parsed


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "parameter_set_id",
        "strategy_name",
        "timeframe",
        "take_profit_pct",
        "stop_loss_pct",
        "max_hold_bars",
        "number_of_trades",
        "gross_return_pct",
        "net_return_pct",
        "profit_factor_net",
        "max_drawdown_pct",
        "win_rate_net",
        "expectancy",
        "round_trip_estimated_cost_pct",
        "promotion_required_return_pct",
        "gross_winners_became_net_losers",
        "single_trade_return_concentration",
        "walk_forward_passed",
        "folds_profitable_count",
        "folds_with_min_trades_count",
        "worst_fold_net_return_pct",
        "median_fold_net_return_pct",
        "research_promising",
        "fallback_prediction_used",
        "active_model_valid",
        "economically_viable",
        "paper_forward_eligible",
        "rejection_reasons",
        "rank_score",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    return [field for field in preferred if any(field in row for row in rows)] + extras


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
