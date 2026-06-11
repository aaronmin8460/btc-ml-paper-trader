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
from itertools import combinations, product
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
    parse_timeframe_duration,
    stale_threshold_for_timeframe,
)
from app.db.database import SessionLocal, init_db
from app.db.models import CollectedMarketData
from app.ml.registry import ModelRegistry
from app.risk.risk_manager import PositionState
from app.strategy.strategies import MarketContext, MarketRegimeFilter, TrendPullbackStrategy


LEGACY_RESEARCH_TIMEFRAMES = ("5Min", "15Min")
V3_RESEARCH_TIMEFRAMES = ("15Min", "1H")
HTF_RESEARCH_TIMEFRAMES = ("1H", "4H", "1D")
RESEARCH_TIMEFRAMES = LEGACY_RESEARCH_TIMEFRAMES
SUPPORTED_RESEARCH_TIMEFRAMES = ("5Min", "15Min", "1H", "4H", "1D")
TREND_PULLBACK_STRATEGY = "trend_pullback"
BUY_THE_DIP_STRATEGY = "buy_the_dip_mean_reversion"
UPTREND_PULLBACK_STRATEGY = "uptrend_pullback"
VOLATILITY_BREAKOUT_STRATEGY = "volatility_breakout"
HTF_TREND_CONTINUATION_STRATEGY = "htf_trend_continuation"
HTF_VOLATILITY_EXPANSION_BREAKOUT_STRATEGY = "htf_volatility_expansion_breakout"
HTF_RISK_OFF_HOLD_FILTER_STRATEGY = "htf_risk_off_hold_filter"
VOLATILITY_FOCUS_STRATEGY = "volatility_focus"
V3_STRATEGIES = (UPTREND_PULLBACK_STRATEGY, VOLATILITY_BREAKOUT_STRATEGY)
HTF_STRATEGIES = (
    HTF_TREND_CONTINUATION_STRATEGY,
    HTF_VOLATILITY_EXPANSION_BREAKOUT_STRATEGY,
    HTF_RISK_OFF_HOLD_FILTER_STRATEGY,
)
VOLATILITY_FOCUS_STRATEGIES = (
    VOLATILITY_BREAKOUT_STRATEGY,
    HTF_VOLATILITY_EXPANSION_BREAKOUT_STRATEGY,
)
VOLATILITY_FOCUS_DEFAULT_TIMEFRAMES = ("1H",)
VOLATILITY_FOCUS_SUPPORTED_TIMEFRAMES = ("15Min", "1H", "4H")
STRATEGY_CHOICES = (
    "all",
    TREND_PULLBACK_STRATEGY,
    BUY_THE_DIP_STRATEGY,
    UPTREND_PULLBACK_STRATEGY,
    VOLATILITY_BREAKOUT_STRATEGY,
    HTF_TREND_CONTINUATION_STRATEGY,
    HTF_VOLATILITY_EXPANSION_BREAKOUT_STRATEGY,
    HTF_RISK_OFF_HOLD_FILTER_STRATEGY,
    VOLATILITY_FOCUS_STRATEGY,
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
HTF_TREND_TAKE_PROFIT_VALUES = (0.03, 0.05, 0.08, 0.12)
HTF_TREND_STOP_LOSS_VALUES = (0.015, 0.025, 0.04, 0.06)
HTF_BREAKOUT_TAKE_PROFIT_VALUES = (0.04, 0.06, 0.10, 0.15)
HTF_BREAKOUT_STOP_LOSS_VALUES = (0.02, 0.03, 0.05)
HTF_MAX_HOLD_BARS_VALUES = (12, 24, 48, 96)
VOLATILITY_FOCUS_TAKE_PROFIT_VALUES = (0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08)
VOLATILITY_FOCUS_STOP_LOSS_VALUES = (0.012, 0.015, 0.018, 0.02, 0.025, 0.03)
VOLATILITY_FOCUS_MAX_HOLD_BARS_VALUES = (12, 24, 36, 48, 72, 96)
VOLATILITY_FOCUS_BREAKOUT_LOOKBACK_VALUES = (12, 16, 20, 24, 36, 48)
VOLATILITY_FOCUS_CONSOLIDATION_LOOKBACK_VALUES = (8, 12, 16, 20, 24)
VOLATILITY_FOCUS_MIN_BODY_VS_AVG_VALUES = (0.8, 1.0, 1.2, 1.5)
VOLATILITY_FOCUS_MIN_RECENT_RETURN_VALUES = (0.001, 0.002, 0.003, 0.004, 0.006, 0.008)
VOLATILITY_FOCUS_MIN_TREND_STRENGTH_VALUES = (0.0, 0.02, 0.05, 0.08, 0.10)
VOLATILITY_FOCUS_MIN_VOLUME_ZSCORE_VALUES = (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
VOLATILITY_FOCUS_MAX_ATR_EXPANSION_VALUES = (1.8, 2.2, 2.6, 3.0, 3.5)
VOLATILITY_FOCUS_V7_MIN_CONFIGS = 4000
VOLATILITY_FOCUS_V7_BASE_CONFIGS = 1000
VOLATILITY_FOCUS_V8_MIN_CONFIGS = 5000
VOLATILITY_FOCUS_V9_CONFIGS = 3000
VOLATILITY_FOCUS_TRACK_A = "A_vff_vbo_00808_expansion"
VOLATILITY_FOCUS_TRACK_B = "B_vff_vbo_00001_exit_fix"
VOLATILITY_FOCUS_TRACK_M = "M_vff_vbo_00001_maker_only"
VOLATILITY_FOCUS_TRACK_T = "T_taker_survival_swing_breakout"
VOLATILITY_FOCUS_TRACK_M9 = "M9_v8m_00086_drawdown_reduction"
VOLATILITY_FOCUS_V9_ANCHOR_PARAMETER_SET_ID = "v8m_00086"
VOLATILITY_FOCUS_V9_ANCHOR_MAX_DRAWDOWN_PCT = 0.1079
EXIT_MODE_FIXED = "fixed_tp_sl_timeout"
EXIT_MODE_BREAK_EVEN_1R = "break_even_stop_after_1r"
EXIT_MODE_TRAILING_1R = "trailing_stop_after_1r"
EXIT_MODE_MFE_PROTECT_1R_50 = "mfe_protection_1r_50"
EXIT_MODE_TIME_STOP_MOMENTUM_WEAK = "time_stop_momentum_weak"
EXIT_MODE_BREAK_EVEN_AFTER_1R = "break_even_after_1r"
EXIT_MODE_TRAILING_AFTER_1R = "trailing_after_1r"
EXIT_MODE_MFE_PROTECTION_EXIT = "mfe_protection_exit"
VOLATILITY_FOCUS_V7_EXIT_MODES = (
    EXIT_MODE_FIXED,
    EXIT_MODE_BREAK_EVEN_1R,
    EXIT_MODE_TRAILING_1R,
    EXIT_MODE_MFE_PROTECT_1R_50,
    EXIT_MODE_TIME_STOP_MOMENTUM_WEAK,
)
VOLATILITY_FOCUS_V8_MAKER_EXIT_MODES = (
    EXIT_MODE_FIXED,
    EXIT_MODE_TIME_STOP_MOMENTUM_WEAK,
    EXIT_MODE_MFE_PROTECTION_EXIT,
)
VOLATILITY_FOCUS_V8_TAKER_EXIT_MODES = (
    EXIT_MODE_FIXED,
    EXIT_MODE_BREAK_EVEN_AFTER_1R,
    EXIT_MODE_TRAILING_AFTER_1R,
    EXIT_MODE_MFE_PROTECTION_EXIT,
    EXIT_MODE_TIME_STOP_MOMENTUM_WEAK,
)
VOLATILITY_FOCUS_V9_EXIT_MODES = (
    EXIT_MODE_FIXED,
    EXIT_MODE_TIME_STOP_MOMENTUM_WEAK,
    EXIT_MODE_MFE_PROTECTION_EXIT,
    EXIT_MODE_BREAK_EVEN_AFTER_1R,
)
VOLATILITY_FOCUS_V9_SEARCH_SPACE = {
    "strategy_name": (VOLATILITY_BREAKOUT_STRATEGY,),
    "timeframe": ("1H",),
    "take_profit_pct": (0.045, 0.0475, 0.05, 0.0525, 0.055),
    "stop_loss_pct": (0.016, 0.018, 0.019, 0.02, 0.021, 0.022),
    "max_hold_bars": (36, 42, 48, 54, 60),
    "breakout_lookback": (18, 20, 22),
    "consolidation_lookback": (10, 12, 14),
    "min_body_vs_avg": (1.0, 1.1, 1.2, 1.3),
    "min_recent_return_pct": (0.0025, 0.003, 0.0035),
    "min_trend_strength": (0.0, 0.02, 0.03),
    "max_atr_expansion": (2.2, 2.6, 3.0),
    "min_volume_zscore": (0.0, 0.25, 0.5),
    "exit_mode": VOLATILITY_FOCUS_V9_EXIT_MODES,
}
VOLATILITY_FOCUS_V9_ANCHOR_SPEC = {
    "exit_mode": EXIT_MODE_FIXED,
    "take_profit_pct": 0.045,
    "stop_loss_pct": 0.022,
    "max_hold_bars": 48,
    "breakout_lookback": 20,
    "consolidation_lookback": 12,
    "min_body_vs_avg": 1.2,
    "min_recent_return_pct": 0.003,
    "min_trend_strength": 0.0,
    "max_atr_expansion": 3.0,
    "min_volume_zscore": 0.25,
}
VOLATILITY_FOCUS_V9_RANKING_PRIORITY = (
    "maker_current_net_return_pct > 0",
    "maker_current_profit_factor >= 1.05",
    "number_of_trades >= 20",
    "walk_forward_passed == true",
    "folds_with_min_trades_count == fold_count",
    "max_drawdown_pct <= configured_drawdown_gate",
    "beats_buy_hold_risk_adjusted == true",
    "beats_dca_daily_risk_adjusted == true",
    "lower_single_trade_return_concentration",
    "higher_median_fold_net_return_pct",
    "lower_worst_fold_net_return_pct_loss_magnitude",
)
VOLATILITY_FOCUS_V9_TERMINAL_FAILURE_RECOMMENDATION = (
    "abandon_1h_volatility_breakout_maker_only_and_switch_strategy_family"
)
VOLATILITY_FOCUS_V9_TERMINAL_FOUND_RECOMMENDATION = (
    "maker_only_research_candidate_found_but_not_live_tradable"
)
VOLATILITY_FOCUS_V9_TERMINAL_BLOCKERS = (
    "max_drawdown_above_configured_limit",
    "maker_current_net_return_not_positive",
    "maker_current_profit_factor_below_1_05",
    "number_of_trades_below_20",
    "walk_forward_not_passed",
    "folds_with_min_trades_below_required",
    "does_not_beat_buy_and_hold_risk_adjusted",
    "does_not_beat_dca_risk_adjusted",
    "statistically_weak",
    "invalid_data_source",
)
VOLATILITY_FOCUS_RECOMMENDATIONS = (
    "no_edge_found",
    "cost_drag_only",
    "entry_quality_problem",
    "exit_logic_problem",
    "more_data_needed",
    "candidate_found_keep_trading_disabled",
    "refine_volatility_breakout_more",
    "retire_volatility_breakout",
)
RISK_OFF_FILTER_PROFILES = (
    {
        "drawdown_threshold": 0.12,
        "max_atr_expansion": 2.0,
        "min_recent_return_pct": -0.04,
    },
    {
        "drawdown_threshold": 0.18,
        "max_atr_expansion": 2.4,
        "min_recent_return_pct": -0.06,
    },
    {
        "drawdown_threshold": 0.25,
        "max_atr_expansion": 3.0,
        "min_recent_return_pct": -0.08,
    },
)
COST_SCENARIOS = ("current_taker", "maker_current", "maker_low_slippage", "zero_cost_sanity")
DERIVATION_SOURCES_BY_TIMEFRAME = {
    "1H": ("15Min",),
    "4H": ("1H", "15Min"),
    "1D": ("1H", "15Min"),
}
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
    require_ema_trend_filter: bool | None = None
    require_positive_ema20_slope: bool | None = None
    require_close_above_ema200: bool | None = None
    max_breakout_candle_atr_multiple: float | None = None
    min_close_position_in_candle: float | None = None
    max_recent_runup_pct: float | None = None
    min_consolidation_compression: float | None = None
    require_volume_expansion: bool | None = None
    max_atr_percentile: float | None = None
    track_id: str | None = None
    exit_mode: str = EXIT_MODE_FIXED


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
    parser.add_argument("--max-rows-4h", type=int, default=None, help="Maximum 4H rows.")
    parser.add_argument("--max-rows-1d", type=int, default=None, help="Maximum 1D rows.")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        choices=SUPPORTED_RESEARCH_TIMEFRAMES,
        default=None,
        help="Timeframes to evaluate. Defaults preserve legacy strategy behavior and use 15Min/1H for v3.",
    )
    parser.add_argument("--higher-timeframe-audit", action="store_true", help="Prefer 1H/4H/1D audit defaults.")
    parser.add_argument("--exclude-15min", action="store_true", help="Do not evaluate 15Min configs.")
    parser.add_argument(
        "--volatility-focus",
        action="store_true",
        help="Run only focused 1H/optional 4H volatility breakout research.",
    )
    parser.add_argument("--export-trades", action="store_true", help="Export selected config trade-by-trade audit logs.")
    parser.add_argument(
        "--export-focused-trades",
        action="store_true",
        help="Export volatility-focus top config trade audits with volatility_focus_top_* filenames.",
    )
    parser.add_argument(
        "--trade-log-dir",
        default="logs/trade_audits",
        help="Directory for trade audit CSV/JSONL outputs.",
    )
    parser.add_argument(
        "--top-n-trade-configs",
        type=int,
        default=10,
        help="Number of ranked configs to export when --export-trades is set.",
    )
    parser.add_argument(
        "--include-rejected-trades",
        action="store_true",
        help="Include rejected configs in trade audit export selection.",
    )
    parser.add_argument(
        "--audit-mode",
        choices=("standard", "reality"),
        default="standard",
        help="Enable Strategy Reality Audit v5 diagnostics when set to reality.",
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
        "--min-focused-trades",
        type=int,
        default=MIN_RESEARCH_TRADES,
        help="Minimum total trades for volatility-focus research-promising configs.",
    )
    parser.add_argument(
        "--target-focused-trades",
        type=int,
        default=PREFERRED_RESEARCH_TRADES,
        help="Preferred trade count for volatility-focus ranking.",
    )
    parser.add_argument(
        "--save-focused-summary",
        default="logs/volatility_focus_summary.json",
        help="Path for the dedicated volatility focus summary JSON.",
    )
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
    volatility_focus = bool(args.volatility_focus or args.strategy == VOLATILITY_FOCUS_STRATEGY)
    strategy = VOLATILITY_FOCUS_STRATEGY if volatility_focus else args.strategy
    timeframes = _resolve_requested_timeframes(
        strategy,
        args.timeframes,
        higher_timeframe_audit=args.higher_timeframe_audit,
        exclude_15min=args.exclude_15min,
    )
    row_limits = _row_limits_by_timeframe(
        default_limit=bar_limit,
        max_rows_per_timeframe=args.max_rows_per_timeframe,
        max_rows_5min=args.max_rows_5min,
        max_rows_15min=args.max_rows_15min,
        max_rows_1h=args.max_rows_1h,
        max_rows_4h=args.max_rows_4h,
        max_rows_1d=args.max_rows_1d,
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
        strategy=strategy,
        max_buy_dip_configs=args.max_buy_dip_configs,
        max_v3_configs=args.max_v3_configs,
        walk_forward_splits=args.walk_forward_splits,
        min_trades=args.min_focused_trades if volatility_focus else args.min_trades,
        min_trades_per_split=args.min_trades_per_split,
        output_dir=Path(settings.log_dir),
        allow_synthetic_fallback=_synthetic_research_mode_enabled(),
        audit_mode=args.audit_mode,
        export_trades=args.export_trades,
        trade_log_dir=Path(args.trade_log_dir),
        top_n_trade_configs=args.top_n_trade_configs,
        include_rejected_trades=args.include_rejected_trades,
        higher_timeframe_audit=args.higher_timeframe_audit,
        exclude_15min=args.exclude_15min,
        volatility_focus=volatility_focus,
        min_focused_trades=args.min_focused_trades,
        target_focused_trades=args.target_focused_trades,
        save_focused_summary=Path(args.save_focused_summary),
        export_focused_trades=args.export_focused_trades,
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
    audit_mode: str = "standard",
    export_trades: bool = False,
    trade_log_dir: Path | None = None,
    top_n_trade_configs: int = 10,
    include_rejected_trades: bool = False,
    higher_timeframe_audit: bool = False,
    exclude_15min: bool = False,
    volatility_focus: bool = False,
    min_focused_trades: int = MIN_RESEARCH_TRADES,
    target_focused_trades: int = PREFERRED_RESEARCH_TRADES,
    save_focused_summary: Path | None = None,
    export_focused_trades: bool = False,
) -> dict[str, Any]:
    settings = research_settings(base_settings)
    volatility_focus = bool(volatility_focus or strategy == VOLATILITY_FOCUS_STRATEGY)
    if volatility_focus:
        strategy = VOLATILITY_FOCUS_STRATEGY
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
    if min_focused_trades <= 0:
        raise ValueError("min_focused_trades must be positive")
    if target_focused_trades <= 0:
        raise ValueError("target_focused_trades must be positive")
    if min_trades_per_split <= 0:
        raise ValueError("min_trades_per_split must be positive")
    if audit_mode not in {"standard", "reality"}:
        raise ValueError("audit_mode must be standard or reality")
    if top_n_trade_configs <= 0:
        raise ValueError("top_n_trade_configs must be positive")
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
    requested_strategy_timeframes = _resolve_requested_timeframes(
        strategy,
        timeframes,
        higher_timeframe_audit=higher_timeframe_audit,
        exclude_15min=exclude_15min,
    )
    row_limits = max_rows_by_timeframe or {
        timeframe: int(bar_limit) for timeframe in requested_strategy_timeframes
    }
    for timeframe in requested_strategy_timeframes:
        limit = int(row_limits.get(timeframe, bar_limit))
        raw_source_bars_by_timeframe = dict(bars_by_timeframe)
        raw_source_reports_by_timeframe = dict(data_source_reports)
        result = await _fetch_or_derive_research_bars(
            client,
            settings,
            timeframe=timeframe,
            limit=limit,
            existing_bars_by_timeframe=raw_source_bars_by_timeframe,
            existing_reports_by_timeframe=raw_source_reports_by_timeframe,
            row_limits=row_limits,
            session_factory=session_factory,
            allow_synthetic_fallback=allow_synthetic_fallback,
            now=current_time,
            start=window_start,
            end=window_end,
            allowed_derivation_source_timeframes=_allowed_raw_derivation_sources_for_timeframe(
                timeframe,
                requested_strategy_timeframes,
                volatility_focus=volatility_focus,
            ),
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
        timeframes=requested_strategy_timeframes,
        audit_mode=audit_mode,
        volatility_focus=volatility_focus,
        min_focused_trades=min_focused_trades,
        target_focused_trades=target_focused_trades,
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
        requested_timeframes=requested_strategy_timeframes,
        audit_mode=audit_mode,
        bars_by_timeframe=bars_by_timeframe,
    )
    if audit_mode == "reality":
        reality_summary_path = output_path / "strategy_reality_audit_summary.json"
        summary = build_strategy_reality_audit_summary(
            summary,
            rows,
            settings,
            data_source_reports=data_source_reports,
            bars_by_timeframe=bars_by_timeframe,
            reality_summary_path=reality_summary_path,
            cost_scenarios=COST_SCENARIOS,
            trade_audit_paths=[],
        )
    trade_audit_paths: list[dict[str, Any]] = []
    if export_trades:
        trade_audit_paths = export_trade_audit_logs(
            rows,
            bars_by_timeframe,
            settings,
            data_source_reports=data_source_reports,
            strategy=strategy,
            max_buy_dip_configs=max_buy_dip_configs,
            max_v3_configs=max_v3_configs,
            walk_forward_splits=walk_forward_splits,
            min_trades_per_split=min_trades_per_split,
            timeframes=requested_strategy_timeframes,
            output_dir=trade_log_dir or (output_path / "trade_audits"),
            top_n=top_n_trade_configs,
            include_rejected=include_rejected_trades,
        )
        summary["trade_audits"] = trade_audit_paths
        summary["best_trade_audit_path"] = (
            trade_audit_paths[0].get("csv_path") if trade_audit_paths else None
        )
        if audit_mode == "reality":
            summary = build_strategy_reality_audit_summary(
                summary,
                rows,
                settings,
                data_source_reports=data_source_reports,
                bars_by_timeframe=bars_by_timeframe,
                reality_summary_path=output_path / "strategy_reality_audit_summary.json",
                cost_scenarios=COST_SCENARIOS,
                trade_audit_paths=trade_audit_paths,
            )
    focused_trade_audit_paths: list[dict[str, Any]] = []
    if volatility_focus:
        focused_summary_path = save_focused_summary or (output_path / "volatility_focus_summary.json")
        focused_output_stem = _focused_output_stem(focused_summary_path)
        if export_focused_trades:
            focused_trade_audit_paths = export_trade_audit_logs(
                rows,
                bars_by_timeframe,
                settings,
                data_source_reports=data_source_reports,
                strategy=VOLATILITY_FOCUS_STRATEGY,
                max_buy_dip_configs=max_buy_dip_configs,
                max_v3_configs=max_v3_configs,
                walk_forward_splits=walk_forward_splits,
                min_trades_per_split=min_trades_per_split,
                timeframes=requested_strategy_timeframes,
                output_dir=trade_log_dir or (output_path / "trade_audits"),
                top_n=top_n_trade_configs,
                include_rejected=True,
                filename_prefix=f"{focused_output_stem}_top_",
            )
            summary["focused_trade_audits"] = focused_trade_audit_paths
        focused_output_dir = focused_summary_path.parent
        focused_top_csv_path = focused_output_dir / f"{focused_output_stem}_top_configs.csv"
        focused_rejections_path = focused_output_dir / f"{focused_output_stem}_rejections.json"
        focused_summary = build_volatility_focus_summary(
            rows,
            settings,
            base_summary=summary,
            data_source_reports=data_source_reports,
            bars_by_timeframe=bars_by_timeframe,
            min_focused_trades=min_focused_trades,
            target_focused_trades=target_focused_trades,
            max_focused_configs=max_v3_configs,
            focused_summary_path=focused_summary_path,
            top_configs_csv_path=focused_top_csv_path,
            rejections_path=focused_rejections_path,
            trade_audit_paths=focused_trade_audit_paths,
        )
        write_volatility_focus_outputs(
            rows,
            focused_summary,
            summary_path=focused_summary_path,
            top_configs_csv_path=focused_top_csv_path,
            rejections_path=focused_rejections_path,
        )
        summary["volatility_focus"] = focused_summary
        summary["volatility_focus_summary_path"] = str(focused_summary_path)
        summary["volatility_focus_top_configs_csv_path"] = str(focused_top_csv_path)
        summary["volatility_focus_rejections_path"] = str(focused_rejections_path)
    write_research_outputs(rows, summary, csv_path=csv_path, summary_path=summary_path)
    if audit_mode == "reality":
        (output_path / "strategy_reality_audit_summary.json").write_text(
            json.dumps(_json_safe(summary), indent=2, allow_nan=False),
            encoding="utf-8",
        )
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


def _focused_output_stem(focused_summary_path: Path) -> str:
    stem = focused_summary_path.stem
    return stem.removesuffix("_summary") if stem.endswith("_summary") else "volatility_focus"


def research_config_settings(
    settings: Settings,
    config: ResearchConfig,
    *,
    cost_scenario: str | None = None,
) -> Settings:
    data = {
        **settings.model_dump(),
        "take_profit_pct": config.take_profit_pct,
        "stop_loss_pct": config.stop_loss_pct,
        "scalping_take_profit_pct": config.take_profit_pct,
        "scalping_stop_loss_pct": config.stop_loss_pct,
        "label_horizon_bars": max(1, int(config.max_hold_bars or settings.label_horizon_bars)),
    }
    if cost_scenario == "current_taker":
        data["backtest_use_taker_fees"] = True
    elif cost_scenario == "maker_current":
        data["backtest_use_taker_fees"] = False
    elif cost_scenario == "maker_low_slippage":
        data.update(
            {
                "backtest_use_taker_fees": False,
                "slippage_bps": 2,
                "max_spread_bps": 2,
            }
        )
    elif cost_scenario == "zero_cost_sanity":
        data.update(
            {
                "backtest_use_taker_fees": False,
                "taker_fee_bps": 0,
                "maker_fee_bps": 0,
                "slippage_bps": 0,
                "max_spread_bps": 0,
            }
        )
    elif cost_scenario is not None:
        raise ValueError(f"Unsupported cost scenario: {cost_scenario}")
    return research_settings(Settings(_env_file=None, **data))


def _row_limits_by_timeframe(
    *,
    default_limit: int,
    max_rows_per_timeframe: int | None = None,
    max_rows_5min: int | None = None,
    max_rows_15min: int | None = None,
    max_rows_1h: int | None = None,
    max_rows_4h: int | None = None,
    max_rows_1d: int | None = None,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, int]:
    base = max(1, int(max_rows_per_timeframe or default_limit))
    limits = {
        "5Min": max(1, int(max_rows_5min or base)),
        "15Min": max(1, int(max_rows_15min or base)),
        "1H": max(1, int(max_rows_1h or base)),
        "4H": max(1, int(max_rows_4h or base)),
        "1D": max(1, int(max_rows_1d or base)),
    }
    requested = tuple(timeframes or SUPPORTED_RESEARCH_TIMEFRAMES)
    return {timeframe: limits[timeframe] for timeframe in requested}


def _resolve_requested_timeframes(
    strategy: str,
    timeframes: tuple[str, ...] | list[str] | None,
    *,
    higher_timeframe_audit: bool = False,
    exclude_15min: bool = False,
) -> tuple[str, ...]:
    if timeframes:
        requested = tuple(dict.fromkeys(str(timeframe) for timeframe in timeframes))
    elif strategy == VOLATILITY_FOCUS_STRATEGY:
        requested = VOLATILITY_FOCUS_DEFAULT_TIMEFRAMES
    elif higher_timeframe_audit:
        requested = HTF_RESEARCH_TIMEFRAMES
    elif strategy in V3_STRATEGIES or strategy == "all":
        requested = V3_RESEARCH_TIMEFRAMES
    elif strategy in HTF_STRATEGIES:
        requested = HTF_RESEARCH_TIMEFRAMES
    else:
        requested = LEGACY_RESEARCH_TIMEFRAMES
    if exclude_15min:
        requested = tuple(timeframe for timeframe in requested if timeframe != "15Min")
    if strategy == VOLATILITY_FOCUS_STRATEGY:
        unsupported_focus = [
            timeframe for timeframe in requested if timeframe not in VOLATILITY_FOCUS_SUPPORTED_TIMEFRAMES
        ]
        if unsupported_focus:
            raise ValueError(
                "Volatility focus supports only explicit 15Min plus 1H/4H timeframes: "
                + ", ".join(unsupported_focus)
            )
    unsupported = [timeframe for timeframe in requested if timeframe not in SUPPORTED_RESEARCH_TIMEFRAMES]
    if unsupported:
        raise ValueError(f"Unsupported research timeframe(s): {', '.join(unsupported)}")
    if not requested:
        raise ValueError("At least one research timeframe must be requested")
    return requested


def _allowed_raw_derivation_sources_for_timeframe(
    target_timeframe: str,
    requested_strategy_timeframes: tuple[str, ...] | list[str],
    *,
    volatility_focus: bool,
) -> tuple[str, ...] | None:
    if not volatility_focus:
        return None
    candidate_sources = DERIVATION_SOURCES_BY_TIMEFRAME.get(target_timeframe, ())
    requested = set(requested_strategy_timeframes)
    allowed = {source for source in candidate_sources if source != "15Min"}
    if target_timeframe == "1H" or "15Min" in requested:
        allowed.add("15Min")
    return tuple(source for source in candidate_sources if source in allowed)


def _market_data_timeframe(timeframe: str) -> str:
    return {
        "1H": "1Hour",
        "4H": "4Hour",
        "1D": "1Day",
    }.get(timeframe, timeframe)


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
    audit_mode: str = "standard",
    volatility_focus: bool = False,
    min_focused_trades: int = MIN_RESEARCH_TRADES,
    target_focused_trades: int = PREFERRED_RESEARCH_TRADES,
) -> list[dict[str, Any]]:
    if settings.symbol != ALLOWED_SYMBOL:
        raise ValueError("Higher-timeframe research is BTC/USD-only.")
    if audit_mode not in {"standard", "reality"}:
        raise ValueError("audit_mode must be standard or reality")
    volatility_focus = bool(volatility_focus or strategy == VOLATILITY_FOCUS_STRATEGY)
    if volatility_focus:
        strategy = VOLATILITY_FOCUS_STRATEGY
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
    baselines_by_timeframe = build_baselines_by_timeframe(bars_by_timeframe, settings)
    for config in generate_research_configs(
        strategy=strategy,
        max_buy_dip_configs=max_buy_dip_configs,
        max_v3_configs=max_v3_configs,
        timeframes=requested_timeframes,
    ):
        candidate_settings = research_config_settings(settings, config)
        bars = bars_by_timeframe.get(config.timeframe, pd.DataFrame())
        source_report = source_reports.get(config.timeframe)
        source_valid = source_report.research_result_valid if source_report is not None else True
        synthetic_data_used = source_report.synthetic_data_used if source_report is not None else False
        trades = pd.DataFrame()
        signal_frame = pd.DataFrame()
        filtered_hold: dict[str, Any] = {}
        if bars.empty or not source_valid:
            metrics = _empty_research_metrics("invalid_research_data_source" if not source_valid else "no_bars")
            walk_forward = empty_walk_forward_result(walk_forward_splits)
        elif config.strategy_name == HTF_RISK_OFF_HOLD_FILTER_STRATEGY:
            features = v3_features.get(config.timeframe, pd.DataFrame())
            filtered_hold = calculate_risk_off_hold_filter(features, config)
            metrics = {
                **_empty_research_metrics("risk_off_hold_filter_not_entry_strategy"),
                "net_return_pct": _metric_float(filtered_hold.get("filtered_hold_return_pct")),
                "gross_return_pct": _metric_float(filtered_hold.get("filtered_hold_return_pct")),
                "max_drawdown_pct": _metric_float(filtered_hold.get("filtered_hold_max_drawdown_pct")),
                "profit_factor_net": 0.0,
            }
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
        cost_summary = (
            evaluate_cost_scenarios_for_config(
                trades,
                signal_frame,
                settings,
                config,
                fallback_metrics=metrics,
            )
            if audit_mode == "reality" or volatility_focus
            else {}
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
        baseline_comparison = strategy_baseline_comparison(
            metrics,
            baselines_by_timeframe.get(config.timeframe, {}),
        )
        baseline_passed = bool(baseline_comparison.get("beats_any_relevant_baseline_risk_adjusted"))
        rejection_reasons = list(readiness["rejection_reasons"])
        economically_viable = bool(readiness["economically_viable"])
        paper_forward_eligible = bool(readiness["paper_forward_eligible"])
        research_promising = economically_viable
        if audit_mode == "reality":
            if not baseline_passed:
                rejection_reasons.append("does_not_beat_relevant_baseline_risk_adjusted")
                economically_viable = False
                paper_forward_eligible = False
                research_promising = False
            if config.strategy_name == HTF_RISK_OFF_HOLD_FILTER_STRATEGY:
                rejection_reasons.append("risk_off_hold_filter_not_trainable_entry_strategy")
                economically_viable = False
                paper_forward_eligible = False
                research_promising = False
        concentration = single_trade_return_concentration(metrics.get("trade_details", []))
        focused_gate: dict[str, Any] = {}
        maker_gate: dict[str, Any] = {}
        focused_diagnostics: dict[str, Any] = {}
        maker_diagnostics: dict[str, Any] = {}
        if volatility_focus:
            focused_gate = volatility_focus_research_gate(
                metrics,
                settings,
                config,
                cost_summary=cost_summary,
                source_report=source_report,
                synthetic_data_used=synthetic_data_used,
                research_result_valid=source_valid,
                baseline_comparison=baseline_comparison,
                walk_forward=walk_forward,
                concentration=concentration,
                active_model_valid=active_model_valid,
                min_focused_trades=min_focused_trades,
            )
            maker_gate = volatility_focus_maker_research_gate(
                metrics,
                settings,
                config,
                cost_summary=cost_summary,
                source_report=source_report,
                synthetic_data_used=synthetic_data_used,
                research_result_valid=source_valid,
                baseline_comparison=baseline_comparison,
                walk_forward=walk_forward,
                min_focused_trades=min_focused_trades,
            )
            maker_gate["maker_only_candidate"] = bool(maker_gate["maker_research_promising"]) and not bool(
                focused_gate["research_promising"]
            )
            economically_viable = bool(focused_gate["economically_viable"])
            research_promising = bool(focused_gate["research_promising"])
            paper_forward_eligible = bool(focused_gate["paper_forward_eligible"])
            rejection_reasons = list(focused_gate["paper_forward_rejection_reasons"])
            maker_diagnostics = volatility_focus_maker_execution_diagnostics(settings, cost_summary)
            focused_diagnostics = volatility_focus_trade_diagnostics(
                trades,
                signal_frame,
                metrics,
                config,
                walk_forward=walk_forward,
                cost_summary=cost_summary,
            )
            rank_details = volatility_focus_rank_details(
                metrics,
                focused_gate,
                concentration=concentration,
                walk_forward=walk_forward,
                cost_summary=cost_summary,
                baseline_comparison=baseline_comparison,
                target_focused_trades=target_focused_trades,
            )
        else:
            rank_details = research_rank_details(
                metrics,
                {
                    **readiness,
                    "economically_viable": economically_viable,
                    "paper_forward_eligible": paper_forward_eligible,
                },
                concentration=concentration,
                walk_forward=walk_forward,
            )
        rows.append(
            {
                "parameter_set_id": config.parameter_set_id,
                "track_id": config.track_id,
                "strategy_name": config.strategy_name,
                "timeframe": config.timeframe,
                "exit_mode": config.exit_mode,
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
                "require_ema_trend_filter": config.require_ema_trend_filter,
                "require_positive_ema20_slope": config.require_positive_ema20_slope,
                "require_close_above_ema200": config.require_close_above_ema200,
                "max_breakout_candle_atr_multiple": config.max_breakout_candle_atr_multiple,
                "min_close_position_in_candle": config.min_close_position_in_candle,
                "max_recent_runup_pct": config.max_recent_runup_pct,
                "min_consolidation_compression": config.min_consolidation_compression,
                "require_volume_expansion": config.require_volume_expansion,
                "max_atr_percentile": config.max_atr_percentile,
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
                "current_taker_net_return_pct": _metric_float(
                    (cost_summary.get("net_return_by_cost_scenario") or {}).get("current_taker")
                ),
                "maker_current_net_return_pct": _metric_float(
                    (cost_summary.get("net_return_by_cost_scenario") or {}).get("maker_current")
                ),
                "maker_low_slippage_net_return_pct": _metric_float(
                    (cost_summary.get("net_return_by_cost_scenario") or {}).get("maker_low_slippage")
                ),
                "zero_cost_net_return_pct": _metric_float(
                    (cost_summary.get("net_return_by_cost_scenario") or {}).get("zero_cost_sanity")
                ),
                "current_taker_profit_factor": _profit_factor_value(
                    (cost_summary.get("profit_factor_by_cost_scenario") or {}).get("current_taker")
                ),
                "maker_current_profit_factor": _profit_factor_value(
                    (cost_summary.get("profit_factor_by_cost_scenario") or {}).get("maker_current")
                ),
                "maker_low_slippage_profit_factor": _profit_factor_value(
                    (cost_summary.get("profit_factor_by_cost_scenario") or {}).get("maker_low_slippage")
                ),
                "zero_cost_profit_factor": _profit_factor_value(
                    (cost_summary.get("profit_factor_by_cost_scenario") or {}).get("zero_cost_sanity")
                ),
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
                "economically_viable": economically_viable,
                "research_promising": research_promising,
                "paper_forward_eligible": paper_forward_eligible,
                "rejection_reasons": ";".join(dict.fromkeys(rejection_reasons)),
                "research_rejection_reasons": ";".join(
                    dict.fromkeys(focused_gate.get("research_rejection_reasons", []))
                ) if volatility_focus else ";".join(
                    reason for reason in dict.fromkeys(rejection_reasons) if reason != "active_model_invalid"
                ),
                "paper_forward_rejection_reasons": ";".join(
                    dict.fromkeys(focused_gate.get("paper_forward_rejection_reasons", rejection_reasons))
                ),
                "training_rejection_reasons": ";".join(
                    dict.fromkeys(focused_gate.get("training_rejection_reasons", []))
                ),
                "training_eligible": bool(focused_gate.get("training_eligible", False)),
                "maker_research_promising": bool(maker_gate.get("maker_research_promising", False)),
                "maker_economically_viable": bool(maker_gate.get("maker_economically_viable", False)),
                "maker_only_candidate": bool(maker_gate.get("maker_only_candidate", False)),
                "maker_rejection_reasons": ";".join(
                    dict.fromkeys(maker_gate.get("maker_rejection_reasons", []))
                ),
                "volatility_focus": volatility_focus,
                "rank_score": rank_details["raw_rank_score"],
                **baseline_comparison,
                **cost_summary,
                **focused_diagnostics,
                **maker_diagnostics,
                **filtered_hold,
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
    if strategy == HTF_TREND_CONTINUATION_STRATEGY:
        return generate_htf_trend_continuation_configs(max_configs=max_v3_configs, timeframes=requested_timeframes)
    if strategy == HTF_VOLATILITY_EXPANSION_BREAKOUT_STRATEGY:
        return generate_htf_volatility_expansion_breakout_configs(
            max_configs=max_v3_configs,
            timeframes=requested_timeframes,
        )
    if strategy == HTF_RISK_OFF_HOLD_FILTER_STRATEGY:
        return generate_htf_risk_off_hold_filter_configs(max_configs=max_v3_configs, timeframes=requested_timeframes)
    if strategy == VOLATILITY_FOCUS_STRATEGY:
        return generate_volatility_focus_configs(max_configs=max_v3_configs, timeframes=requested_timeframes)
    legacy = (
        generate_trend_pullback_configs(timeframes=requested_timeframes)
        + generate_buy_the_dip_configs(max_configs=max_buy_dip_configs, timeframes=requested_timeframes)
    )
    v3 = generate_v3_configs(max_configs=max_v3_configs, timeframes=requested_timeframes)
    htf = generate_htf_configs(max_configs=max_v3_configs, timeframes=requested_timeframes)
    return legacy + v3 + htf


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


def generate_htf_configs(
    *,
    max_configs: int = DEFAULT_MAX_V3_CONFIGS,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> list[ResearchConfig]:
    configs = (
        generate_htf_trend_continuation_configs(max_configs=max_configs, timeframes=timeframes)
        + generate_htf_volatility_expansion_breakout_configs(max_configs=max_configs, timeframes=timeframes)
        + generate_htf_risk_off_hold_filter_configs(max_configs=max_configs, timeframes=timeframes)
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


def generate_htf_trend_continuation_configs(
    *,
    max_configs: int = DEFAULT_MAX_V3_CONFIGS,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> list[ResearchConfig]:
    profiles = (
        {
            "support": "ema20",
            "support_distance_pct": 0.012,
            "breakout_lookback": 24,
            "min_volume_zscore": -0.25,
            "min_trend_strength": 0.0,
            "max_atr_expansion": 2.0,
        },
        {
            "support": "ema50",
            "support_distance_pct": 0.018,
            "breakout_lookback": 32,
            "min_volume_zscore": 0.0,
            "min_trend_strength": 0.05,
            "max_atr_expansion": 2.3,
        },
        {
            "support": "vwap",
            "support_distance_pct": 0.015,
            "breakout_lookback": 48,
            "min_volume_zscore": 0.25,
            "min_trend_strength": 0.08,
            "max_atr_expansion": 2.5,
        },
    )
    raw_configs: list[ResearchConfig] = []
    requested_timeframes = tuple(
        timeframe for timeframe in (timeframes or HTF_RESEARCH_TIMEFRAMES) if timeframe in HTF_RESEARCH_TIMEFRAMES
    )
    for timeframe, take_profit_pct, stop_loss_pct, max_hold_bars, profile in product(
        requested_timeframes,
        HTF_TREND_TAKE_PROFIT_VALUES,
        HTF_TREND_STOP_LOSS_VALUES,
        HTF_MAX_HOLD_BARS_VALUES,
        profiles,
    ):
        raw_index = len(raw_configs)
        raw_configs.append(
            ResearchConfig(
                parameter_set_id=f"htft_{raw_index:05d}",
                strategy_name=HTF_TREND_CONTINUATION_STRATEGY,
                timeframe=timeframe,
                take_profit_pct=float(take_profit_pct),
                stop_loss_pct=float(stop_loss_pct),
                max_hold_bars=int(max_hold_bars),
                support=str(profile["support"]),
                support_distance_pct=float(profile["support_distance_pct"]),
                breakout_lookback=int(profile["breakout_lookback"]),
                min_volume_zscore=float(profile["min_volume_zscore"]),
                min_trend_strength=float(profile["min_trend_strength"]),
                max_atr_expansion=float(profile["max_atr_expansion"]),
            )
        )
    if len(raw_configs) <= max_configs:
        return raw_configs
    return [raw_configs[index] for index in _evenly_spaced_indexes(len(raw_configs), max_configs)]


def generate_htf_volatility_expansion_breakout_configs(
    *,
    max_configs: int = DEFAULT_MAX_V3_CONFIGS,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> list[ResearchConfig]:
    profiles = (
        {
            "breakout_lookback": 24,
            "consolidation_lookback": 16,
            "min_volume_zscore": 0.5,
            "min_body_vs_avg": 1.0,
            "min_recent_return_pct": 0.004,
            "min_trend_strength": 0.05,
            "max_atr_expansion": 2.4,
        },
        {
            "breakout_lookback": 32,
            "consolidation_lookback": 20,
            "min_volume_zscore": 0.75,
            "min_body_vs_avg": 1.15,
            "min_recent_return_pct": 0.006,
            "min_trend_strength": 0.08,
            "max_atr_expansion": 2.6,
        },
        {
            "breakout_lookback": 48,
            "consolidation_lookback": 24,
            "min_volume_zscore": 1.0,
            "min_body_vs_avg": 1.3,
            "min_recent_return_pct": 0.008,
            "min_trend_strength": 0.10,
            "max_atr_expansion": 2.8,
        },
    )
    raw_configs: list[ResearchConfig] = []
    requested_timeframes = tuple(
        timeframe for timeframe in (timeframes or HTF_RESEARCH_TIMEFRAMES) if timeframe in HTF_RESEARCH_TIMEFRAMES
    )
    for timeframe, take_profit_pct, stop_loss_pct, max_hold_bars, profile in product(
        requested_timeframes,
        HTF_BREAKOUT_TAKE_PROFIT_VALUES,
        HTF_BREAKOUT_STOP_LOSS_VALUES,
        HTF_MAX_HOLD_BARS_VALUES,
        profiles,
    ):
        raw_index = len(raw_configs)
        raw_configs.append(
            ResearchConfig(
                parameter_set_id=f"htfb_{raw_index:05d}",
                strategy_name=HTF_VOLATILITY_EXPANSION_BREAKOUT_STRATEGY,
                timeframe=timeframe,
                take_profit_pct=float(take_profit_pct),
                stop_loss_pct=float(stop_loss_pct),
                max_hold_bars=int(max_hold_bars),
                breakout_lookback=int(profile["breakout_lookback"]),
                consolidation_lookback=int(profile["consolidation_lookback"]),
                min_volume_zscore=float(profile["min_volume_zscore"]),
                min_body_vs_avg=float(profile["min_body_vs_avg"]),
                min_recent_return_pct=float(profile["min_recent_return_pct"]),
                min_trend_strength=float(profile["min_trend_strength"]),
                max_atr_expansion=float(profile["max_atr_expansion"]),
            )
        )
    if len(raw_configs) <= max_configs:
        return raw_configs
    return [raw_configs[index] for index in _evenly_spaced_indexes(len(raw_configs), max_configs)]


def generate_volatility_focus_configs(
    *,
    max_configs: int = DEFAULT_MAX_V3_CONFIGS,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> list[ResearchConfig]:
    requested_timeframes = tuple(
        timeframe
        for timeframe in (timeframes or VOLATILITY_FOCUS_DEFAULT_TIMEFRAMES)
        if timeframe in VOLATILITY_FOCUS_SUPPORTED_TIMEFRAMES
    )
    base_specs: list[dict[str, Any]] = []

    def add_spec(spec: dict[str, Any]) -> None:
        key = tuple((name, _normalise_spec_value(spec.get(name))) for name in sorted(spec))
        if key in seen_specs:
            return
        seen_specs.add(key)
        base_specs.append(spec)

    seen_specs: set[tuple[tuple[str, Any], ...]] = set()
    for timeframe in requested_timeframes:
        add_spec(
            {
                "strategy_name": HTF_VOLATILITY_EXPANSION_BREAKOUT_STRATEGY,
                "timeframe": timeframe,
                "take_profit_pct": 0.04,
                "stop_loss_pct": 0.02,
                "max_hold_bars": 96,
                "breakout_lookback": 24,
                "consolidation_lookback": 16,
                "min_body_vs_avg": 1.0,
                "min_recent_return_pct": 0.004,
                "min_trend_strength": 0.05,
                "min_volume_zscore": 0.5,
                "max_atr_expansion": 2.4,
            }
        )
        add_spec(
            {
                "strategy_name": VOLATILITY_BREAKOUT_STRATEGY,
                "timeframe": timeframe,
                "take_profit_pct": 0.04,
                "stop_loss_pct": 0.02,
                "max_hold_bars": 48,
                "breakout_lookback": 20,
                "consolidation_lookback": 12,
                "min_body_vs_avg": 1.0,
                "min_recent_return_pct": 0.002,
                "min_trend_strength": 0.0,
                "min_volume_zscore": 0.5,
                "max_atr_expansion": 2.6,
            }
        )

    signal_profiles = volatility_focus_signal_profiles()
    quality_profiles = volatility_focus_quality_filter_profiles()
    for (
        strategy_name,
        timeframe,
        take_profit_pct,
        stop_loss_pct,
        max_hold_bars,
        signal_profile,
        quality_profile,
    ) in product(
        VOLATILITY_FOCUS_STRATEGIES,
        requested_timeframes,
        VOLATILITY_FOCUS_TAKE_PROFIT_VALUES,
        VOLATILITY_FOCUS_STOP_LOSS_VALUES,
        VOLATILITY_FOCUS_MAX_HOLD_BARS_VALUES,
        signal_profiles,
        quality_profiles,
    ):
        add_spec(
            {
                "strategy_name": strategy_name,
                "timeframe": timeframe,
                "take_profit_pct": float(take_profit_pct),
                "stop_loss_pct": float(stop_loss_pct),
                "max_hold_bars": int(max_hold_bars),
                **signal_profile,
                **quality_profile,
            }
        )

    if _volatility_focus_v9_enabled(max_configs, requested_timeframes):
        raw_specs = generate_volatility_focus_v9_targeted_specs(max_specs=int(max_configs))
    elif _volatility_focus_v8_enabled(max_configs, requested_timeframes):
        raw_specs = generate_volatility_focus_v8_targeted_specs(max_specs=int(max_configs))
    elif _volatility_focus_v7_enabled(max_configs, requested_timeframes):
        base_budget = min(VOLATILITY_FOCUS_V7_BASE_CONFIGS, max(0, int(max_configs) // 4))
        targeted_budget = max(0, int(max_configs) - base_budget)
        raw_specs = generate_volatility_focus_v7_targeted_specs(max_specs=targeted_budget)
        raw_specs.extend(_sample_volatility_focus_specs(base_specs, base_budget, anchor_count=2))
    else:
        raw_specs = _sample_volatility_focus_specs(
            base_specs,
            int(max_configs),
            anchor_count=2 * max(1, len(requested_timeframes)),
        )

    configs: list[ResearchConfig] = []
    for index, spec in enumerate(raw_specs[: int(max_configs)]):
        strategy_name = str(spec["strategy_name"])
        prefix = "vff_vbo" if strategy_name == VOLATILITY_BREAKOUT_STRATEGY else "vff_htfb"
        configs.append(_research_config_from_volatility_focus_spec(spec, index=index, prefix=prefix))
    return configs


def _volatility_focus_v7_enabled(max_configs: int, requested_timeframes: tuple[str, ...]) -> bool:
    return int(max_configs) >= VOLATILITY_FOCUS_V7_MIN_CONFIGS and requested_timeframes == ("1H",)


def _volatility_focus_v9_enabled(max_configs: int, requested_timeframes: tuple[str, ...]) -> bool:
    return int(max_configs) == VOLATILITY_FOCUS_V9_CONFIGS and requested_timeframes == ("1H",)


def _volatility_focus_v8_enabled(max_configs: int, requested_timeframes: tuple[str, ...]) -> bool:
    return int(max_configs) >= VOLATILITY_FOCUS_V8_MIN_CONFIGS and requested_timeframes == ("1H",)


def _sample_volatility_focus_specs(
    specs: list[dict[str, Any]],
    desired: int,
    *,
    anchor_count: int,
) -> list[dict[str, Any]]:
    if desired <= 0 or not specs:
        return []
    if len(specs) <= desired:
        return list(specs)
    anchors = specs[: min(anchor_count, desired)]
    remaining = specs[len(anchors) :]
    desired_remaining = max(0, int(desired) - len(anchors))
    sampled_remaining = (
        [remaining[index] for index in _evenly_spaced_indexes(len(remaining), desired_remaining)]
        if desired_remaining and remaining
        else []
    )
    return anchors + sampled_remaining


def _research_config_from_volatility_focus_spec(
    spec: dict[str, Any],
    *,
    index: int,
    prefix: str,
) -> ResearchConfig:
    return ResearchConfig(
        parameter_set_id=str(spec.get("parameter_set_id") or f"{prefix}_{index:05d}"),
        strategy_name=str(spec["strategy_name"]),
        timeframe=str(spec["timeframe"]),
        take_profit_pct=float(spec["take_profit_pct"]),
        stop_loss_pct=float(spec["stop_loss_pct"]),
        max_hold_bars=int(spec["max_hold_bars"]),
        breakout_lookback=int(spec["breakout_lookback"]),
        consolidation_lookback=int(spec["consolidation_lookback"]),
        min_volume_zscore=float(spec["min_volume_zscore"]),
        min_body_vs_avg=float(spec["min_body_vs_avg"]),
        min_recent_return_pct=float(spec["min_recent_return_pct"]),
        min_trend_strength=float(spec["min_trend_strength"]),
        max_atr_expansion=float(spec["max_atr_expansion"]),
        require_ema_trend_filter=bool(spec.get("require_ema_trend_filter", False)),
        require_positive_ema20_slope=bool(spec.get("require_positive_ema20_slope", False)),
        require_close_above_ema200=bool(spec.get("require_close_above_ema200", False)),
        max_breakout_candle_atr_multiple=(
            None
            if spec.get("max_breakout_candle_atr_multiple") is None
            else float(spec["max_breakout_candle_atr_multiple"])
        ),
        min_close_position_in_candle=(
            None if spec.get("min_close_position_in_candle") is None else float(spec["min_close_position_in_candle"])
        ),
        max_recent_runup_pct=(None if spec.get("max_recent_runup_pct") is None else float(spec["max_recent_runup_pct"])),
        min_consolidation_compression=(
            None
            if spec.get("min_consolidation_compression") is None
            else float(spec["min_consolidation_compression"])
        ),
        require_volume_expansion=bool(spec.get("require_volume_expansion", False)),
        max_atr_percentile=(None if spec.get("max_atr_percentile") is None else float(spec["max_atr_percentile"])),
        track_id=spec.get("track_id"),
        exit_mode=str(spec.get("exit_mode") or EXIT_MODE_FIXED),
    )


def generate_volatility_focus_v7_targeted_specs(*, max_specs: int) -> list[dict[str, Any]]:
    if max_specs <= 0:
        return []
    track_a_budget = int(max_specs) // 2
    track_a = _sample_volatility_focus_v7_track_specs(
        track_id=VOLATILITY_FOCUS_TRACK_A,
        desired=track_a_budget,
        parameter_prefix="v7a",
        take_profit_values=(0.06, 0.05, 0.055, 0.065, 0.07),
        stop_loss_values=(0.02, 0.018, 0.022, 0.025),
        max_hold_values=(96, 72, 120),
        breakout_values=(20, 16, 24),
        consolidation_values=(16, 12, 20),
        body_values=(1.2, 1.0, 1.1),
        recent_values=(0.004, 0.002, 0.003, 0.005),
        trend_values=(0.05, 0.0, 0.03),
        atr_values=(2.2, 2.6, 3.0),
        volume_values=(0.5, 0.0, 0.25),
    )
    track_b = _sample_volatility_focus_v7_track_specs(
        track_id=VOLATILITY_FOCUS_TRACK_B,
        desired=max(0, int(max_specs) - len(track_a)),
        parameter_prefix="v7b",
        take_profit_values=(0.04, 0.045, 0.05, 0.055, 0.06),
        stop_loss_values=(0.02, 0.018, 0.022, 0.025),
        max_hold_values=(48, 36, 60, 72),
        breakout_values=(20, 16, 24),
        consolidation_values=(12, 8, 16),
        body_values=(1.0, 0.8, 1.2),
        recent_values=(0.002, 0.001, 0.003),
        trend_values=(0.0, 0.03),
        atr_values=(2.6, 2.2, 3.0, 3.5),
        volume_values=(0.5, 0.0, 0.25),
    )
    return track_a + track_b


def generate_volatility_focus_v8_targeted_specs(*, max_specs: int) -> list[dict[str, Any]]:
    if max_specs <= 0:
        return []
    track_m_budget = int(max_specs) // 2
    track_m = _sample_volatility_focus_v7_track_specs(
        track_id=VOLATILITY_FOCUS_TRACK_M,
        desired=track_m_budget,
        parameter_prefix="v8m",
        exit_modes=VOLATILITY_FOCUS_V8_MAKER_EXIT_MODES,
        quality_profiles=(_volatility_focus_v7_quality_profiles()[0],),
        take_profit_values=(0.045, 0.05, 0.055, 0.06, 0.065),
        stop_loss_values=(0.02, 0.018, 0.022, 0.025),
        max_hold_values=(48, 60, 72, 96),
        breakout_values=(20, 16, 24),
        consolidation_values=(12, 8, 16),
        body_values=(0.8, 1.0, 1.2),
        recent_values=(0.003, 0.002, 0.004),
        trend_values=(0.0, 0.03, 0.05),
        atr_values=(2.2, 2.6, 3.0),
        volume_values=(0.25, 0.0, 0.5),
    )
    track_t = _sample_volatility_focus_v7_track_specs(
        track_id=VOLATILITY_FOCUS_TRACK_T,
        desired=max(0, int(max_specs) - len(track_m)),
        parameter_prefix="v8t",
        exit_modes=VOLATILITY_FOCUS_V8_TAKER_EXIT_MODES,
        quality_profiles=(_volatility_focus_v7_quality_profiles()[0],),
        take_profit_values=(0.07, 0.08, 0.09, 0.10, 0.12),
        stop_loss_values=(0.018, 0.02, 0.025, 0.03),
        max_hold_values=(96, 120, 144, 168),
        breakout_values=(16, 20, 24, 36),
        consolidation_values=(12, 16, 20, 24),
        body_values=(0.8, 1.0, 1.2),
        recent_values=(0.002, 0.003, 0.004, 0.006),
        trend_values=(0.0, 0.03, 0.05),
        atr_values=(2.2, 2.6, 3.0, 3.5),
        volume_values=(0.0, 0.25, 0.5),
    )
    return track_m + track_t


def generate_volatility_focus_v9_targeted_specs(*, max_specs: int) -> list[dict[str, Any]]:
    if max_specs <= 0:
        return []
    dimensions: tuple[tuple[str, tuple[Any, ...]], ...] = (
        ("exit_mode", VOLATILITY_FOCUS_V9_SEARCH_SPACE["exit_mode"]),
        ("take_profit_pct", VOLATILITY_FOCUS_V9_SEARCH_SPACE["take_profit_pct"]),
        ("stop_loss_pct", VOLATILITY_FOCUS_V9_SEARCH_SPACE["stop_loss_pct"]),
        ("max_hold_bars", VOLATILITY_FOCUS_V9_SEARCH_SPACE["max_hold_bars"]),
        ("breakout_lookback", VOLATILITY_FOCUS_V9_SEARCH_SPACE["breakout_lookback"]),
        ("consolidation_lookback", VOLATILITY_FOCUS_V9_SEARCH_SPACE["consolidation_lookback"]),
        ("min_body_vs_avg", VOLATILITY_FOCUS_V9_SEARCH_SPACE["min_body_vs_avg"]),
        ("min_recent_return_pct", VOLATILITY_FOCUS_V9_SEARCH_SPACE["min_recent_return_pct"]),
        ("min_trend_strength", VOLATILITY_FOCUS_V9_SEARCH_SPACE["min_trend_strength"]),
        ("max_atr_expansion", VOLATILITY_FOCUS_V9_SEARCH_SPACE["max_atr_expansion"]),
        ("min_volume_zscore", VOLATILITY_FOCUS_V9_SEARCH_SPACE["min_volume_zscore"]),
    )
    quality_profile = _volatility_focus_v7_quality_profiles()[0]
    selected: list[dict[str, Any]] = []
    for distance in range(len(dimensions) + 1):
        level_specs: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for changed_indexes in combinations(range(len(dimensions)), distance):
            changed = set(changed_indexes)
            option_groups: list[tuple[tuple[Any, ...], ...]] = []
            for index, (name, values) in enumerate(dimensions):
                anchor_value = VOLATILITY_FOCUS_V9_ANCHOR_SPEC[name]
                options = _volatility_focus_v9_options_by_anchor_distance(
                    values,
                    anchor_value=anchor_value,
                    changed=index in changed,
                )
                option_groups.append(tuple((name, value) for value in options))
            for values in product(*option_groups):
                spec_values = {name: value for name, value in values}
                spec = {
                    "track_id": VOLATILITY_FOCUS_TRACK_M9,
                    "strategy_name": VOLATILITY_BREAKOUT_STRATEGY,
                    "timeframe": "1H",
                    **spec_values,
                    **quality_profile,
                }
                level_specs.append((_volatility_focus_v9_anchor_sort_key(spec), spec))
        for _, spec in sorted(level_specs, key=lambda item: item[0]):
            selected.append(spec)
            if len(selected) >= int(max_specs):
                return [
                    {
                        **selected_spec,
                        "parameter_set_id": f"v9m_{index:05d}",
                    }
                    for index, selected_spec in enumerate(selected)
                ]
    return [
        {
            **selected_spec,
            "parameter_set_id": f"v9m_{index:05d}",
        }
        for index, selected_spec in enumerate(selected[: int(max_specs)])
    ]


def _volatility_focus_v9_options_by_anchor_distance(
    values: tuple[Any, ...],
    *,
    anchor_value: Any,
    changed: bool,
) -> tuple[Any, ...]:
    if changed:
        options = [value for value in values if _normalise_spec_value(value) != _normalise_spec_value(anchor_value)]
    else:
        options = [anchor_value]
    return tuple(sorted(options, key=lambda value: _volatility_focus_v9_value_distance_key(value, anchor_value)))


def _volatility_focus_v9_anchor_sort_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        _volatility_focus_v9_value_distance_key(spec[name], VOLATILITY_FOCUS_V9_ANCHOR_SPEC[name])
        for name in VOLATILITY_FOCUS_V9_ANCHOR_SPEC
    )


def _volatility_focus_v9_value_distance_key(value: Any, anchor_value: Any) -> tuple[Any, ...]:
    if isinstance(value, (int, float)) and isinstance(anchor_value, (int, float)):
        return (abs(float(value) - float(anchor_value)), float(value))
    return (
        0 if _normalise_spec_value(value) == _normalise_spec_value(anchor_value) else 1,
        str(value),
    )


def _sample_volatility_focus_v7_track_specs(
    *,
    track_id: str,
    desired: int,
    parameter_prefix: str,
    exit_modes: tuple[str, ...] = VOLATILITY_FOCUS_V7_EXIT_MODES,
    quality_profiles: tuple[dict[str, Any], ...] | None = None,
    take_profit_values: tuple[float, ...],
    stop_loss_values: tuple[float, ...],
    max_hold_values: tuple[int, ...],
    breakout_values: tuple[int, ...],
    consolidation_values: tuple[int, ...],
    body_values: tuple[float, ...],
    recent_values: tuple[float, ...],
    trend_values: tuple[float, ...],
    atr_values: tuple[float, ...],
    volume_values: tuple[float, ...],
) -> list[dict[str, Any]]:
    if desired <= 0:
        return []
    quality_profiles = quality_profiles or _volatility_focus_v7_quality_profiles()
    dimensions = (
        exit_modes,
        take_profit_values,
        stop_loss_values,
        max_hold_values,
        breakout_values,
        consolidation_values,
        body_values,
        recent_values,
        trend_values,
        atr_values,
        volume_values,
        quality_profiles,
    )
    total = math.prod(len(dimension) for dimension in dimensions)
    sampled_indexes = _evenly_spaced_indexes(total, max(1, int(desired) - len(exit_modes)))
    specs: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    anchor_values = (
        take_profit_values[0],
        stop_loss_values[0],
        max_hold_values[0],
        breakout_values[0],
        consolidation_values[0],
        body_values[0],
        recent_values[0],
        trend_values[0],
        atr_values[0],
        volume_values[0],
        quality_profiles[0],
    )
    for exit_mode in exit_modes:
        _append_volatility_focus_v7_track_spec(
            specs,
            seen,
            track_id=track_id,
            values=(exit_mode, *anchor_values),
        )
    for index in sampled_indexes:
        _append_volatility_focus_v7_track_spec(
            specs,
            seen,
            track_id=track_id,
            values=_product_values_at_index(dimensions, index),
        )
        if len(specs) >= desired:
            break
    fill_index = 0
    while len(specs) < desired and fill_index < total:
        _append_volatility_focus_v7_track_spec(
            specs,
            seen,
            track_id=track_id,
            values=_product_values_at_index(dimensions, fill_index),
        )
        fill_index += 1
    return [
        {
            **spec,
            "parameter_set_id": f"{parameter_prefix}_{index:05d}",
        }
        for index, spec in enumerate(specs[:desired])
    ]


def _volatility_focus_v7_quality_profiles() -> tuple[dict[str, Any], ...]:
    disabled = {
        "require_ema_trend_filter": False,
        "require_positive_ema20_slope": False,
        "require_close_above_ema200": False,
        "max_breakout_candle_atr_multiple": None,
        "min_close_position_in_candle": None,
        "max_recent_runup_pct": None,
        "min_consolidation_compression": None,
        "require_volume_expansion": False,
        "max_atr_percentile": None,
    }
    light = {
        **disabled,
        "require_close_above_ema200": True,
        "max_breakout_candle_atr_multiple": 3.0,
        "min_close_position_in_candle": 0.50,
        "max_recent_runup_pct": 0.10,
        "min_consolidation_compression": 0.80,
        "max_atr_percentile": 0.97,
    }
    return disabled, light


def _append_volatility_focus_v7_track_spec(
    specs: list[dict[str, Any]],
    seen: set[tuple[tuple[str, Any], ...]],
    *,
    track_id: str,
    values: tuple[Any, ...],
) -> None:
    (
        exit_mode,
        take_profit_pct,
        stop_loss_pct,
        max_hold_bars,
        breakout_lookback,
        consolidation_lookback,
        min_body_vs_avg,
        min_recent_return_pct,
        min_trend_strength,
        max_atr_expansion,
        min_volume_zscore,
        quality_profile,
    ) = values
    spec = {
        "track_id": track_id,
        "exit_mode": exit_mode,
        "strategy_name": VOLATILITY_BREAKOUT_STRATEGY,
        "timeframe": "1H",
        "take_profit_pct": float(take_profit_pct),
        "stop_loss_pct": float(stop_loss_pct),
        "max_hold_bars": int(max_hold_bars),
        "breakout_lookback": int(breakout_lookback),
        "consolidation_lookback": int(consolidation_lookback),
        "min_body_vs_avg": float(min_body_vs_avg),
        "min_recent_return_pct": float(min_recent_return_pct),
        "min_trend_strength": float(min_trend_strength),
        "max_atr_expansion": float(max_atr_expansion),
        "min_volume_zscore": float(min_volume_zscore),
        **quality_profile,
    }
    key = tuple((name, _normalise_spec_value(spec.get(name))) for name in sorted(spec))
    if key not in seen:
        seen.add(key)
        specs.append(spec)


def _product_values_at_index(dimensions: tuple[tuple[Any, ...], ...], index: int) -> tuple[Any, ...]:
    values: list[Any] = []
    remaining = int(index)
    for dimension in reversed(dimensions):
        values.append(dimension[remaining % len(dimension)])
        remaining //= len(dimension)
    return tuple(reversed(values))


def volatility_focus_signal_profiles() -> tuple[dict[str, Any], ...]:
    profiles = [
        (20, 12, 1.0, 0.002, 0.00, 0.50, 2.6),
        (24, 16, 1.0, 0.004, 0.05, 0.50, 2.4),
        (12, 8, 0.8, 0.001, 0.00, -0.25, 3.5),
        (16, 8, 0.8, 0.002, 0.02, 0.00, 3.0),
        (16, 12, 1.0, 0.003, 0.02, 0.25, 2.6),
        (20, 16, 1.2, 0.004, 0.05, 0.50, 2.2),
        (24, 20, 1.2, 0.006, 0.08, 0.75, 2.2),
        (36, 20, 1.5, 0.006, 0.08, 0.75, 1.8),
        (36, 24, 1.2, 0.008, 0.10, 1.00, 2.6),
        (48, 24, 1.5, 0.008, 0.10, 1.00, 3.0),
        (48, 16, 1.0, 0.003, 0.05, 0.25, 3.5),
        (12, 24, 0.8, 0.001, 0.00, -0.25, 1.8),
    ]
    return tuple(
        {
            "breakout_lookback": int(breakout_lookback),
            "consolidation_lookback": int(consolidation_lookback),
            "min_body_vs_avg": float(min_body_vs_avg),
            "min_recent_return_pct": float(min_recent_return_pct),
            "min_trend_strength": float(min_trend_strength),
            "min_volume_zscore": float(min_volume_zscore),
            "max_atr_expansion": float(max_atr_expansion),
        }
        for (
            breakout_lookback,
            consolidation_lookback,
            min_body_vs_avg,
            min_recent_return_pct,
            min_trend_strength,
            min_volume_zscore,
            max_atr_expansion,
        ) in profiles
    )


def volatility_focus_quality_filter_profiles() -> tuple[dict[str, Any], ...]:
    return (
        {
            "require_ema_trend_filter": False,
            "require_positive_ema20_slope": False,
            "require_close_above_ema200": False,
            "max_breakout_candle_atr_multiple": None,
            "min_close_position_in_candle": None,
            "max_recent_runup_pct": None,
            "min_consolidation_compression": None,
            "require_volume_expansion": False,
            "max_atr_percentile": None,
        },
        {
            "require_ema_trend_filter": False,
            "require_positive_ema20_slope": False,
            "require_close_above_ema200": True,
            "max_breakout_candle_atr_multiple": 3.0,
            "min_close_position_in_candle": 0.50,
            "max_recent_runup_pct": 0.10,
            "min_consolidation_compression": 0.80,
            "require_volume_expansion": False,
            "max_atr_percentile": 0.97,
        },
        {
            "require_ema_trend_filter": True,
            "require_positive_ema20_slope": True,
            "require_close_above_ema200": True,
            "max_breakout_candle_atr_multiple": 2.5,
            "min_close_position_in_candle": 0.60,
            "max_recent_runup_pct": 0.08,
            "min_consolidation_compression": 1.00,
            "require_volume_expansion": True,
            "max_atr_percentile": 0.95,
        },
        {
            "require_ema_trend_filter": True,
            "require_positive_ema20_slope": True,
            "require_close_above_ema200": True,
            "max_breakout_candle_atr_multiple": 2.2,
            "min_close_position_in_candle": 0.65,
            "max_recent_runup_pct": 0.06,
            "min_consolidation_compression": 1.10,
            "require_volume_expansion": True,
            "max_atr_percentile": 0.92,
        },
        {
            "require_ema_trend_filter": True,
            "require_positive_ema20_slope": True,
            "require_close_above_ema200": True,
            "max_breakout_candle_atr_multiple": 1.8,
            "min_close_position_in_candle": 0.70,
            "max_recent_runup_pct": 0.04,
            "min_consolidation_compression": 1.20,
            "require_volume_expansion": True,
            "max_atr_percentile": 0.90,
        },
    )


def _normalise_spec_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    return value


def generate_htf_risk_off_hold_filter_configs(
    *,
    max_configs: int = DEFAULT_MAX_V3_CONFIGS,
    timeframes: tuple[str, ...] | list[str] | None = None,
) -> list[ResearchConfig]:
    raw_configs: list[ResearchConfig] = []
    requested_timeframes = tuple(
        timeframe for timeframe in (timeframes or HTF_RESEARCH_TIMEFRAMES) if timeframe in HTF_RESEARCH_TIMEFRAMES
    )
    for timeframe, profile in product(requested_timeframes, RISK_OFF_FILTER_PROFILES):
        raw_index = len(raw_configs)
        raw_configs.append(
            ResearchConfig(
                parameter_set_id=f"htfr_{raw_index:05d}",
                strategy_name=HTF_RISK_OFF_HOLD_FILTER_STRATEGY,
                timeframe=timeframe,
                take_profit_pct=0.0,
                stop_loss_pct=0.0,
                max_hold_bars=1,
                drawdown_threshold=float(profile["drawdown_threshold"]),
                max_atr_expansion=float(profile["max_atr_expansion"]),
                min_recent_return_pct=float(profile["min_recent_return_pct"]),
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
                **_trade_audit_entry_exit_fields(row, exit_result),
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
    atr_percentile_200 = features["atr_14"].rolling(200, min_periods=20).apply(
        lambda values: float(np.mean(values <= values[-1])) if len(values) else np.nan,
        raw=True,
    )
    rolling_high_20 = close.rolling(20).max()
    rolling_high_50 = close.rolling(50).max()
    rolling_high_200 = close.rolling(200).max()
    rolling_low_20 = close.rolling(20).min()
    atr_mean_20 = features["atr_14"].rolling(20).mean()
    typical_price = (features["high"] + features["low"] + features["close"]) / 3
    volume = features["volume"].replace(0, np.nan)
    rolling_vwap_50 = (typical_price * volume).rolling(50).sum() / volume.rolling(50).sum()
    features["ema_200"] = ema_200
    features["ema_20_slope_5"] = ema_20.pct_change(5)
    features["ema_50_slope_5"] = ema_50.pct_change(5)
    features["ema_200_distance"] = (close - ema_200) / close.replace(0, np.nan)
    features["ema_20_above_50"] = ema_20 > ema_50
    features["ema_50_above_200"] = ema_50 > ema_200
    features["close_above_ema_200"] = close > ema_200
    features["recent_high_50"] = rolling_high_50
    features["pullback_from_high_50"] = (rolling_high_50 - close) / rolling_high_50.replace(0, np.nan)
    features["drawdown_from_high_200"] = (rolling_high_200 - close) / rolling_high_200.replace(0, np.nan)
    features["support_distance_ema20_abs"] = ((close - ema_20) / close.replace(0, np.nan)).abs()
    features["support_distance_ema50_abs"] = ((close - ema_50) / close.replace(0, np.nan)).abs()
    features["support_distance_vwap_abs"] = ((close - features["vwap"]) / close.replace(0, np.nan)).abs()
    features["rolling_vwap_50"] = rolling_vwap_50
    features["close_above_rolling_vwap_50"] = close > rolling_vwap_50
    features["lower_wick_ratio"] = (lower_body - features["low"]) / candle_range
    features["upper_wick_ratio"] = (features["high"] - upper_body) / candle_range
    features["close_position_in_candle"] = (features["close"] - features["low"]) / candle_range
    features["bullish_close"] = features["close"] > features["open"]
    features["recovers_prior_high"] = features["close"] > features["prev_high"]
    features["body_pct"] = body_pct
    features["body_vs_avg_20"] = body_pct / body_mean_20.replace(0, np.nan)
    features["breakout_candle_atr_multiple"] = features["high_low_range_pct"] / features["atr_14"].replace(0, np.nan)
    features["recent_runup_pct_5"] = close / close.rolling(5).min().replace(0, np.nan) - 1
    features["atr_percentile_200"] = atr_percentile_200
    features["prior_rolling_high_12"] = close.rolling(12).max().shift(1)
    features["prior_rolling_high_16"] = close.rolling(16).max().shift(1)
    features["prior_rolling_high_20"] = rolling_high_20.shift(1)
    features["prior_rolling_high_24"] = close.rolling(24).max().shift(1)
    features["prior_rolling_high_32"] = close.rolling(32).max().shift(1)
    features["prior_rolling_high_36"] = close.rolling(36).max().shift(1)
    features["prior_rolling_high_48"] = close.rolling(48).max().shift(1)
    features["prior_rolling_high_64"] = close.rolling(64).max().shift(1)
    features["prior_rolling_high_96"] = close.rolling(96).max().shift(1)
    features["rolling_low_20"] = rolling_low_20
    features["range_width_8"] = (close.rolling(8).max() - close.rolling(8).min()) / close.replace(0, np.nan)
    features["range_width_12"] = (close.rolling(12).max() - close.rolling(12).min()) / close.replace(0, np.nan)
    features["range_width_16"] = (close.rolling(16).max() - close.rolling(16).min()) / close.replace(0, np.nan)
    features["range_width_20"] = (close.rolling(20).max() - close.rolling(20).min()) / close.replace(0, np.nan)
    features["range_width_24"] = (close.rolling(24).max() - close.rolling(24).min()) / close.replace(0, np.nan)
    features["range_width_32"] = (close.rolling(32).max() - close.rolling(32).min()) / close.replace(0, np.nan)
    features["range_width_48"] = (close.rolling(48).max() - close.rolling(48).min()) / close.replace(0, np.nan)
    for lookback in (8, 12, 16, 20, 24):
        features[f"range_compression_{lookback}"] = features["range_width_48"] / features[
            f"range_width_{lookback}"
        ].replace(0, np.nan)
    features["atr_expansion_20"] = features["atr_14"] / atr_mean_20.replace(0, np.nan)
    features["atr_downside_explosion"] = (features["log_return_3"] < -0.025) & (features["atr_expansion_20"] > 2.0)
    features["extreme_crash_candle"] = (
        (features["close_open_pct"] <= -0.03)
        | (features["high_low_range_pct"] >= 0.08)
        | (features["log_return_5"] <= -0.05)
    )
    features["large_negative_candle"] = features["close_open_pct"] <= -0.025
    features["downside_volatility_cluster"] = (
        (features["log_return_1"] < 0).rolling(5).sum() >= 4
    ) & (features["atr_expansion_20"] >= 1.5)
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
    if config.strategy_name == HTF_TREND_CONTINUATION_STRATEGY:
        return build_htf_trend_continuation_research_trades(
            v3_features if v3_features is not None else prepare_v3_features(bars),
            settings,
            config,
        )
    if config.strategy_name == HTF_VOLATILITY_EXPANSION_BREAKOUT_STRATEGY:
        return build_htf_volatility_expansion_breakout_research_trades(
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
    required = list(dict.fromkeys(required + volatility_breakout_quality_required_columns(config)))
    if any(column not in features for column in required):
        return pd.DataFrame(), pd.DataFrame()
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
    mask = apply_volatility_breakout_quality_filters(data, mask, config)
    return _build_research_trades_from_mask(
        data,
        mask,
        config,
        strategy_name=VOLATILITY_BREAKOUT_STRATEGY,
        entry_reason="volatility_breakout_momentum_continuation_candidate",
        regime="higher_timeframe_momentum_breakout",
    )


def build_htf_trend_continuation_research_trades(
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
    breakout_lookback = int(config.breakout_lookback or 24)
    high_column = f"prior_rolling_high_{breakout_lookback}"
    if support_column is None or high_column not in features:
        return pd.DataFrame(), pd.DataFrame()
    required = [
        "timestamp",
        "close",
        "open",
        "high",
        "low",
        "ema_20",
        "ema_50",
        "ema_200",
        "ema_20_slope_5",
        "close_above_ema_200",
        "ema_50_above_200",
        "close_above_rolling_vwap_50",
        "volume_zscore_20",
        "atr_expansion_20",
        "trend_strength_20",
        "extreme_crash_candle",
        support_column,
        high_column,
    ]
    data = features.dropna(subset=required).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    clear_uptrend = (
        data["close_above_ema_200"].astype(bool)
        & data["ema_50_above_200"].astype(bool)
        & (data["ema_20_slope_5"] > 0)
        & data["close_above_rolling_vwap_50"].astype(bool)
    )
    pullback_or_continuation = (
        data[support_column] <= float(config.support_distance_pct or 0.015)
    ) | (data["close"] > data[high_column])
    risk_ok = (
        ~data["extreme_crash_candle"].astype(bool)
        & (data["atr_expansion_20"] <= float(config.max_atr_expansion or 2.4))
        & (data["volume_zscore_20"] >= float(config.min_volume_zscore or -0.25))
        & (data["trend_strength_20"].abs() >= float(config.min_trend_strength or 0.0))
    )
    mask = clear_uptrend & pullback_or_continuation & risk_ok
    return _build_research_trades_from_mask(
        data,
        mask,
        config,
        strategy_name=HTF_TREND_CONTINUATION_STRATEGY,
        entry_reason="htf_trend_continuation_pullback_or_breakout",
        regime="higher_timeframe_clear_uptrend",
    )


def build_htf_volatility_expansion_breakout_research_trades(
    features: pd.DataFrame,
    settings: Settings,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    breakout_lookback = int(config.breakout_lookback or 32)
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
        "ema_50_above_200",
        "close_above_ema_200",
        "log_return_3",
        "log_return_5",
        "volume_zscore_20",
        "atr_expansion_20",
        "body_vs_avg_20",
        "trend_strength_20",
        "extreme_crash_candle",
        high_column,
        range_column,
    ]
    required = list(dict.fromkeys(required + volatility_breakout_quality_required_columns(config)))
    if any(column not in features for column in required):
        return pd.DataFrame(), pd.DataFrame()
    data = features.dropna(subset=required).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    breakout = data["close"] > data[high_column]
    trend_support = (
        data["close_above_ema_200"].astype(bool)
        & data["ema_50_above_200"].astype(bool)
        & (data["ema_20"] > data["ema_50"])
        & (data["ema_50_slope_5"] > 0)
    )
    expansion = (
        (data["volume_zscore_20"] >= float(config.min_volume_zscore or 0.75))
        & (data["atr_expansion_20"] >= 1.0)
        & (data["atr_expansion_20"] <= float(config.max_atr_expansion or 2.6))
        & (data["body_vs_avg_20"] >= float(config.min_body_vs_avg or 1.1))
        & (data["log_return_3"] >= float(config.min_recent_return_pct or 0.004))
        & (data["log_return_5"] > 0)
    )
    compression = data[range_column] <= max(0.10, float(config.take_profit_pct) * 4)
    trend_strength = data["trend_strength_20"].abs() >= float(config.min_trend_strength or 0.05)
    mask = breakout & trend_support & expansion & compression & trend_strength & ~data["extreme_crash_candle"].astype(bool)
    mask = apply_volatility_breakout_quality_filters(data, mask, config)
    return _build_research_trades_from_mask(
        data,
        mask,
        config,
        strategy_name=HTF_VOLATILITY_EXPANSION_BREAKOUT_STRATEGY,
        entry_reason="htf_volatility_expansion_breakout",
        regime="higher_timeframe_volatility_expansion",
    )


def volatility_breakout_quality_required_columns(config: ResearchConfig) -> list[str]:
    required: list[str] = []
    if config.require_ema_trend_filter:
        required.extend(["ema_20", "ema_50", "ema_50_above_200"])
    if config.require_positive_ema20_slope:
        required.append("ema_20_slope_5")
    if config.require_close_above_ema200:
        required.append("close_above_ema_200")
    if config.max_breakout_candle_atr_multiple is not None:
        required.append("breakout_candle_atr_multiple")
    if config.min_close_position_in_candle is not None:
        required.append("close_position_in_candle")
    if config.max_recent_runup_pct is not None:
        required.extend(["recent_runup_pct_5", "log_return_1"])
    if config.min_consolidation_compression is not None:
        required.append(f"range_compression_{int(config.consolidation_lookback or 20)}")
    if config.require_volume_expansion:
        required.extend(["normalized_volume", "volume_zscore_20"])
    if config.max_atr_percentile is not None:
        required.append("atr_percentile_200")
    return list(dict.fromkeys(required))


def apply_volatility_breakout_quality_filters(
    data: pd.DataFrame,
    mask: pd.Series,
    config: ResearchConfig,
) -> pd.Series:
    filtered = mask.copy()
    if config.require_ema_trend_filter:
        filtered &= (
            data["ema_50_above_200"].astype(bool)
            & (data["ema_20"] > data["ema_50"])
        )
    if config.require_positive_ema20_slope:
        filtered &= data["ema_20_slope_5"] > 0
    if config.require_close_above_ema200:
        filtered &= data["close_above_ema_200"].astype(bool)
    if config.max_breakout_candle_atr_multiple is not None:
        filtered &= data["breakout_candle_atr_multiple"] <= float(config.max_breakout_candle_atr_multiple)
    if config.min_close_position_in_candle is not None:
        filtered &= data["close_position_in_candle"] >= float(config.min_close_position_in_candle)
    if config.max_recent_runup_pct is not None:
        max_runup = float(config.max_recent_runup_pct)
        filtered &= (data["recent_runup_pct_5"] <= max_runup) & (data["log_return_1"] <= max_runup)
    if config.min_consolidation_compression is not None:
        compression_column = f"range_compression_{int(config.consolidation_lookback or 20)}"
        filtered &= data[compression_column] >= float(config.min_consolidation_compression)
    if config.require_volume_expansion:
        filtered &= (
            (data["normalized_volume"] >= 1.0)
            & (data["volume_zscore_20"] >= float(config.min_volume_zscore or 0.0))
        )
    if config.max_atr_percentile is not None:
        filtered &= data["atr_percentile_200"] <= float(config.max_atr_percentile)
    return filtered


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
                **_trade_audit_entry_exit_fields(row, exit_result),
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
                **_trade_audit_entry_exit_fields(row, exit_result),
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
    initial_stop_loss_price = entry_close * (1 - config.stop_loss_pct)
    stop_loss_price = initial_stop_loss_price
    max_exit_index = min(len(features) - 1, entry_index + config.max_hold_bars)
    exit_mode = _normalise_exit_mode(config.exit_mode or EXIT_MODE_FIXED)
    one_r_price = entry_close * (1 + config.stop_loss_pct)
    one_r_reached = False
    high_watermark = entry_close
    time_stop_after_bars = max(3, min(max(3, int(config.max_hold_bars) - 1), int(config.max_hold_bars * 0.50)))
    for offset, row_index in enumerate(range(entry_index + 1, max_exit_index + 1), start=1):
        row = features.iloc[row_index]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        high_watermark = max(high_watermark, high)
        one_r_reached = one_r_reached or high >= one_r_price
        if one_r_reached and exit_mode == EXIT_MODE_BREAK_EVEN_1R:
            stop_loss_price = max(stop_loss_price, entry_close)
        elif one_r_reached and exit_mode == EXIT_MODE_TRAILING_1R:
            stop_loss_price = max(stop_loss_price, entry_close, high_watermark * (1 - config.stop_loss_pct))
        hit_take_profit = high >= take_profit_price
        hit_stop_loss = low <= stop_loss_price
        window = features.iloc[entry_index + 1 : row_index + 1]
        if hit_take_profit and hit_stop_loss:
            gross_return = stop_loss_price / entry_close - 1
            exit_reason = (
                "ambiguous_stop_first"
                if stop_loss_price <= initial_stop_loss_price
                else _dynamic_stop_exit_reason(exit_mode)
            )
            return _exit_result(
                gross_return,
                exit_reason,
                offset,
                high,
                low,
                entry_close=entry_close,
                exit_timestamp=row.get("timestamp"),
                exit_index=row_index,
                window=window,
            )
        if hit_stop_loss:
            gross_return = stop_loss_price / entry_close - 1
            exit_reason = "research_stop_loss" if stop_loss_price <= initial_stop_loss_price else _dynamic_stop_exit_reason(exit_mode)
            return _exit_result(
                gross_return,
                exit_reason,
                offset,
                high,
                low,
                entry_close=entry_close,
                exit_timestamp=row.get("timestamp"),
                exit_index=row_index,
                window=window,
            )
        if one_r_reached and exit_mode == EXIT_MODE_MFE_PROTECT_1R_50:
            max_favorable_pct = max(0.0, high_watermark / entry_close - 1)
            protect_price = entry_close * (1 + max_favorable_pct * 0.50)
            if low <= protect_price:
                return _exit_result(
                    protect_price / entry_close - 1,
                    "research_mfe_protection",
                    offset,
                    high,
                    low,
                    entry_close=entry_close,
                    exit_timestamp=row.get("timestamp"),
                    exit_index=row_index,
                    window=window,
                )
        if hit_take_profit:
            return _exit_result(
                config.take_profit_pct,
                "research_take_profit",
                offset,
                high,
                low,
                entry_close=entry_close,
                exit_timestamp=row.get("timestamp"),
                exit_index=row_index,
                window=window,
            )
        if exit_mode == EXIT_MODE_TIME_STOP_MOMENTUM_WEAK and offset >= time_stop_after_bars:
            if _momentum_weak_for_time_stop(row):
                return _exit_result(
                    close / entry_close - 1,
                    "research_time_stop_momentum_weak",
                    offset,
                    high,
                    low,
                    entry_close=entry_close,
                    exit_timestamp=row.get("timestamp"),
                    exit_index=row_index,
                    window=window,
                )
    exit_row = features.iloc[max_exit_index]
    gross_return = float(exit_row["close"] / entry_close - 1)
    window = features.iloc[entry_index + 1 : max_exit_index + 1]
    return _exit_result(
        gross_return,
        "research_max_hold",
        max(1, max_exit_index - entry_index),
        float(exit_row["high"]),
        float(exit_row["low"]),
        entry_close=entry_close,
        exit_timestamp=exit_row.get("timestamp"),
        exit_index=max_exit_index,
        window=window,
    )


def _dynamic_stop_exit_reason(exit_mode: str) -> str:
    exit_mode = _normalise_exit_mode(exit_mode)
    if exit_mode == EXIT_MODE_BREAK_EVEN_1R:
        return "research_break_even_stop"
    if exit_mode == EXIT_MODE_TRAILING_1R:
        return "research_trailing_stop"
    return "research_stop_loss"


def _normalise_exit_mode(exit_mode: str) -> str:
    aliases = {
        EXIT_MODE_BREAK_EVEN_AFTER_1R: EXIT_MODE_BREAK_EVEN_1R,
        EXIT_MODE_TRAILING_AFTER_1R: EXIT_MODE_TRAILING_1R,
        EXIT_MODE_MFE_PROTECTION_EXIT: EXIT_MODE_MFE_PROTECT_1R_50,
    }
    return aliases.get(str(exit_mode), str(exit_mode))


def _momentum_weak_for_time_stop(row: pd.Series) -> bool:
    close = _metric_float(row.get("close"))
    ema_20 = _metric_float(row.get("ema_20"))
    log_return_3 = _metric_float(row.get("log_return_3"))
    ema_50_slope = _metric_float(row.get("ema_50_slope_5"))
    if close > 0 and ema_20 > 0 and close < ema_20:
        return True
    return log_return_3 <= 0 or ema_50_slope <= 0


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
    elif config.strategy_name in V3_STRATEGIES or config.strategy_name in HTF_STRATEGIES:
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
        elif config.strategy_name == HTF_TREND_CONTINUATION_STRATEGY:
            trades, signals = build_htf_trend_continuation_research_trades(fold, settings, config)
        elif config.strategy_name == HTF_VOLATILITY_EXPANSION_BREAKOUT_STRATEGY:
            trades, signals = build_htf_volatility_expansion_breakout_research_trades(fold, settings, config)
        elif config.strategy_name == HTF_RISK_OFF_HOLD_FILTER_STRATEGY:
            continue
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


def build_baselines_by_timeframe(
    bars_by_timeframe: dict[str, pd.DataFrame],
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    return {
        timeframe: calculate_baselines(bars, settings)
        for timeframe, bars in bars_by_timeframe.items()
    }


def calculate_baselines(bars: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    if bars.empty:
        return {
            "cash_return_pct": 0.0,
            "buy_and_hold_return_pct": 0.0,
            "buy_hold_max_drawdown_pct": 0.0,
            "buy_hold_return_over_drawdown": 0.0,
            "dca_daily_return_pct": 0.0,
            "dca_daily_max_drawdown_pct": 0.0,
            "dca_daily_return_over_drawdown": 0.0,
            "dca_weekly_return_pct": 0.0,
            "dca_weekly_max_drawdown_pct": 0.0,
            "dca_weekly_return_over_drawdown": 0.0,
            "dca_interval_return_pct": 0.0,
            "dca_interval_max_drawdown_pct": 0.0,
            "dca_interval_return_over_drawdown": 0.0,
            "baseline_window_start": None,
            "baseline_window_end": None,
            "baseline_rows": 0,
        }
    data = normalize_ohlcv(bars)
    closes = pd.to_numeric(data["close"], errors="coerce").dropna()
    if closes.empty:
        return calculate_baselines(pd.DataFrame(), settings)
    buy_hold_curve = closes / float(closes.iloc[0]) - 1
    buy_hold_return = float(closes.iloc[-1] / closes.iloc[0] - 1)
    buy_hold_drawdown = max_drawdown_from_return_curve(buy_hold_curve)
    daily = calculate_dca_baseline(data, settings, frequency="daily")
    weekly = calculate_dca_baseline(data, settings, frequency="weekly")
    interval = calculate_dca_baseline(data, settings, frequency="interval")
    return {
        "cash_return_pct": 0.0,
        "buy_and_hold_return_pct": buy_hold_return,
        "buy_hold_max_drawdown_pct": buy_hold_drawdown,
        "buy_hold_return_over_drawdown": return_over_drawdown(buy_hold_return, buy_hold_drawdown),
        "dca_daily_return_pct": daily["return_pct"],
        "dca_daily_max_drawdown_pct": daily["max_drawdown_pct"],
        "dca_daily_return_over_drawdown": daily["return_over_drawdown"],
        "dca_weekly_return_pct": weekly["return_pct"],
        "dca_weekly_max_drawdown_pct": weekly["max_drawdown_pct"],
        "dca_weekly_return_over_drawdown": weekly["return_over_drawdown"],
        "dca_interval_return_pct": interval["return_pct"],
        "dca_interval_max_drawdown_pct": interval["max_drawdown_pct"],
        "dca_interval_return_over_drawdown": interval["return_over_drawdown"],
        "baseline_window_start": data["timestamp"].iloc[0].isoformat(),
        "baseline_window_end": data["timestamp"].iloc[-1].isoformat(),
        "baseline_rows": int(len(data)),
    }


def calculate_dca_baseline(
    bars: pd.DataFrame,
    settings: Settings,
    *,
    frequency: str,
) -> dict[str, float]:
    data = normalize_ohlcv(bars)
    if data.empty:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0, "return_over_drawdown": 0.0}
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    if frequency == "daily":
        buy_keys = timestamps.dt.floor("D")
        buy_indexes = set(data.groupby(buy_keys, sort=True).head(1).index.tolist())
    elif frequency == "weekly":
        buy_keys = timestamps.dt.strftime("%G-%V")
        buy_indexes = set(data.groupby(buy_keys, sort=True).head(1).index.tolist())
    elif frequency == "interval":
        buy_indexes = set(data.index.tolist())
    else:
        raise ValueError(f"Unsupported DCA frequency: {frequency}")
    notional = max(0.0, float(settings.order_notional_usd))
    btc_qty = 0.0
    contributed = 0.0
    returns: list[float] = []
    for index, row in data.iterrows():
        close = float(row["close"])
        if index in buy_indexes and close > 0:
            contributed += notional
            btc_qty += notional / close
        value = btc_qty * close
        returns.append(float(value / contributed - 1) if contributed > 0 else 0.0)
    return_pct = returns[-1] if returns else 0.0
    drawdown = max_drawdown_from_return_curve(pd.Series(returns, dtype=float))
    return {
        "return_pct": return_pct,
        "max_drawdown_pct": drawdown,
        "return_over_drawdown": return_over_drawdown(return_pct, drawdown),
    }


def strategy_baseline_comparison(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    strategy_return = _metric_float(metrics.get("net_return_pct"))
    strategy_drawdown = _metric_float(metrics.get("max_drawdown_pct"))
    strategy_rod = return_over_drawdown(strategy_return, strategy_drawdown)
    buy_hold_rod = _metric_float(baseline.get("buy_hold_return_over_drawdown"))
    dca_daily_rod = _metric_float(baseline.get("dca_daily_return_over_drawdown"))
    best_baseline_rod = max(buy_hold_rod, dca_daily_rod)
    beats_buy_hold = strategy_rod > buy_hold_rod and strategy_return > _metric_float(
        baseline.get("buy_and_hold_return_pct")
    )
    beats_dca_daily = strategy_rod > dca_daily_rod and strategy_return > _metric_float(
        baseline.get("dca_daily_return_pct")
    )
    return {
        "buy_and_hold_return_pct": _metric_float(baseline.get("buy_and_hold_return_pct")),
        "dca_daily_return_pct": _metric_float(baseline.get("dca_daily_return_pct")),
        "dca_weekly_return_pct": _metric_float(baseline.get("dca_weekly_return_pct")),
        "strategy_net_return_pct": strategy_return,
        "strategy_excess_return_vs_buy_hold": strategy_return
        - _metric_float(baseline.get("buy_and_hold_return_pct")),
        "strategy_excess_return_vs_dca_daily": strategy_return - _metric_float(baseline.get("dca_daily_return_pct")),
        "strategy_max_drawdown_pct": strategy_drawdown,
        "buy_hold_max_drawdown_pct": _metric_float(baseline.get("buy_hold_max_drawdown_pct")),
        "dca_max_drawdown_pct": _metric_float(baseline.get("dca_daily_max_drawdown_pct")),
        "strategy_return_over_drawdown": strategy_rod,
        "baseline_return_over_drawdown": best_baseline_rod,
        "buy_hold_return_over_drawdown": buy_hold_rod,
        "dca_daily_return_over_drawdown": dca_daily_rod,
        "beats_buy_hold_risk_adjusted": beats_buy_hold,
        "beats_dca_daily_risk_adjusted": beats_dca_daily,
        "beats_any_relevant_baseline_risk_adjusted": beats_buy_hold or beats_dca_daily,
    }


def evaluate_cost_scenarios_for_config(
    trades: pd.DataFrame,
    signal_frame: pd.DataFrame,
    settings: Settings,
    config: ResearchConfig,
    *,
    fallback_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scenario_metrics: dict[str, dict[str, Any]] = {}
    for scenario in COST_SCENARIOS:
        scenario_settings = research_config_settings(settings, config, cost_scenario=scenario)
        if trades.empty and fallback_metrics is not None and config.strategy_name == HTF_RISK_OFF_HOLD_FILTER_STRATEGY:
            metrics = dict(fallback_metrics)
        else:
            metrics = calculate_fee_aware_metrics(trades, scenario_settings, signal_frame=signal_frame)
        scenario_metrics[scenario] = metrics
    net_by_scenario = {
        scenario: _metric_float(metrics.get("net_return_pct"))
        for scenario, metrics in scenario_metrics.items()
    }
    profit_factor_by_scenario = {
        scenario: _profit_factor_value(metrics.get("profit_factor_net"))
        for scenario, metrics in scenario_metrics.items()
    }
    drawdown_by_scenario = {
        scenario: _metric_float(metrics.get("max_drawdown_pct"))
        for scenario, metrics in scenario_metrics.items()
    }
    profitable_zero = net_by_scenario.get("zero_cost_sanity", 0.0) > 0
    profitable_low = net_by_scenario.get("maker_low_slippage", 0.0) > 0
    profitable_maker = net_by_scenario.get("maker_current", 0.0) > 0
    profitable_taker = net_by_scenario.get("current_taker", 0.0) > 0
    return {
        "cost_scenarios_tested": list(COST_SCENARIOS),
        "net_return_by_cost_scenario": net_by_scenario,
        "profit_factor_by_cost_scenario": profit_factor_by_scenario,
        "max_drawdown_by_cost_scenario": drawdown_by_scenario,
        "profitable_under_zero_cost": profitable_zero,
        "profitable_under_maker_low_slippage": profitable_low,
        "profitable_under_current_taker": profitable_taker,
        "cost_sensitivity_classification": classify_cost_sensitivity(
            profitable_zero=profitable_zero,
            profitable_low=profitable_low,
            profitable_maker=profitable_maker,
            profitable_taker=profitable_taker,
        ),
    }


def classify_cost_sensitivity(
    *,
    profitable_zero: bool,
    profitable_low: bool,
    profitable_maker: bool,
    profitable_taker: bool,
) -> str:
    if profitable_taker:
        return "signal_survives_current_taker_cost"
    if profitable_maker:
        return "signal_survives_maker_cost"
    if profitable_low:
        return "signal_positive_low_cost_only"
    if profitable_zero:
        return "signal_positive_only_zero_cost"
    return "signal_negative_even_zero_cost"


def calculate_risk_off_hold_filter(features: pd.DataFrame, config: ResearchConfig) -> dict[str, Any]:
    if features.empty:
        return {
            "filtered_hold_return_pct": 0.0,
            "filtered_hold_max_drawdown_pct": 0.0,
            "filtered_hold_excess_return_vs_buy_hold": 0.0,
            "filtered_hold_drawdown_reduction": 0.0,
            "time_in_market_pct": 0.0,
        }
    required = [
        "timestamp",
        "close",
        "ema_200",
        "ema_50_above_200",
        "drawdown_from_high_200",
        "atr_expansion_20",
        "large_negative_candle",
        "downside_volatility_cluster",
        "log_return_5",
    ]
    data = features.dropna(subset=required).reset_index(drop=True)
    if len(data) < 2:
        return {
            "filtered_hold_return_pct": 0.0,
            "filtered_hold_max_drawdown_pct": 0.0,
            "filtered_hold_excess_return_vs_buy_hold": 0.0,
            "filtered_hold_drawdown_reduction": 0.0,
            "time_in_market_pct": 0.0,
        }
    risk_off = (
        (data["close"] < data["ema_200"])
        | ~data["ema_50_above_200"].astype(bool)
        | (data["drawdown_from_high_200"] >= float(config.drawdown_threshold or 0.18))
        | (data["atr_expansion_20"] >= float(config.max_atr_expansion or 2.4))
        | data["large_negative_candle"].astype(bool)
        | data["downside_volatility_cluster"].astype(bool)
        | (data["log_return_5"] <= float(config.min_recent_return_pct or -0.06))
    )
    exposure = (~risk_off).astype(float).shift(1).fillna(0.0)
    returns = pd.to_numeric(data["close"], errors="coerce").pct_change().fillna(0.0)
    filtered_returns = exposure * returns
    filtered_curve = (1 + filtered_returns).cumprod() - 1
    filtered_return = float(filtered_curve.iloc[-1]) if len(filtered_curve) else 0.0
    filtered_drawdown = max_drawdown_from_return_curve(filtered_curve)
    buy_hold_return = float(data["close"].iloc[-1] / data["close"].iloc[0] - 1)
    buy_hold_curve = data["close"] / data["close"].iloc[0] - 1
    buy_hold_drawdown = max_drawdown_from_return_curve(buy_hold_curve)
    return {
        "filtered_hold_return_pct": filtered_return,
        "filtered_hold_max_drawdown_pct": filtered_drawdown,
        "filtered_hold_excess_return_vs_buy_hold": filtered_return - buy_hold_return,
        "filtered_hold_drawdown_reduction": buy_hold_drawdown - filtered_drawdown,
        "time_in_market_pct": float(exposure.mean()) if len(exposure) else 0.0,
    }


def max_drawdown_from_return_curve(return_curve: pd.Series | list[float] | np.ndarray) -> float:
    values = np.asarray(return_curve, dtype=float)
    if len(values) == 0:
        return 0.0
    equity = 1 + values
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    return float(np.max(drawdowns)) if len(drawdowns) else 0.0


def max_drawdown_from_trade_returns(returns: list[float] | np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    if len(values) == 0:
        return 0.0
    equity = np.cumprod(1 + values) - 1
    return max_drawdown_from_return_curve(equity)


def return_over_drawdown(return_pct: float, drawdown_pct: float) -> float:
    drawdown = abs(float(drawdown_pct))
    if drawdown <= 0:
        return 1_000_000.0 if float(return_pct) > 0 else 0.0
    return float(return_pct) / drawdown


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
    if source_used is not None and not _is_collected_market_data_source(source_used):
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


def volatility_focus_research_gate(
    metrics: dict[str, Any],
    settings: Settings,
    config: ResearchConfig,
    *,
    cost_summary: dict[str, Any],
    source_report: ResearchDataReport | None,
    synthetic_data_used: bool,
    research_result_valid: bool,
    baseline_comparison: dict[str, Any],
    walk_forward: dict[str, Any],
    concentration: float,
    active_model_valid: bool,
    min_focused_trades: int,
) -> dict[str, Any]:
    research_reasons: list[str] = []
    if not research_result_valid:
        research_reasons.append("research_data_source_invalid")
    if synthetic_data_used:
        research_reasons.append("synthetic_data_used")
    if source_report is None or not _is_collected_market_data_source(source_report.source_used):
        research_reasons.append("data_source_not_collected_market_data")
    if config.timeframe not in {"1H", "4H"}:
        research_reasons.append("timeframe_not_1h_or_valid_4h")
    if config.timeframe == "4H" and source_report is not None and not source_report.research_result_valid:
        research_reasons.append("valid_4h_rows_missing")

    trade_count = int(metrics.get("number_of_trades", 0) or 0)
    if trade_count < int(min_focused_trades):
        research_reasons.append(f"number_of_trades_below_{int(min_focused_trades)}")
    if int(walk_forward.get("folds_with_min_trades_count", 0) or 0) < 3:
        research_reasons.append("folds_with_min_trades_below_3")
    if not bool(walk_forward.get("walk_forward_passed")):
        research_reasons.append("walk_forward_not_passed")

    current_taker_net = _metric_float((cost_summary.get("net_return_by_cost_scenario") or {}).get("current_taker"))
    current_taker_pf = _profit_factor_value(
        (cost_summary.get("profit_factor_by_cost_scenario") or {}).get("current_taker")
    )
    if current_taker_net <= 0:
        research_reasons.append("current_taker_net_return_not_positive")
    if current_taker_pf < MIN_RESEARCH_PROFIT_FACTOR_NET:
        research_reasons.append("current_taker_profit_factor_below_1_05")
    if _metric_float(metrics.get("max_drawdown_pct")) > float(settings.max_backtest_drawdown_pct):
        research_reasons.append("max_drawdown_above_configured_limit")
    if not bool(baseline_comparison.get("beats_buy_hold_risk_adjusted")):
        research_reasons.append("does_not_beat_buy_and_hold_risk_adjusted")
    if not bool(baseline_comparison.get("beats_dca_daily_risk_adjusted")):
        research_reasons.append("does_not_beat_dca_risk_adjusted")
    if _metric_float(concentration) > MAX_SINGLE_TRADE_RETURN_SHARE:
        research_reasons.append("single_trade_return_concentration_too_high")

    research_reasons = list(dict.fromkeys(research_reasons))
    research_promising = not research_reasons
    paper_reasons = list(research_reasons)
    if not active_model_valid:
        paper_reasons.append("active_model_invalid")
    training_reasons = list(research_reasons)
    if not active_model_valid:
        training_reasons.append("active_model_invalid")
    training_reasons.append("training_deferred_volatility_focus_no_ml_yet")
    return {
        "research_promising": research_promising,
        "economically_viable": research_promising,
        "paper_forward_eligible": research_promising and active_model_valid,
        "training_eligible": False,
        "research_rejection_reasons": research_reasons,
        "paper_forward_rejection_reasons": list(dict.fromkeys(paper_reasons)),
        "training_rejection_reasons": list(dict.fromkeys(training_reasons)),
    }


def volatility_focus_maker_research_gate(
    metrics: dict[str, Any],
    settings: Settings,
    config: ResearchConfig,
    *,
    cost_summary: dict[str, Any],
    source_report: ResearchDataReport | None,
    synthetic_data_used: bool,
    research_result_valid: bool,
    baseline_comparison: dict[str, Any],
    walk_forward: dict[str, Any],
    min_focused_trades: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not research_result_valid:
        reasons.append("research_data_source_invalid")
    if synthetic_data_used:
        reasons.append("synthetic_data_used")
    if source_report is None or not _is_collected_market_data_source(source_report.source_used):
        reasons.append("data_source_not_collected_market_data")
    if settings.symbol != ALLOWED_SYMBOL:
        reasons.append("symbol_not_btc_usd")
    if config.timeframe != "1H":
        reasons.append("timeframe_not_1h")
    if config.strategy_name != VOLATILITY_BREAKOUT_STRATEGY:
        reasons.append("strategy_not_volatility_breakout")

    trade_count = int(metrics.get("number_of_trades", 0) or 0)
    if trade_count < int(min_focused_trades):
        reasons.append(f"number_of_trades_below_{int(min_focused_trades)}")
    if int(walk_forward.get("folds_with_min_trades_count", 0) or 0) != int(walk_forward.get("fold_count", 4) or 4):
        reasons.append("folds_with_min_trades_not_all_4")
    if not bool(walk_forward.get("walk_forward_passed")):
        reasons.append("walk_forward_not_passed")

    maker_net = _metric_float((cost_summary.get("net_return_by_cost_scenario") or {}).get("maker_current"))
    maker_pf = _profit_factor_value((cost_summary.get("profit_factor_by_cost_scenario") or {}).get("maker_current"))
    if maker_net <= 0:
        reasons.append("maker_current_net_return_not_positive")
    if maker_pf < MIN_RESEARCH_PROFIT_FACTOR_NET:
        reasons.append("maker_current_profit_factor_below_1_05")
    if _metric_float(metrics.get("max_drawdown_pct")) > float(settings.max_backtest_drawdown_pct):
        reasons.append("max_drawdown_above_configured_limit")
    if not bool(baseline_comparison.get("beats_buy_hold_risk_adjusted")):
        reasons.append("does_not_beat_buy_and_hold_risk_adjusted")
    if not bool(baseline_comparison.get("beats_dca_daily_risk_adjusted")):
        reasons.append("does_not_beat_dca_risk_adjusted")

    reasons = list(dict.fromkeys(reasons))
    maker_research_promising = not reasons
    return {
        "maker_research_promising": maker_research_promising,
        "maker_economically_viable": maker_research_promising,
        "maker_rejection_reasons": reasons,
    }


def volatility_focus_maker_execution_diagnostics(settings: Settings, cost_summary: dict[str, Any]) -> dict[str, Any]:
    net_by_scenario = cost_summary.get("net_return_by_cost_scenario") or {}
    maker_net = _metric_float(net_by_scenario.get("maker_current"))
    taker_net = _metric_float(net_by_scenario.get("current_taker"))
    maker_taker_gap = maker_net - taker_net
    if maker_net > 0 and taker_net < 0 and maker_taker_gap > 0:
        fill_rate_required = max(0.0, min(1.0, -taker_net / maker_taker_gap))
        max_taker_fallback_rate = max(0.0, min(1.0, maker_net / maker_taker_gap))
    elif maker_net > 0 and taker_net >= 0:
        fill_rate_required = 0.0
        max_taker_fallback_rate = 1.0
    else:
        fill_rate_required = 1.0
        max_taker_fallback_rate = 0.0
    return {
        "estimated_fill_rate_required_to_remain_profitable": fill_rate_required,
        "maker_vs_taker_net_gap": maker_taker_gap,
        "max_allowed_taker_fallback_rate_before_net_negative": max_taker_fallback_rate,
        "spread_bps_assumption": _metric_float(getattr(settings, "max_spread_bps", 0.0)),
        "slippage_bps_assumption": _metric_float(getattr(settings, "slippage_bps", 0.0)),
        "no_market_fallback_required": True,
        "post_only_required": True,
        "unfilled_cancel_required": True,
    }


def volatility_focus_trade_diagnostics(
    trades: pd.DataFrame,
    signal_frame: pd.DataFrame,
    metrics: dict[str, Any],
    config: ResearchConfig,
    *,
    walk_forward: dict[str, Any],
    cost_summary: dict[str, Any],
) -> dict[str, Any]:
    if trades.empty:
        mfe = np.asarray([], dtype=float)
        mae = np.asarray([], dtype=float)
        bars_held = np.asarray([], dtype=float)
        exit_reasons: list[str] = []
    else:
        mfe = np.asarray(
            [_metric_float(value) for value in trades.get("max_favorable_excursion_pct", pd.Series(dtype=float))],
            dtype=float,
        )
        mae = np.asarray(
            [_metric_float(value) for value in trades.get("max_adverse_excursion_pct", pd.Series(dtype=float))],
            dtype=float,
        )
        bars_held = np.asarray(
            [_metric_float(value) for value in trades.get("buy_hold_bars", pd.Series(dtype=float))],
            dtype=float,
        )
        exit_reasons = [str(value) for value in trades.get("buy_exit_reason", pd.Series(dtype=object)).tolist()]
    trade_count = int(metrics.get("number_of_trades", 0) or 0)
    stop_risk = max(1e-12, float(config.stop_loss_pct))
    quick_stop_bars = max(1.0, min(3.0, float(config.max_hold_bars) * 0.25))
    stopped = np.asarray(
        [
            reason
            in {
                "research_stop_loss",
                "ambiguous_stop_first",
                "research_break_even_stop",
                "research_trailing_stop",
            }
            for reason in exit_reasons
        ],
        dtype=bool,
    )
    timed_out = np.asarray(
        [reason in {"research_max_hold", "research_time_stop_momentum_weak"} for reason in exit_reasons],
        dtype=bool,
    )
    took_profit = np.asarray([reason == "research_take_profit" for reason in exit_reasons], dtype=bool)
    protected = np.asarray([reason == "research_mfe_protection" for reason in exit_reasons], dtype=bool)
    stopped_quickly = stopped & (bars_held <= quick_stop_bars) if len(bars_held) else np.asarray([], dtype=bool)
    return {
        "total_entries": int(len(signal_frame)),
        "total_exits": int(len(trades)),
        "trade_count": trade_count,
        "average_mfe": float(mfe.mean()) if len(mfe) else 0.0,
        "average_mae": float(mae.mean()) if len(mae) else 0.0,
        "median_mfe": float(np.median(mfe)) if len(mfe) else 0.0,
        "median_mae": float(np.median(mae)) if len(mae) else 0.0,
        "pct_trades_reaching_1r_before_stop": float((mfe >= stop_risk).mean()) if len(mfe) else 0.0,
        "pct_trades_reaching_2r_before_stop": float((mfe >= stop_risk * 2).mean()) if len(mfe) else 0.0,
        "pct_trades_stopped_quickly": float(stopped_quickly.mean()) if len(stopped_quickly) else 0.0,
        "pct_trades_timing_out": float(timed_out.mean()) if len(timed_out) else 0.0,
        "pct_trades_exiting_by_take_profit": float(took_profit.mean()) if len(took_profit) else 0.0,
        "pct_trades_exiting_by_stop_loss": float(stopped.mean()) if len(stopped) else 0.0,
        "pct_trades_exiting_by_protective_stop": float(protected.mean()) if len(protected) else 0.0,
        "average_bars_held": float(bars_held.mean()) if len(bars_held) else 0.0,
        "median_bars_held": float(np.median(bars_held)) if len(bars_held) else 0.0,
        "fold_by_fold_returns": walk_forward.get("per_fold_net_return_pct", []),
        "fold_by_fold_trade_counts": walk_forward.get("per_fold_number_of_trades", []),
        "fold_by_fold_profit_factor": walk_forward.get("per_fold_profit_factor_net", []),
        "cost_sensitivity_classification": cost_summary.get("cost_sensitivity_classification"),
    }


def volatility_focus_rank_details(
    metrics: dict[str, Any],
    focused_gate: dict[str, Any],
    *,
    concentration: float,
    walk_forward: dict[str, Any],
    cost_summary: dict[str, Any],
    baseline_comparison: dict[str, Any],
    target_focused_trades: int,
) -> dict[str, Any]:
    readiness = {
        "economically_viable": focused_gate.get("economically_viable"),
        "paper_forward_eligible": focused_gate.get("paper_forward_eligible"),
    }
    base = research_rank_details(metrics, readiness, concentration=concentration, walk_forward=walk_forward)
    current_net = _metric_float((cost_summary.get("net_return_by_cost_scenario") or {}).get("current_taker"))
    current_pf = min(
        5.0,
        _profit_factor_value((cost_summary.get("profit_factor_by_cost_scenario") or {}).get("current_taker")),
    )
    trades = int(metrics.get("number_of_trades", 0) or 0)
    target = max(1, int(target_focused_trades))
    trade_score = max(0.0, min(1.0, trades / target))
    baseline_edge = _metric_float(baseline_comparison.get("strategy_return_over_drawdown")) - _metric_float(
        baseline_comparison.get("baseline_return_over_drawdown")
    )
    adjusted = _metric_float(base["adjusted_rank_score"])
    adjusted += current_net * 250_000
    adjusted += current_pf * 3_000
    adjusted += trade_score * 30_000
    adjusted += int(walk_forward.get("folds_with_min_trades_count", 0) or 0) * 5_000
    adjusted += baseline_edge * 10_000
    if not focused_gate.get("research_promising"):
        adjusted -= 250_000
    reasons = [base.get("reason_ranked_lower_if_any")]
    reasons.extend(focused_gate.get("research_rejection_reasons", []))
    base["adjusted_rank_score"] = adjusted
    base["reason_ranked_lower_if_any"] = ";".join(
        dict.fromkeys(reason for reason in reasons if reason)
    ) or None
    return base


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
    audit_mode: str = "standard",
    bars_by_timeframe: dict[str, pd.DataFrame] | None = None,
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
    rejection_summary = build_15min_rejection_summary(ranked, requested_timeframes=requested_timeframes)
    baselines = (
        build_baselines_by_timeframe(bars_by_timeframe, settings)
        if bars_by_timeframe is not None
        else build_baselines_from_rows(ranked)
    )
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
            "audit_mode": audit_mode,
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
            "baselines": baselines,
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
            **rejection_summary,
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
                HTF_TREND_CONTINUATION_STRATEGY: {
                    "timeframes": list(HTF_RESEARCH_TIMEFRAMES),
                    "take_profit_pct": list(HTF_TREND_TAKE_PROFIT_VALUES),
                    "stop_loss_pct": list(HTF_TREND_STOP_LOSS_VALUES),
                    "max_hold_bars": list(HTF_MAX_HOLD_BARS_VALUES),
                },
                HTF_VOLATILITY_EXPANSION_BREAKOUT_STRATEGY: {
                    "timeframes": list(HTF_RESEARCH_TIMEFRAMES),
                    "take_profit_pct": list(HTF_BREAKOUT_TAKE_PROFIT_VALUES),
                    "stop_loss_pct": list(HTF_BREAKOUT_STOP_LOSS_VALUES),
                    "max_hold_bars": list(HTF_MAX_HOLD_BARS_VALUES),
                },
                HTF_RISK_OFF_HOLD_FILTER_STRATEGY: {
                    "timeframes": list(HTF_RESEARCH_TIMEFRAMES),
                    "profiles": list(RISK_OFF_FILTER_PROFILES),
                    "trainable_entry_strategy": False,
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


def build_strategy_reality_audit_summary(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    settings: Settings,
    *,
    data_source_reports: dict[str, ResearchDataReport | dict[str, Any]],
    bars_by_timeframe: dict[str, pd.DataFrame],
    reality_summary_path: Path,
    cost_scenarios: tuple[str, ...],
    trade_audit_paths: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = sorted(rows, key=_rank_sort_key, reverse=True)
    source_reports = _summary_source_reports(data_source_reports=data_source_reports, data_sources=None)
    best_by_cost = {
        scenario: best_config_for_cost_scenario(ranked, scenario)
        for scenario in cost_scenarios
    }
    baseline_candidates = [
        row
        for row in ranked
        if row.get("beats_any_relevant_baseline_risk_adjusted")
    ]
    best_vs_baseline = max(
        baseline_candidates,
        key=lambda row: (
            _metric_float(row.get("strategy_return_over_drawdown"))
            - _metric_float(row.get("baseline_return_over_drawdown")),
            _metric_float(row.get("strategy_excess_return_vs_buy_hold")),
            _metric_float(row.get("net_return_pct")),
        ),
        default=None,
    )
    updated = dict(summary)
    updated.update(
        {
            "generated_at": summary.get("generated_at") or datetime.now(UTC).isoformat(),
            "symbol": settings.symbol,
            "paper_trading_only": settings.paper_trading_only,
            "trading_enabled": settings.trading_enabled,
            "auto_trade_enabled": settings.auto_trade_enabled,
            "orders_placed": 0,
            "synthetic_data_used": bool(summary.get("synthetic_data_used")),
            "data_ready": bool(summary.get("data_ready", summary.get("research_result_valid"))),
            "research_result_valid": bool(summary.get("research_result_valid")),
            "timeframes_used": list(summary.get("timeframes") or bars_by_timeframe.keys()),
            "source_used_by_timeframe": {
                timeframe: report.source_used for timeframe, report in source_reports.items()
            },
            "cost_scenarios_tested": list(cost_scenarios),
            "baselines": summary.get("baselines") or build_baselines_by_timeframe(bars_by_timeframe, settings),
            "rejected_strategy_families": summary.get("rejected_strategy_families", []),
            "best_strategy_by_cost_scenario": best_by_cost,
            "best_strategy_vs_baseline": best_vs_baseline,
            "best_trade_audit_path": (
                trade_audit_paths[0].get("csv_path")
                if trade_audit_paths
                else summary.get("best_trade_audit_path")
            ),
            "strategy_reality_audit_summary_path": str(reality_summary_path),
            "recommendation": reality_audit_recommendation(summary, ranked),
            "strategy_families": summary.get("strategy_breakdown", {}),
        }
    )
    return _json_safe(updated)


def build_volatility_focus_summary(
    rows: list[dict[str, Any]],
    settings: Settings,
    *,
    base_summary: dict[str, Any],
    data_source_reports: dict[str, ResearchDataReport | dict[str, Any]],
    bars_by_timeframe: dict[str, pd.DataFrame],
    min_focused_trades: int,
    target_focused_trades: int,
    max_focused_configs: int,
    focused_summary_path: Path,
    top_configs_csv_path: Path,
    rejections_path: Path,
    trade_audit_paths: list[dict[str, Any]],
) -> dict[str, Any]:
    v9_mode = _volatility_focus_v9_mode(rows)
    rank_key = _maker_rank_sort_key if v9_mode else _rank_sort_key
    ranked = sorted(rows, key=rank_key, reverse=True)
    source_reports = _summary_source_reports(data_source_reports=data_source_reports, data_sources=None)
    synthetic_data_used = bool(base_summary.get("synthetic_data_used")) or any(
        bool(row.get("synthetic_data_used")) for row in ranked
    )
    research_result_valid = bool(base_summary.get("research_result_valid")) and not synthetic_data_used
    eligible_rows = [] if synthetic_data_used or not research_result_valid else ranked
    current_profitable = [
        row
        for row in eligible_rows
        if _metric_float((row.get("net_return_by_cost_scenario") or {}).get("current_taker")) > 0
    ]
    maker_profitable = [
        row
        for row in eligible_rows
        if _metric_float((row.get("net_return_by_cost_scenario") or {}).get("maker_current")) > 0
    ]
    low_profitable = [
        row
        for row in eligible_rows
        if _metric_float((row.get("net_return_by_cost_scenario") or {}).get("maker_low_slippage")) > 0
    ]
    zero_profitable = [
        row
        for row in eligible_rows
        if _metric_float((row.get("net_return_by_cost_scenario") or {}).get("zero_cost_sanity")) > 0
    ]
    walk_forward_rows = [row for row in eligible_rows if row.get("walk_forward_passed")]
    research_promising_rows = [row for row in eligible_rows if row.get("research_promising")]
    economically_viable_rows = [row for row in eligible_rows if row.get("economically_viable")]
    paper_forward_rows = [row for row in eligible_rows if row.get("paper_forward_eligible")]
    all_track_a_rows = [row for row in ranked if row.get("track_id") == VOLATILITY_FOCUS_TRACK_A]
    all_track_b_rows = [row for row in ranked if row.get("track_id") == VOLATILITY_FOCUS_TRACK_B]
    all_track_m_rows = [row for row in ranked if row.get("track_id") == VOLATILITY_FOCUS_TRACK_M]
    all_track_t_rows = [row for row in ranked if row.get("track_id") == VOLATILITY_FOCUS_TRACK_T]
    all_track_m9_rows = [row for row in ranked if row.get("track_id") == VOLATILITY_FOCUS_TRACK_M9]
    track_a_rows = [row for row in eligible_rows if row.get("track_id") == VOLATILITY_FOCUS_TRACK_A]
    track_b_rows = [row for row in eligible_rows if row.get("track_id") == VOLATILITY_FOCUS_TRACK_B]
    track_m_rows = [row for row in eligible_rows if row.get("track_id") == VOLATILITY_FOCUS_TRACK_M]
    track_t_rows = [row for row in eligible_rows if row.get("track_id") == VOLATILITY_FOCUS_TRACK_T]
    track_m9_rows = [row for row in eligible_rows if row.get("track_id") == VOLATILITY_FOCUS_TRACK_M9]
    maker_research_promising_rows = [row for row in eligible_rows if row.get("maker_research_promising")]
    maker_economically_viable_rows = [row for row in eligible_rows if row.get("maker_economically_viable")]
    maker_only_rows = [row for row in eligible_rows if row.get("maker_only_candidate")]
    maker_twenty_plus_rows = [
        row for row in eligible_rows if int(row.get("number_of_trades", 0) or 0) >= MIN_RESEARCH_TRADES
    ]
    maker_walk_forward_rows = [row for row in maker_twenty_plus_rows if bool(row.get("walk_forward_passed"))]
    twenty_plus_current_positive = [
        row
        for row in eligible_rows
        if int(row.get("number_of_trades", 0) or 0) >= MIN_RESEARCH_TRADES
        and _metric_float((row.get("net_return_by_cost_scenario") or {}).get("current_taker")) > 0
    ]
    walk_forward_current_positive = [
        row
        for row in twenty_plus_current_positive
        if bool(row.get("walk_forward_passed"))
    ]
    all_research_gates_passed = [
        row
        for row in eligible_rows
        if not str(row.get("research_rejection_reasons") or "")
        and bool(row.get("research_promising"))
    ]
    v9_rows_for_candidate_selection = track_m9_rows if research_result_valid and not synthetic_data_used else all_track_m9_rows
    v9_twenty_plus_rows = [
        row
        for row in v9_rows_for_candidate_selection
        if int(row.get("number_of_trades", 0) or 0) >= MIN_RESEARCH_TRADES
    ]
    v9_drawdown_reduced_rows = [
        row
        for row in v9_twenty_plus_rows
        if _metric_float(row.get("max_drawdown_pct")) < VOLATILITY_FOCUS_V9_ANCHOR_MAX_DRAWDOWN_PCT
    ]
    v9_under_drawdown_limit_rows = [
        row
        for row in v9_twenty_plus_rows
        if _metric_float(row.get("max_drawdown_pct")) <= float(settings.max_backtest_drawdown_pct)
    ]
    diagnosis = volatility_focus_diagnosis(ranked, research_result_valid=research_result_valid)
    v7_failure = volatility_focus_v7_failure_analysis(eligible_rows if research_result_valid else ranked)
    v8_blockers = {
        "current_taker": volatility_focus_blocker_summary(
            eligible_rows if research_result_valid else ranked,
            "research_rejection_reasons",
        ),
        "maker": volatility_focus_blocker_summary(
            eligible_rows if research_result_valid else ranked,
            "maker_rejection_reasons",
        ),
    }
    recommendation = volatility_focus_recommendation(
        ranked,
        research_result_valid=research_result_valid,
        synthetic_data_used=synthetic_data_used,
        diagnosis=diagnosis,
        max_focused_configs=max_focused_configs,
    )
    v9_terminal = volatility_focus_v9_terminal_state(
        v9_rows_for_candidate_selection,
        settings,
        research_result_valid=research_result_valid,
        synthetic_data_used=synthetic_data_used,
    )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol": settings.symbol,
        "paper_trading_only": settings.paper_trading_only,
        "trading_enabled": settings.trading_enabled,
        "auto_trade_enabled": settings.auto_trade_enabled,
        "fallback_trading_allowed": settings.allow_fallback_trading,
        "btc_usd_only": settings.symbol == ALLOWED_SYMBOL,
        "long_only": True,
        "orders_placed": 0,
        "synthetic_data_used": synthetic_data_used,
        "research_result_valid": research_result_valid,
        "data_ready": research_result_valid,
        "timeframes_used": list(base_summary.get("timeframes") or bars_by_timeframe.keys()),
        "source_used_by_timeframe": {
            timeframe: report.source_used for timeframe, report in source_reports.items()
        },
        "row_count": {
            timeframe: report.row_count for timeframe, report in source_reports.items()
        },
        "baselines": base_summary.get("baselines") or build_baselines_by_timeframe(bars_by_timeframe, settings),
        "cost_scenarios_tested": list(COST_SCENARIOS),
        "configs_tested": len(ranked),
        "max_focused_configs": int(max_focused_configs),
        "min_focused_trades": int(min_focused_trades),
        "target_focused_trades": int(target_focused_trades),
        "v9_track_configs": volatility_focus_v9_track_configs(all_track_m9_rows),
        "configs_with_20_plus_trades": sum(1 for row in ranked if int(row.get("number_of_trades", 0) or 0) >= 20),
        "configs_with_50_plus_trades": sum(1 for row in ranked if int(row.get("number_of_trades", 0) or 0) >= 50),
        "profitable_current_cost_configs": len(current_profitable),
        "profitable_maker_cost_configs": len(maker_profitable),
        "profitable_low_cost_configs": len(low_profitable),
        "profitable_zero_cost_configs": len(zero_profitable),
        "walk_forward_passed_count": len(walk_forward_rows),
        "research_promising_count": len(research_promising_rows),
        "current_taker_research_promising_count": len(research_promising_rows),
        "maker_research_promising_count": len(maker_research_promising_rows),
        "economically_viable_count": len(economically_viable_rows),
        "maker_economically_viable_count": len(maker_economically_viable_rows),
        "maker_only_candidate_count": len(maker_only_rows),
        "paper_forward_eligible_count": len(paper_forward_rows),
        "best_current_cost_config": best_config_for_cost_scenario(eligible_rows, "current_taker"),
        "best_current_taker_candidate": best_config_for_cost_scenario(eligible_rows, "current_taker"),
        "best_low_cost_config": best_config_for_cost_scenario(eligible_rows, "maker_low_slippage"),
        "best_zero_cost_config": best_config_for_cost_scenario(eligible_rows, "zero_cost_sanity"),
        "best_walk_forward_config": best_ranked_config(walk_forward_rows),
        "candidate_a_best": best_ranked_config(track_a_rows or all_track_a_rows),
        "candidate_b_best": best_ranked_config(track_b_rows or all_track_b_rows),
        "best_maker_candidate": best_maker_ranked_config(track_m_rows or eligible_rows),
        "best_maker_candidate_with_20_plus_trades": best_maker_ranked_config(maker_twenty_plus_rows),
        "best_maker_walk_forward_candidate": best_maker_ranked_config(maker_walk_forward_rows),
        "best_v9_maker_candidate": best_maker_ranked_config(v9_twenty_plus_rows),
        "best_v9_drawdown_reduced_candidate": best_maker_ranked_config(v9_drawdown_reduced_rows),
        "best_taker_swing_candidate": best_ranked_config(track_t_rows or all_track_t_rows),
        "best_20_plus_current_cost_positive": best_ranked_config(twenty_plus_current_positive),
        "best_walk_forward_current_cost_positive": best_ranked_config(walk_forward_current_positive),
        "best_all_research_gates_passed": best_ranked_config(all_research_gates_passed),
        "best_all_current_taker_research_gates_passed": best_ranked_config(all_research_gates_passed),
        "best_all_maker_research_gates_passed": best_maker_ranked_config(maker_research_promising_rows),
        "best_candidate_under_drawdown_limit": best_maker_ranked_config(v9_under_drawdown_limit_rows),
        "any_config_passed_all_research_gates": bool(all_research_gates_passed),
        "any_current_taker_config_passed_all_research_gates": bool(all_research_gates_passed),
        "any_maker_only_config_passed_maker_research_gates": bool(maker_only_rows),
        "any_maker_config_passed_maker_research_gates": bool(maker_research_promising_rows),
        "volatility_focus_v7_failure_analysis": v7_failure,
        "volatility_focus_v8_blockers": v8_blockers,
        "volatility_focus_v7_track_counts": {
            VOLATILITY_FOCUS_TRACK_A: len(all_track_a_rows),
            VOLATILITY_FOCUS_TRACK_B: len(all_track_b_rows),
        },
        "volatility_focus_v8_track_counts": {
            VOLATILITY_FOCUS_TRACK_M: len(all_track_m_rows),
            VOLATILITY_FOCUS_TRACK_T: len(all_track_t_rows),
        },
        "volatility_focus_v9_track_counts": {
            VOLATILITY_FOCUS_TRACK_M9: len(all_track_m9_rows),
        },
        "terminal_line_failed": v9_terminal["terminal_line_failed"],
        "terminal_recommendation": v9_terminal["terminal_recommendation"],
        "exact_blockers": v9_terminal["exact_blockers"],
        "top_configs": ranked[:10],
        "rejection_reason_counts": {
            "research": rejection_reason_counts_for_field(ranked, "research_rejection_reasons"),
            "maker": rejection_reason_counts_for_field(ranked, "maker_rejection_reasons"),
            "paper_forward": rejection_reason_counts_for_field(ranked, "paper_forward_rejection_reasons"),
            "training": rejection_reason_counts_for_field(ranked, "training_rejection_reasons"),
            "combined": rejection_reason_counts(ranked),
        },
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "focused_summary_path": str(focused_summary_path),
        "top_configs_csv_path": str(top_configs_csv_path),
        "rejections_path": str(rejections_path),
        "trade_audits": trade_audit_paths,
        "notes": [
            "Volatility focus is offline research only.",
            "It does not enable trading, auto trading, fallback trading, shorting, leverage, or model promotion.",
            "research_rejection_reasons intentionally excludes active_model_invalid.",
        ],
    }
    return _json_safe(summary)


def write_volatility_focus_outputs(
    rows: list[dict[str, Any]],
    focused_summary: dict[str, Any],
    *,
    summary_path: Path,
    top_configs_csv_path: Path,
    rejections_path: Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    top_configs_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rejections_path.parent.mkdir(parents=True, exist_ok=True)
    rank_key = _maker_rank_sort_key if focused_summary.get("v9_track_configs") else _rank_sort_key
    ranked = sorted(rows, key=rank_key, reverse=True)
    top_rows = ranked[:50]
    top_fields = [
        "track_id",
        "parameter_set_id",
        "strategy_name",
        "timeframe",
        "exit_mode",
        "take_profit_pct",
        "stop_loss_pct",
        "max_hold_bars",
        "breakout_lookback",
        "consolidation_lookback",
        "min_body_vs_avg",
        "min_recent_return_pct",
        "min_trend_strength",
        "min_volume_zscore",
        "max_atr_expansion",
        "number_of_trades",
        "gross_return_pct",
        "current_taker_net_return_pct",
        "maker_current_net_return_pct",
        "maker_low_slippage_net_return_pct",
        "zero_cost_net_return_pct",
        "current_taker_profit_factor",
        "maker_current_profit_factor",
        "maker_low_slippage_profit_factor",
        "zero_cost_profit_factor",
        "net_return_pct",
        "profit_factor_net",
        "max_drawdown_pct",
        "win_rate_net",
        "expectancy",
        "fold_count",
        "average_mfe",
        "average_mae",
        "pct_trades_reaching_1r_before_stop",
        "pct_trades_reaching_2r_before_stop",
        "pct_trades_timing_out",
        "pct_trades_exiting_by_take_profit",
        "pct_trades_exiting_by_stop_loss",
        "pct_trades_exiting_by_protective_stop",
        "average_bars_held",
        "walk_forward_passed",
        "statistically_weak",
        "profit_factor_reliable",
        "single_trade_return_concentration",
        "folds_profitable_count",
        "folds_with_min_trades_count",
        "worst_fold_net_return_pct",
        "median_fold_net_return_pct",
        "per_fold_number_of_trades",
        "per_fold_net_return_pct",
        "per_fold_profit_factor_net",
        "fold_by_fold_returns",
        "fold_by_fold_trade_counts",
        "fold_by_fold_profit_factor",
        "beats_buy_hold_risk_adjusted",
        "beats_dca_daily_risk_adjusted",
        "cost_sensitivity_classification",
        "research_promising",
        "economically_viable",
        "maker_research_promising",
        "maker_economically_viable",
        "maker_only_candidate",
        "paper_forward_eligible",
        "estimated_fill_rate_required_to_remain_profitable",
        "maker_vs_taker_net_gap",
        "max_allowed_taker_fallback_rate_before_net_negative",
        "spread_bps_assumption",
        "slippage_bps_assumption",
        "no_market_fallback_required",
        "post_only_required",
        "unfilled_cancel_required",
        "research_rejection_reasons",
        "maker_rejection_reasons",
        "paper_forward_rejection_reasons",
        "training_rejection_reasons",
        "adjusted_rank_score",
    ]
    with top_configs_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=top_fields)
        writer.writeheader()
        for row in top_rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in top_fields})
    rejections = {
        "generated_at": focused_summary.get("generated_at"),
        "configs_tested": focused_summary.get("configs_tested"),
        "rejection_reason_counts": focused_summary.get("rejection_reason_counts"),
        "rejected_configs": [
            {
                "parameter_set_id": row.get("parameter_set_id"),
                "track_id": row.get("track_id"),
                "strategy_name": row.get("strategy_name"),
                "timeframe": row.get("timeframe"),
                "exit_mode": row.get("exit_mode"),
                "number_of_trades": row.get("number_of_trades"),
                "current_taker_net_return_pct": row.get("current_taker_net_return_pct"),
                "maker_current_net_return_pct": row.get("maker_current_net_return_pct"),
                "maker_low_slippage_net_return_pct": row.get("maker_low_slippage_net_return_pct"),
                "zero_cost_net_return_pct": row.get("zero_cost_net_return_pct"),
                "current_taker_profit_factor": row.get("current_taker_profit_factor"),
                "maker_current_profit_factor": row.get("maker_current_profit_factor"),
                "net_return_pct": row.get("net_return_pct"),
                "max_drawdown_pct": row.get("max_drawdown_pct"),
                "walk_forward_passed": row.get("walk_forward_passed"),
                "maker_research_promising": row.get("maker_research_promising"),
                "maker_economically_viable": row.get("maker_economically_viable"),
                "maker_only_candidate": row.get("maker_only_candidate"),
                "research_rejection_reasons": row.get("research_rejection_reasons"),
                "maker_rejection_reasons": row.get("maker_rejection_reasons"),
                "paper_forward_rejection_reasons": row.get("paper_forward_rejection_reasons"),
                "training_rejection_reasons": row.get("training_rejection_reasons"),
            }
            for row in ranked
            if row.get("research_rejection_reasons")
            or row.get("maker_rejection_reasons")
            or row.get("paper_forward_rejection_reasons")
            or row.get("training_rejection_reasons")
        ][:250],
    }
    summary_path.write_text(json.dumps(_json_safe(focused_summary), indent=2, allow_nan=False), encoding="utf-8")
    rejections_path.write_text(json.dumps(_json_safe(rejections), indent=2, allow_nan=False), encoding="utf-8")


def rejection_reason_counts_for_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        raw = str(row.get(field) or "")
        for reason in [part for part in raw.split(";") if part]:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def volatility_focus_v7_failure_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "primary_failure_mode": "no_configs_to_evaluate",
            "failure_counts": {},
        }
    reason_counts = rejection_reason_counts_for_field(rows, "research_rejection_reasons")
    failure_counts = {
        "trade_count": sum(
            count
            for reason, count in reason_counts.items()
            if reason.startswith("number_of_trades_below") or reason.startswith("folds_with_min_trades_below")
        ),
        "cost_drag": sum(
            count
            for reason, count in reason_counts.items()
            if reason in {"current_taker_net_return_not_positive", "current_taker_profit_factor_below_1_05"}
        ),
        "drawdown": reason_counts.get("max_drawdown_above_configured_limit", 0),
        "walk_forward_instability": reason_counts.get("walk_forward_not_passed", 0),
        "baseline": sum(
            count
            for reason, count in reason_counts.items()
            if reason in {"does_not_beat_buy_and_hold_risk_adjusted", "does_not_beat_dca_risk_adjusted"}
        ),
        "data_source": sum(
            count
            for reason, count in reason_counts.items()
            if reason in {"research_data_source_invalid", "data_source_not_collected_market_data", "synthetic_data_used"}
        ),
    }
    primary = max(failure_counts.items(), key=lambda item: (item[1], item[0]))[0]
    if all(count == 0 for count in failure_counts.values()):
        primary = "none_all_research_gates_passed"
    return {
        "primary_failure_mode": primary,
        "failure_counts": failure_counts,
        "research_rejection_reason_counts": reason_counts,
    }


def volatility_focus_diagnosis(rows: list[dict[str, Any]], *, research_result_valid: bool) -> dict[str, Any]:
    if not research_result_valid:
        return {
            "primary_failure_mode": "more_data_needed",
            "cost_drag_observed": False,
            "entry_quality_problem": False,
            "exit_logic_problem": False,
            "one_fold_dominates": False,
        }
    rows_20 = [row for row in rows if int(row.get("number_of_trades", 0) or 0) >= 20]
    zero_profitable = [
        row for row in rows if _metric_float((row.get("net_return_by_cost_scenario") or {}).get("zero_cost_sanity")) > 0
    ]
    current_profitable = [
        row for row in rows if _metric_float((row.get("net_return_by_cost_scenario") or {}).get("current_taker")) > 0
    ]
    avg_reach_1r = float(np.mean([_metric_float(row.get("pct_trades_reaching_1r_before_stop")) for row in rows_20])) if rows_20 else 0.0
    avg_timeout = float(np.mean([_metric_float(row.get("pct_trades_timing_out")) for row in rows_20])) if rows_20 else 0.0
    avg_stop = float(np.mean([_metric_float(row.get("pct_trades_exiting_by_stop_loss")) for row in rows_20])) if rows_20 else 0.0
    one_fold_dominates = any(
        max([_metric_float(value) for value in (row.get("fold_by_fold_returns") or [])], default=0.0)
        > max(0.0, _metric_float(row.get("net_return_pct"))) * 0.80
        for row in rows_20
        if _metric_float(row.get("net_return_pct")) > 0
    )
    cost_drag = bool(zero_profitable and not current_profitable)
    exit_problem = bool(rows_20 and avg_reach_1r >= 0.40 and avg_timeout >= 0.35)
    entry_problem = bool(rows_20 and avg_reach_1r < 0.30 and avg_stop >= 0.35)
    if not rows_20:
        primary = "more_data_needed"
    elif cost_drag:
        primary = "cost_drag_only"
    elif exit_problem:
        primary = "exit_logic_problem"
    elif entry_problem:
        primary = "entry_quality_problem"
    else:
        primary = "refine_volatility_breakout_more"
    return {
        "primary_failure_mode": primary,
        "cost_drag_observed": cost_drag,
        "entry_quality_problem": entry_problem,
        "exit_logic_problem": exit_problem,
        "one_fold_dominates": one_fold_dominates,
        "average_pct_reaching_1r_for_20_plus_trade_configs": avg_reach_1r,
        "average_timeout_pct_for_20_plus_trade_configs": avg_timeout,
        "average_stop_loss_exit_pct_for_20_plus_trade_configs": avg_stop,
    }


def volatility_focus_recommendation(
    rows: list[dict[str, Any]],
    *,
    research_result_valid: bool,
    synthetic_data_used: bool,
    diagnosis: dict[str, Any],
    max_focused_configs: int,
) -> str:
    if synthetic_data_used or not research_result_valid:
        return "more_data_needed"
    if any(row.get("research_promising") for row in rows):
        return "candidate_found_keep_trading_disabled"
    rows_20 = [row for row in rows if int(row.get("number_of_trades", 0) or 0) >= 20]
    if not rows_20:
        return "more_data_needed"
    primary = str(diagnosis.get("primary_failure_mode") or "")
    if primary in VOLATILITY_FOCUS_RECOMMENDATIONS:
        return primary
    zero_profitable = any(
        _metric_float((row.get("net_return_by_cost_scenario") or {}).get("zero_cost_sanity")) > 0
        for row in rows
    )
    if not zero_profitable and len(rows) >= int(max_focused_configs):
        return "retire_volatility_breakout"
    if not zero_profitable:
        return "no_edge_found"
    return "refine_volatility_breakout_more"


def reality_audit_recommendation(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    if bool(summary.get("synthetic_data_used")):
        return "insufficient_data"
    if not bool(summary.get("research_result_valid")):
        return "insufficient_data"
    htf_rows = [
        row
        for row in rows
        if row.get("timeframe") in {"1H", "4H", "1D"}
        and row.get("strategy_name") in set(HTF_STRATEGIES) | set(V3_STRATEGIES)
    ]
    if any(row.get("research_promising") for row in htf_rows):
        return "candidate_found_keep_trading_disabled"
    classifications = {
        str(row.get("cost_sensitivity_classification"))
        for row in rows
        if row.get("cost_sensitivity_classification")
    }
    if "signal_survives_current_taker_cost" in classifications or "signal_survives_maker_cost" in classifications:
        return "investigate_execution_model"
    if summary.get("fifteen_min_rejected") and htf_rows:
        return "move_to_higher_timeframe_research"
    if summary.get("fifteen_min_rejected"):
        return "reject_current_15min_strategies"
    return "no_edge_found"


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
            "trades_tested": sum(int(row.get("number_of_trades", 0) or 0) for row in strategy_rows),
            "take_profit_vs_cost_safe_count": sum(
                1 for row in strategy_rows if bool(row.get("take_profit_vs_cost_safe"))
            ),
            "profitable_current_cost_configs": sum(
                1 for row in strategy_rows if _metric_float(row.get("net_return_pct")) > 0
            ),
            "profitable_maker_cost_configs": sum(
                1
                for row in strategy_rows
                if _metric_float((row.get("net_return_by_cost_scenario") or {}).get("maker_current")) > 0
            ),
            "profitable_low_cost_configs": sum(
                1
                for row in strategy_rows
                if _metric_float((row.get("net_return_by_cost_scenario") or {}).get("maker_low_slippage")) > 0
            ),
            "profitable_zero_cost_configs": sum(
                1
                for row in strategy_rows
                if _metric_float((row.get("net_return_by_cost_scenario") or {}).get("zero_cost_sanity")) > 0
            ),
            "economically_viable_count": sum(1 for row in strategy_rows if row.get("parameter_set_id") in economic_ids),
            "economically_viable_configs": sum(1 for row in strategy_rows if row.get("parameter_set_id") in economic_ids),
            "research_promising_configs": sum(1 for row in strategy_rows if row.get("research_promising")),
            "paper_forward_eligible_count": sum(1 for row in strategy_rows if row.get("parameter_set_id") in eligible_ids),
            "beats_buy_hold_count": sum(1 for row in strategy_rows if row.get("beats_buy_hold_risk_adjusted")),
            "beats_dca_count": sum(1 for row in strategy_rows if row.get("beats_dca_daily_risk_adjusted")),
            "walk_forward_passed_count": sum(1 for row in strategy_rows if row.get("walk_forward_passed")),
            "best_net_return_pct": best.get("net_return_pct") if best else None,
            "best_profit_factor_net": best.get("profit_factor_net") if best else None,
            "best_max_drawdown_pct": best.get("max_drawdown_pct") if best else None,
            "best_config_current_cost": best_ranked_config(strategy_rows),
            "best_config_low_cost": best_config_for_cost_scenario(strategy_rows, "maker_low_slippage"),
            "best_config_zero_cost": best_config_for_cost_scenario(strategy_rows, "zero_cost_sanity"),
            "cost_sensitivity_classification_counts": value_counts(
                row.get("cost_sensitivity_classification") for row in strategy_rows
            ),
            "rejection_reason_counts": rejection_reason_counts(strategy_rows),
            **trade_summary,
        }
    return breakdown


def best_config_for_cost_scenario(rows: list[dict[str, Any]], scenario: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _metric_float((row.get("net_return_by_cost_scenario") or {}).get(scenario)),
            _profit_factor_value((row.get("profit_factor_by_cost_scenario") or {}).get(scenario)),
            -_metric_float((row.get("max_drawdown_by_cost_scenario") or {}).get(scenario)),
            int(row.get("number_of_trades", 0) or 0),
        ),
    )


def value_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_baselines_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    baselines: dict[str, dict[str, Any]] = {}
    for row in rows:
        timeframe = str(row.get("timeframe") or "unknown")
        if timeframe in baselines:
            continue
        baselines[timeframe] = {
            "cash_return_pct": 0.0,
            "buy_and_hold_return_pct": _metric_float(row.get("buy_and_hold_return_pct")),
            "buy_hold_max_drawdown_pct": _metric_float(row.get("buy_hold_max_drawdown_pct")),
            "buy_hold_return_over_drawdown": _metric_float(row.get("buy_hold_return_over_drawdown")),
            "dca_daily_return_pct": _metric_float(row.get("dca_daily_return_pct")),
            "dca_daily_max_drawdown_pct": _metric_float(row.get("dca_max_drawdown_pct")),
            "dca_daily_return_over_drawdown": _metric_float(row.get("dca_daily_return_over_drawdown")),
            "dca_weekly_return_pct": _metric_float(row.get("dca_weekly_return_pct")),
        }
    return baselines


def build_15min_rejection_summary(
    rows: list[dict[str, Any]],
    *,
    requested_timeframes: tuple[str, ...] | list[str] | None,
) -> dict[str, Any]:
    requested = set(requested_timeframes or [])
    fifteen_rows = [row for row in rows if row.get("timeframe") == "15Min"]
    family_map = {
        "buy_the_dip_rejected": BUY_THE_DIP_STRATEGY,
        "trend_pullback_rejected": TREND_PULLBACK_STRATEGY,
        "uptrend_pullback_rejected": UPTREND_PULLBACK_STRATEGY,
        "volatility_breakout_rejected": VOLATILITY_BREAKOUT_STRATEGY,
    }
    rejected_families: list[str] = []
    fields: dict[str, Any] = {}
    for field, strategy in family_map.items():
        strategy_rows = [row for row in fifteen_rows if row.get("strategy_name") == strategy]
        proven = any(
            row.get("research_promising")
            and row.get("walk_forward_passed")
            and row.get("beats_any_relevant_baseline_risk_adjusted", True)
            for row in strategy_rows
        )
        rejected = not proven
        fields[field] = rejected
        if rejected:
            rejected_families.append(f"15Min {strategy}")
    fifteen_min_rejected = bool(rejected_families)
    if "15Min" not in requested and not fifteen_rows:
        fifteen_min_rejected = True
        reason = "15Min_not_evaluated_in_this_run_prior_evidence_keeps_family_rejected"
    elif not fifteen_rows:
        reason = "no_15min_strategy_rows_evaluated"
    elif fifteen_min_rejected:
        reason_counts = rejection_reason_counts(fifteen_rows)
        reason = ";".join(reason_counts.keys()) or "no_15min_family_proved_research_promising"
    else:
        reason = None
    return {
        "fifteen_min_rejected": fifteen_min_rejected,
        "fifteen_min_rejection_reason": reason,
        "rejected_strategy_families": rejected_families,
        **fields,
    }


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


def best_maker_ranked_config(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=_maker_rank_sort_key)


def _volatility_focus_v9_mode(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("track_id") == VOLATILITY_FOCUS_TRACK_M9 for row in rows)


def volatility_focus_v9_track_configs(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return {
        "track_id": VOLATILITY_FOCUS_TRACK_M9,
        "anchor_parameter_set_id": VOLATILITY_FOCUS_V9_ANCHOR_PARAMETER_SET_ID,
        "anchor_max_drawdown_pct": VOLATILITY_FOCUS_V9_ANCHOR_MAX_DRAWDOWN_PCT,
        "configs_generated": len(rows),
        "search_space": VOLATILITY_FOCUS_V9_SEARCH_SPACE,
        "ranking_priority": list(VOLATILITY_FOCUS_V9_RANKING_PRIORITY),
        "scope": "final_narrow_1h_btc_usd_long_only_paper_research_drawdown_reduction",
    }


def volatility_focus_v9_terminal_state(
    rows: list[dict[str, Any]],
    settings: Settings,
    *,
    research_result_valid: bool,
    synthetic_data_used: bool,
) -> dict[str, Any]:
    if not rows:
        return {
            "terminal_line_failed": None,
            "terminal_recommendation": None,
            "exact_blockers": {},
        }
    passed = [row for row in rows if bool(row.get("maker_research_promising"))]
    exact_blockers = volatility_focus_v9_exact_blockers(
        rows,
        settings,
        research_result_valid=research_result_valid,
        synthetic_data_used=synthetic_data_used,
    )
    if passed:
        return {
            "terminal_line_failed": False,
            "terminal_recommendation": VOLATILITY_FOCUS_V9_TERMINAL_FOUND_RECOMMENDATION,
            "exact_blockers": exact_blockers,
        }
    return {
        "terminal_line_failed": True,
        "terminal_recommendation": VOLATILITY_FOCUS_V9_TERMINAL_FAILURE_RECOMMENDATION,
        "exact_blockers": exact_blockers,
    }


def volatility_focus_v9_exact_blockers(
    rows: list[dict[str, Any]],
    settings: Settings,
    *,
    research_result_valid: bool,
    synthetic_data_used: bool,
) -> dict[str, Any]:
    counts = {blocker: 0 for blocker in VOLATILITY_FOCUS_V9_TERMINAL_BLOCKERS}
    row_blockers: list[dict[str, Any]] = []
    for row in rows:
        blockers = volatility_focus_v9_blockers_for_row(
            row,
            settings,
            research_result_valid=research_result_valid,
            synthetic_data_used=synthetic_data_used,
        )
        for blocker in blockers:
            counts[blocker] = counts.get(blocker, 0) + 1
        row_blockers.append(
            {
                "parameter_set_id": row.get("parameter_set_id"),
                "track_id": row.get("track_id"),
                "number_of_trades": row.get("number_of_trades"),
                "maker_current_net_return_pct": row.get("maker_current_net_return_pct"),
                "maker_current_profit_factor": row.get("maker_current_profit_factor"),
                "max_drawdown_pct": row.get("max_drawdown_pct"),
                "walk_forward_passed": row.get("walk_forward_passed"),
                "folds_with_min_trades_count": row.get("folds_with_min_trades_count"),
                "blockers": blockers,
            }
        )
    best_row = best_maker_ranked_config(rows)
    best_under_drawdown = best_maker_ranked_config(
        [
            row
            for row in rows
            if int(row.get("number_of_trades", 0) or 0) >= MIN_RESEARCH_TRADES
            and _metric_float(row.get("max_drawdown_pct")) <= float(settings.max_backtest_drawdown_pct)
        ]
    )
    return {
        "configured_drawdown_limit_pct": float(settings.max_backtest_drawdown_pct),
        "required_min_trades": MIN_RESEARCH_TRADES,
        "required_profit_factor": MIN_RESEARCH_PROFIT_FACTOR_NET,
        "required_folds_with_min_trades": 4,
        "present": {blocker: counts.get(blocker, 0) > 0 for blocker in VOLATILITY_FOCUS_V9_TERMINAL_BLOCKERS},
        "reason_counts": {blocker: count for blocker, count in counts.items() if count},
        "best_v9_maker_candidate_blockers": (
            volatility_focus_v9_blockers_for_row(
                best_row,
                settings,
                research_result_valid=research_result_valid,
                synthetic_data_used=synthetic_data_used,
            )
            if best_row is not None
            else []
        ),
        "best_candidate_under_drawdown_limit_blockers": (
            volatility_focus_v9_blockers_for_row(
                best_under_drawdown,
                settings,
                research_result_valid=research_result_valid,
                synthetic_data_used=synthetic_data_used,
            )
            if best_under_drawdown is not None
            else ["no_candidate_under_configured_drawdown_limit"]
        ),
        "sample_rejected_rows": row_blockers[:10],
    }


def volatility_focus_v9_blockers_for_row(
    row: dict[str, Any],
    settings: Settings,
    *,
    research_result_valid: bool,
    synthetic_data_used: bool,
) -> list[str]:
    blockers: list[str] = []
    maker_net = _metric_float(row.get("maker_current_net_return_pct"))
    maker_pf = _profit_factor_value(row.get("maker_current_profit_factor"))
    trades = int(row.get("number_of_trades", 0) or 0)
    fold_count = int(row.get("fold_count", 4) or 4)
    folds_with_min_trades = int(row.get("folds_with_min_trades_count", 0) or 0)
    max_drawdown = _metric_float(row.get("max_drawdown_pct"))
    if max_drawdown > float(settings.max_backtest_drawdown_pct):
        blockers.append("max_drawdown_above_configured_limit")
    if maker_net <= 0:
        blockers.append("maker_current_net_return_not_positive")
    if maker_pf < MIN_RESEARCH_PROFIT_FACTOR_NET:
        blockers.append("maker_current_profit_factor_below_1_05")
    if trades < MIN_RESEARCH_TRADES:
        blockers.append("number_of_trades_below_20")
    if not bool(row.get("walk_forward_passed")):
        blockers.append("walk_forward_not_passed")
    if folds_with_min_trades < min(4, fold_count):
        blockers.append("folds_with_min_trades_below_required")
    if not bool(row.get("beats_buy_hold_risk_adjusted")):
        blockers.append("does_not_beat_buy_and_hold_risk_adjusted")
    if not bool(row.get("beats_dca_daily_risk_adjusted")):
        blockers.append("does_not_beat_dca_risk_adjusted")
    if bool(row.get("statistically_weak")) or trades < MIN_RESEARCH_TRADES:
        blockers.append("statistically_weak")
    source_invalid = (
        not research_result_valid
        or synthetic_data_used
        or bool(row.get("synthetic_data_used"))
        or not bool(row.get("research_result_valid", True))
        or not _is_collected_market_data_source(str(row.get("source_used") or "unknown"))
    )
    if source_invalid:
        blockers.append("invalid_data_source")
    for reason in _semicolon_values(row.get("maker_rejection_reasons")):
        mapped = _volatility_focus_v9_terminal_blocker_from_reason(reason)
        if mapped:
            blockers.append(mapped)
    return list(dict.fromkeys(blockers))


def _volatility_focus_v9_terminal_blocker_from_reason(reason: str) -> str | None:
    if reason in VOLATILITY_FOCUS_V9_TERMINAL_BLOCKERS:
        return reason
    if reason == "folds_with_min_trades_not_all_4":
        return "folds_with_min_trades_below_required"
    if reason in {"research_data_source_invalid", "data_source_not_collected_market_data", "synthetic_data_used"}:
        return "invalid_data_source"
    if reason.startswith("number_of_trades_below"):
        return "number_of_trades_below_20"
    return None


def _semicolon_values(value: Any) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


def _rank_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row.get("economically_viable")),
        not bool(row.get("statistically_weak")),
        _metric_float(row.get("adjusted_rank_score", row.get("rank_score"))),
        int(row.get("number_of_trades", 0) or 0),
        _metric_float(row.get("net_return_pct")),
        min(5.0, _profit_factor_value(row.get("profit_factor_net"))),
    )


def _maker_rank_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    trades = int(row.get("number_of_trades", 0) or 0)
    maker_net = _metric_float(row.get("maker_current_net_return_pct"))
    maker_pf = _profit_factor_value(row.get("maker_current_profit_factor"))
    fold_count = int(row.get("fold_count", 4) or 4)
    folds_with_min_trades = int(row.get("folds_with_min_trades_count", 0) or 0)
    concentration = _metric_float(row.get("single_trade_return_concentration"))
    drawdown = _metric_float(row.get("max_drawdown_pct"))
    maker_reasons = set(_semicolon_values(row.get("maker_rejection_reasons")))
    return (
        maker_net > 0,
        maker_pf >= MIN_RESEARCH_PROFIT_FACTOR_NET,
        trades >= MIN_RESEARCH_TRADES,
        bool(row.get("walk_forward_passed")),
        folds_with_min_trades >= min(4, fold_count),
        "max_drawdown_above_configured_limit" not in maker_reasons,
        bool(row.get("beats_buy_hold_risk_adjusted")),
        bool(row.get("beats_dca_daily_risk_adjusted")),
        -concentration,
        _metric_float(row.get("median_fold_net_return_pct")),
        _metric_float(row.get("worst_fold_net_return_pct")),
        maker_net,
        min(5.0, maker_pf),
        -drawdown,
        trades,
        bool(row.get("maker_research_promising")),
        bool(row.get("maker_economically_viable")),
    )


def volatility_focus_blocker_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    counts = rejection_reason_counts_for_field(rows, field)
    if not counts:
        return {
            "primary_blocker": None,
            "reason_counts": {},
            "configs_blocked": 0,
        }
    primary, _ = max(counts.items(), key=lambda item: (item[1], item[0]))
    return {
        "primary_blocker": primary,
        "reason_counts": counts,
        "configs_blocked": sum(1 for row in rows if str(row.get(field) or "")),
    }


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


def export_trade_audit_logs(
    rows: list[dict[str, Any]],
    bars_by_timeframe: dict[str, pd.DataFrame],
    settings: Settings,
    *,
    data_source_reports: dict[str, ResearchDataReport | dict[str, Any]],
    strategy: str,
    max_buy_dip_configs: int,
    max_v3_configs: int,
    walk_forward_splits: int,
    min_trades_per_split: int,
    timeframes: tuple[str, ...] | list[str],
    output_dir: Path,
    top_n: int,
    include_rejected: bool,
    filename_prefix: str = "",
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_key = _maker_rank_sort_key if strategy == VOLATILITY_FOCUS_STRATEGY and _volatility_focus_v9_mode(rows) else _rank_sort_key
    ranked = sorted(rows, key=rank_key, reverse=True)
    preferred = [
        row
        for row in ranked
        if row.get("paper_forward_eligible") or row.get("economically_viable") or row.get("research_promising")
    ]
    selected_rows = (ranked if include_rejected else preferred)[:top_n]
    if not selected_rows:
        selected_rows = ranked[:top_n]
    configs = {
        config.parameter_set_id: config
        for config in generate_research_configs(
            strategy=strategy,
            max_buy_dip_configs=max_buy_dip_configs,
            max_v3_configs=max_v3_configs,
            timeframes=timeframes,
        )
    }
    v3_features = {
        timeframe: prepare_v3_features(bars)
        for timeframe, bars in bars_by_timeframe.items()
        if not bars.empty
    }
    buy_the_dip_features = {
        timeframe: prepare_buy_the_dip_features(bars)
        for timeframe, bars in bars_by_timeframe.items()
        if not bars.empty
    }
    audit_outputs: list[dict[str, Any]] = []
    for selected_index, row in enumerate(selected_rows, start=1):
        config = configs.get(str(row.get("parameter_set_id")))
        if config is None:
            continue
        bars = bars_by_timeframe.get(config.timeframe, pd.DataFrame())
        candidate_settings = research_config_settings(settings, config)
        if bars.empty or config.strategy_name == HTF_RISK_OFF_HOLD_FILTER_STRATEGY:
            trades = pd.DataFrame()
            signals = pd.DataFrame()
            metrics = calculate_fee_aware_metrics(trades, candidate_settings, signal_frame=signals)
        else:
            trades, signals = build_strategy_research_trades(
                config,
                bars=bars,
                settings=candidate_settings,
                buy_the_dip_features=buy_the_dip_features.get(config.timeframe, pd.DataFrame()),
                v3_features=v3_features.get(config.timeframe, pd.DataFrame()),
            )
            metrics = calculate_fee_aware_metrics(trades, candidate_settings, signal_frame=signals)
        folds = chronological_walk_forward_splits(bars, splits=walk_forward_splits)
        audit_rows = build_trade_audit_rows(
            config,
            trades,
            metrics,
            result_row=row,
            settings=candidate_settings,
            folds=folds,
        )
        diagnostics = aggregate_trade_diagnostics(audit_rows)
        stem = _safe_filename(
            f"{filename_prefix}{selected_index:02d}_{config.timeframe}_{config.strategy_name}_{config.parameter_set_id}"
        )
        csv_path = output_dir / f"{stem}.csv"
        jsonl_path = output_dir / f"{stem}.jsonl"
        write_trade_audit_files(audit_rows, csv_path=csv_path, jsonl_path=jsonl_path)
        audit_outputs.append(
            {
                "strategy_name": config.strategy_name,
                "parameter_set_id": config.parameter_set_id,
                "timeframe": config.timeframe,
                "csv_path": str(csv_path),
                "jsonl_path": str(jsonl_path),
                "aggregate_diagnostics": diagnostics,
                "rejection_reasons": row.get("rejection_reasons"),
                "walk_forward_splits": walk_forward_splits,
                "min_trades_per_split": min_trades_per_split,
                "source_used": (
                    _coerce_data_report(config.timeframe, data_source_reports[config.timeframe]).source_used
                    if config.timeframe in data_source_reports
                    else row.get("source_used")
                ),
            }
        )
    diagnostics_path = output_dir / "trade_audit_diagnostics.json"
    diagnostics_path.write_text(json.dumps(_json_safe(audit_outputs), indent=2, allow_nan=False), encoding="utf-8")
    return audit_outputs


def build_trade_audit_rows(
    config: ResearchConfig,
    trades: pd.DataFrame,
    metrics: dict[str, Any],
    *,
    result_row: dict[str, Any],
    settings: Settings,
    folds: list[pd.DataFrame],
) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    details = list(metrics.get("trade_details", []) or [])
    notional = max(1e-12, float(settings.order_notional_usd))
    audit_rows: list[dict[str, Any]] = []
    for index, (_, trade_row) in enumerate(trades.iterrows()):
        detail = details[index] if index < len(details) and isinstance(details[index], dict) else {}
        gross_return = _metric_float(detail.get("gross_return_pct", trade_row.get("buy_exit_return_pct")))
        net_return = _metric_float(detail.get("net_return_pct", gross_return))
        fee_cost = _metric_float(detail.get("fee_amount", detail.get("fees"))) / notional
        slippage_cost = _metric_float(detail.get("slippage_amount", detail.get("slippage"))) / notional
        spread_cost = _metric_float(detail.get("spread_cost")) / notional
        total_cost = fee_cost + slippage_cost + spread_cost
        entry_timestamp = trade_row.get("entry_timestamp", trade_row.get("timestamp"))
        audit_rows.append(
            {
                "strategy_name": config.strategy_name,
                "parameter_set_id": config.parameter_set_id,
                "track_id": result_row.get("track_id"),
                "timeframe": config.timeframe,
                "exit_mode": result_row.get("exit_mode"),
                "entry_timestamp": _timestamp_to_iso(entry_timestamp),
                "exit_timestamp": _timestamp_to_iso(trade_row.get("exit_timestamp")),
                "entry_price": _metric_float(trade_row.get("entry_price", trade_row.get("close"))),
                "exit_price": _metric_float(trade_row.get("exit_price")),
                "gross_return_pct": gross_return,
                "net_return_pct": net_return,
                "fee_cost_pct": fee_cost,
                "slippage_cost_pct": slippage_cost,
                "spread_cost_pct": spread_cost,
                "total_cost_pct": total_cost,
                "exit_reason": str(detail.get("exit_reason") or trade_row.get("buy_exit_reason") or "unknown"),
                "max_favorable_excursion_pct": _metric_float(trade_row.get("max_favorable_excursion_pct")),
                "max_adverse_excursion_pct": _metric_float(trade_row.get("max_adverse_excursion_pct")),
                "bars_held": _metric_float(detail.get("hold_bars", trade_row.get("buy_hold_bars"))),
                "signal_features_at_entry": signal_features_at_entry(trade_row),
                "walk_forward_fold": walk_forward_fold_for_timestamp(entry_timestamp, folds),
                "regime_label": trade_row.get("regime"),
                "was_gross_winner": gross_return > 0,
                "was_net_winner": net_return > 0,
                "gross_winner_became_net_loser": gross_return > 0 and net_return <= 0,
                "rejection_reasons": result_row.get("rejection_reasons"),
            }
        )
    return audit_rows


def aggregate_trade_diagnostics(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gross_returns = np.asarray([_metric_float(row.get("gross_return_pct")) for row in audit_rows], dtype=float)
    net_returns = np.asarray([_metric_float(row.get("net_return_pct")) for row in audit_rows], dtype=float)
    costs = np.asarray([_metric_float(row.get("total_cost_pct")) for row in audit_rows], dtype=float)
    bars_held = np.asarray([_metric_float(row.get("bars_held")) for row in audit_rows], dtype=float)
    return {
        "total_trades": int(len(audit_rows)),
        "gross_winners": int((gross_returns > 0).sum()) if len(gross_returns) else 0,
        "net_winners": int((net_returns > 0).sum()) if len(net_returns) else 0,
        "gross_winners_became_net_losers": int(
            sum(1 for row in audit_rows if row.get("gross_winner_became_net_loser"))
        ),
        "avg_gross_return_pct": float(gross_returns.mean()) if len(gross_returns) else 0.0,
        "avg_net_return_pct": float(net_returns.mean()) if len(net_returns) else 0.0,
        "median_gross_return_pct": float(np.median(gross_returns)) if len(gross_returns) else 0.0,
        "median_net_return_pct": float(np.median(net_returns)) if len(net_returns) else 0.0,
        "avg_total_cost_pct": float(costs.mean()) if len(costs) else 0.0,
        "median_total_cost_pct": float(np.median(costs)) if len(costs) else 0.0,
        "avg_bars_held": float(bars_held.mean()) if len(bars_held) else 0.0,
        "median_bars_held": float(np.median(bars_held)) if len(bars_held) else 0.0,
        "largest_winner_pct": float(net_returns.max()) if len(net_returns) else 0.0,
        "largest_loser_pct": float(net_returns.min()) if len(net_returns) else 0.0,
        "worst_drawdown_pct": max_drawdown_from_trade_returns(net_returns),
        "trades_by_exit_reason": value_counts(row.get("exit_reason") for row in audit_rows),
        "trades_by_fold": value_counts(row.get("walk_forward_fold") for row in audit_rows),
        "trades_by_regime": value_counts(row.get("regime_label") for row in audit_rows),
    }


def write_trade_audit_files(
    audit_rows: list[dict[str, Any]],
    *,
    csv_path: Path,
    jsonl_path: Path,
) -> None:
    fields = [
        "strategy_name",
        "parameter_set_id",
        "timeframe",
        "entry_timestamp",
        "track_id",
        "exit_mode",
        "exit_timestamp",
        "entry_price",
        "exit_price",
        "gross_return_pct",
        "net_return_pct",
        "fee_cost_pct",
        "slippage_cost_pct",
        "spread_cost_pct",
        "total_cost_pct",
        "exit_reason",
        "max_favorable_excursion_pct",
        "max_adverse_excursion_pct",
        "bars_held",
        "signal_features_at_entry",
        "walk_forward_fold",
        "regime_label",
        "was_gross_winner",
        "was_net_winner",
        "gross_winner_became_net_loser",
        "rejection_reasons",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in audit_rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(_json_safe(row), allow_nan=False, sort_keys=True) + "\n")


def signal_features_at_entry(row: pd.Series) -> dict[str, Any]:
    excluded = {
        "entry_timestamp",
        "exit_timestamp",
        "entry_price",
        "exit_price",
        "buy_quality_label",
        "buy_exit_return_pct",
        "buy_exit_reason",
        "buy_hold_bars",
        "backtest_exit_high",
        "backtest_exit_low",
        "exit_index",
    }
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if key in excluded:
            continue
        if key in {"timestamp", "close", "strategy_name", "entry_reason", "regime"} or key.startswith(
            (
                "rsi",
                "rolling",
                "vwap",
                "ema",
                "volume",
                "atr",
                "pullback",
                "body",
                "support",
                "confirmation",
                "breakout",
                "consolidation",
                "recent",
                "close_position",
                "require_",
                "research",
                "quant",
                "strategy_",
                "ml_",
            )
        ):
            payload[str(key)] = _json_safe(value)
    return payload


def walk_forward_fold_for_timestamp(timestamp: Any, folds: list[pd.DataFrame]) -> int | None:
    if timestamp is None or pd.isna(timestamp):
        return None
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")
    for index, fold in enumerate(folds):
        if fold.empty or "timestamp" not in fold:
            continue
        start = pd.Timestamp(fold["timestamp"].iloc[0])
        end = pd.Timestamp(fold["timestamp"].iloc[-1])
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        if start <= value <= end:
            return index
    return None


def _timestamp_to_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


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
    allowed_derivation_source_timeframes: tuple[str, ...] | list[str] | set[str] | None = None,
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
    if result.report.research_result_valid:
        return result
    if timeframe not in DERIVATION_SOURCES_BY_TIMEFRAME:
        return result

    rejected_sources: list[dict[str, Any]] = [
        *result.report.rejected_sources,
        _data_report_to_rejected_source(result.report),
    ]
    source_bars = pd.DataFrame()
    source_report: ResearchDataReport | None = None
    source_timeframe: str | None = None
    candidate_sources = DERIVATION_SOURCES_BY_TIMEFRAME[timeframe]
    if allowed_derivation_source_timeframes is not None:
        allowed_sources = set(allowed_derivation_source_timeframes)
        candidate_sources = tuple(source for source in candidate_sources if source in allowed_sources)
    if not candidate_sources:
        return result
    for candidate_source in candidate_sources:
        ratio = _source_bars_per_target_bar(candidate_source, timeframe)
        candidate_limit = max(int(row_limits.get(candidate_source, limit * ratio)), limit * ratio)
        if candidate_source in existing_bars_by_timeframe and candidate_source in existing_reports_by_timeframe:
            candidate_bars = existing_bars_by_timeframe[candidate_source]
            candidate_report = existing_reports_by_timeframe[candidate_source]
        elif candidate_source in DERIVATION_SOURCES_BY_TIMEFRAME:
            candidate_result = await _fetch_or_derive_research_bars(
                client,
                settings,
                timeframe=candidate_source,
                limit=candidate_limit,
                existing_bars_by_timeframe=existing_bars_by_timeframe,
                existing_reports_by_timeframe=existing_reports_by_timeframe,
                row_limits=row_limits,
                session_factory=session_factory,
                allow_synthetic_fallback=allow_synthetic_fallback,
                now=now,
                start=start,
                end=end,
                allowed_derivation_source_timeframes=allowed_derivation_source_timeframes,
            )
            candidate_bars = candidate_result.bars
            candidate_report = candidate_result.report
            existing_bars_by_timeframe[candidate_source] = candidate_bars
            existing_reports_by_timeframe[candidate_source] = candidate_report
        else:
            candidate_result = await _fetch_research_bars(
                client,
                settings,
                timeframe=candidate_source,
                limit=candidate_limit,
                session_factory=session_factory,
                allow_synthetic_fallback=allow_synthetic_fallback,
                now=now,
                start=start,
                end=end,
            )
            candidate_bars = candidate_result.bars
            candidate_report = candidate_result.report
            existing_bars_by_timeframe[candidate_source] = candidate_bars
            existing_reports_by_timeframe[candidate_source] = candidate_report
        rejected_sources.append(
            {
                "source": candidate_report.source_used,
                "timeframe": candidate_source,
                "status": "candidate_for_derivation",
                "reason": candidate_report.rejection_reason,
                "latest_timestamp": candidate_report.latest_timestamp,
                "data_age_minutes": candidate_report.data_age_minutes,
                "row_count": candidate_report.row_count,
            }
        )
        if candidate_report.research_result_valid and not candidate_report.synthetic_data_used:
            source_bars = candidate_bars
            source_report = candidate_report
            source_timeframe = candidate_source
            break

    if source_report is None or source_timeframe is None:
        return ResearchBarsResult(
            pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
            _with_rejected_sources(result.report, rejected_sources),
        )

    derived = derive_higher_timeframe_bars(
        source_bars,
        source_timeframe=source_timeframe,
        target_timeframe=timeframe,
        limit=limit,
    )
    current_time = _utc_timestamp(now or datetime.now(UTC))
    min_rows = _minimum_research_rows(settings, limit)
    derived_report = _assess_research_bars(
        derived,
        source_used=_derived_source_used(source_report.source_used, source_timeframe),
        timeframe=timeframe,
        min_rows=min_rows,
        now=current_time,
        synthetic_data_used=bool(source_report.synthetic_data_used),
        force_invalid_reason=(
            None
            if source_report.research_result_valid and not source_report.synthetic_data_used
            else source_report.rejection_reason or f"source_{source_timeframe}_invalid_for_{timeframe}_derivation"
        ),
        available_rows=int(len(derived)),
        requested_max_rows=limit,
        requested_start=start.isoformat() if start else None,
        requested_end=_utc_timestamp(end or current_time).isoformat(),
        derived_from_timeframe=source_timeframe,
    )
    return ResearchBarsResult(
        derived,
        _with_rejected_sources(
            derived_report,
            [
                *rejected_sources,
                {
                    "source": "collected_market_data",
                    "timeframe": source_timeframe,
                    "status": f"used_for_{timeframe}_derivation" if source_report.research_result_valid else "rejected",
                    "reason": source_report.rejection_reason,
                    "latest_timestamp": source_report.latest_timestamp,
                    "data_age_minutes": source_report.data_age_minutes,
                    "row_count": source_report.row_count,
                },
            ],
        ),
    )


def derive_1h_bars_from_15min(bars: pd.DataFrame, *, limit: int | None = None) -> pd.DataFrame:
    return derive_higher_timeframe_bars(
        bars,
        source_timeframe="15Min",
        target_timeframe="1H",
        limit=limit,
    )


def derive_4h_bars_from_lower_timeframe(
    bars: pd.DataFrame,
    *,
    source_timeframe: str = "1H",
    limit: int | None = None,
) -> pd.DataFrame:
    return derive_higher_timeframe_bars(
        bars,
        source_timeframe=source_timeframe,
        target_timeframe="4H",
        limit=limit,
    )


def derive_1d_bars_from_lower_timeframe(
    bars: pd.DataFrame,
    *,
    source_timeframe: str = "1H",
    limit: int | None = None,
) -> pd.DataFrame:
    return derive_higher_timeframe_bars(
        bars,
        source_timeframe=source_timeframe,
        target_timeframe="1D",
        limit=limit,
    )


def derive_higher_timeframe_bars(
    bars: pd.DataFrame,
    *,
    source_timeframe: str,
    target_timeframe: str,
    limit: int | None = None,
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    data = normalize_ohlcv(bars)
    if data.empty:
        return data
    expected_source_bars = _source_bars_per_target_bar(source_timeframe, target_timeframe)
    freq = _pandas_group_frequency(target_timeframe)
    grouped = data.assign(period_start=data["timestamp"].dt.floor(freq)).groupby("period_start", sort=True)
    rows: list[dict[str, Any]] = []
    for period_start, group in grouped:
        ordered = group.sort_values("timestamp")
        if len(ordered) < expected_source_bars:
            continue
        ordered = ordered.tail(expected_source_bars)
        rows.append(
            {
                "timestamp": period_start,
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


def _source_bars_per_target_bar(source_timeframe: str, target_timeframe: str) -> int:
    source_seconds = parse_timeframe_duration(_market_data_timeframe(source_timeframe)).total_seconds()
    target_seconds = parse_timeframe_duration(_market_data_timeframe(target_timeframe)).total_seconds()
    if source_seconds <= 0 or target_seconds <= 0 or target_seconds < source_seconds:
        raise ValueError(f"Cannot derive {target_timeframe} bars from {source_timeframe} bars")
    ratio = target_seconds / source_seconds
    if not float(ratio).is_integer():
        raise ValueError(f"{source_timeframe} does not divide evenly into {target_timeframe}")
    return int(ratio)


def _derived_source_used(source_used: str, source_timeframe: str) -> str:
    return f"{source_used}_derived_from_{source_timeframe.lower()}"


def _is_collected_market_data_source(source_used: str | None) -> bool:
    return bool(
        source_used == "collected_market_data"
        or (isinstance(source_used, str) and source_used.startswith("collected_market_data_derived_from_"))
    )


def _pandas_group_frequency(timeframe: str) -> str:
    if timeframe == "1H":
        return "1h"
    if timeframe == "4H":
        return "4h"
    if timeframe == "1D":
        return "1D"
    raise ValueError(f"Unsupported derived timeframe: {timeframe}")


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
        "research_track_id": config.track_id,
        "research_exit_mode": config.exit_mode,
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
        "research_track_id": config.track_id,
        "research_exit_mode": config.exit_mode,
        "rsi_14": float(row.get("rsi_14", 0.0)),
        "volume_zscore_20": float(row.get("volume_zscore_20", 0.0)),
        "atr_expansion_20": float(row.get("atr_expansion_20", 0.0)),
        "pullback_from_high_50": float(row.get("pullback_from_high_50", 0.0)),
        "body_vs_avg_20": float(row.get("body_vs_avg_20", 0.0)),
        "close_position_in_candle": float(row.get("close_position_in_candle", 0.0)),
        "breakout_candle_atr_multiple": float(row.get("breakout_candle_atr_multiple", 0.0)),
        "recent_runup_pct_5": float(row.get("recent_runup_pct_5", 0.0)),
        "atr_percentile_200": float(row.get("atr_percentile_200", 0.0)),
        "ema_20_slope_5": float(row.get("ema_20_slope_5", 0.0)),
        "ema_50_slope_5": float(row.get("ema_50_slope_5", 0.0)),
        "close_above_ema_200": bool(row.get("close_above_ema_200", False)),
        "ema_50_above_200": bool(row.get("ema_50_above_200", False)),
        "support": config.support,
        "confirmation": config.confirmation,
        "breakout_lookback": config.breakout_lookback,
        "consolidation_lookback": config.consolidation_lookback,
        "require_ema_trend_filter": bool(config.require_ema_trend_filter),
        "require_positive_ema20_slope": bool(config.require_positive_ema20_slope),
        "require_close_above_ema200": bool(config.require_close_above_ema200),
        "require_volume_expansion": bool(config.require_volume_expansion),
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


def _exit_result(
    gross_return: float,
    exit_reason: str,
    hold_bars: int,
    exit_high: float,
    exit_low: float,
    *,
    entry_close: float | None = None,
    exit_timestamp: Any | None = None,
    exit_index: int | None = None,
    window: pd.DataFrame | None = None,
) -> dict[str, Any]:
    entry = float(entry_close or 0.0)
    max_favorable = 0.0
    max_adverse = 0.0
    if entry > 0 and window is not None and not window.empty:
        max_favorable = float(window["high"].max() / entry - 1)
        max_adverse = float(window["low"].min() / entry - 1)
    return {
        "gross_return": float(gross_return),
        "exit_reason": exit_reason,
        "hold_bars": int(hold_bars),
        "exit_high": float(exit_high),
        "exit_low": float(exit_low),
        "exit_timestamp": exit_timestamp,
        "exit_index": exit_index,
        "exit_price": (entry * (1 + float(gross_return))) if entry > 0 else None,
        "max_favorable_excursion_pct": max_favorable,
        "max_adverse_excursion_pct": max_adverse,
    }


def _trade_audit_entry_exit_fields(row: pd.Series, exit_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry_timestamp": row.get("timestamp"),
        "exit_timestamp": exit_result.get("exit_timestamp"),
        "entry_price": float(row.get("close")),
        "exit_price": exit_result.get("exit_price"),
        "max_favorable_excursion_pct": float(exit_result.get("max_favorable_excursion_pct", 0.0) or 0.0),
        "max_adverse_excursion_pct": float(exit_result.get("max_adverse_excursion_pct", 0.0) or 0.0),
        "exit_index": exit_result.get("exit_index"),
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
        "track_id",
        "parameter_set_id",
        "strategy_name",
        "timeframe",
        "exit_mode",
        "take_profit_pct",
        "stop_loss_pct",
        "max_hold_bars",
        "number_of_trades",
        "current_taker_net_return_pct",
        "maker_current_net_return_pct",
        "maker_low_slippage_net_return_pct",
        "zero_cost_net_return_pct",
        "current_taker_profit_factor",
        "maker_current_profit_factor",
        "maker_low_slippage_profit_factor",
        "zero_cost_profit_factor",
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
        "maker_research_promising",
        "maker_economically_viable",
        "maker_only_candidate",
        "fallback_prediction_used",
        "active_model_valid",
        "economically_viable",
        "paper_forward_eligible",
        "estimated_fill_rate_required_to_remain_profitable",
        "maker_vs_taker_net_gap",
        "max_allowed_taker_fallback_rate_before_net_negative",
        "spread_bps_assumption",
        "slippage_bps_assumption",
        "no_market_fallback_required",
        "post_only_required",
        "unfilled_cancel_required",
        "rejection_reasons",
        "maker_rejection_reasons",
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
    if isinstance(value, datetime):
        return _utc_timestamp(value).isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
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
