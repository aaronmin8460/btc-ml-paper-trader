from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


BUY_THE_DIP_ENTRY_LABEL = "buy_the_dip_entry_label"
BUY_THE_DIP_EXIT_LABEL = "buy_the_dip_exit_label"
BUY_THE_DIP_EXIT_RETURN_PCT = "buy_the_dip_exit_return_pct"
BUY_THE_DIP_EXIT_REASON = "buy_the_dip_exit_reason"
BUY_THE_DIP_HOLD_BARS = "buy_the_dip_hold_bars"
BUY_THE_DIP_TIMEFRAME = "buy_the_dip_label_timeframe"
BUY_THE_DIP_CHRONOLOGICAL = "buy_the_dip_chronological_order_preserved"
BUY_THE_DIP_ALLOWED_TIMEFRAMES = {"5Min", "15Min"}


@dataclass(frozen=True)
class BuyTheDipLabelConfig:
    timeframe: str
    take_profit_pct: float
    stop_loss_pct: float
    max_hold_bars: int
    round_trip_estimated_cost_pct: float
    promotion_required_return_pct: float
    number_of_research_trades: int = 0
    minimum_research_trades: int = 20
    economically_viable_config: bool = False
    synthetic_data_used: bool = False
    source_used: str = "collected_market_data"
    data_age_minutes: float | None = None
    max_data_age_minutes: float = 30.0


def validate_buy_the_dip_label_config(config: BuyTheDipLabelConfig) -> list[str]:
    reasons: list[str] = []
    if config.timeframe not in BUY_THE_DIP_ALLOWED_TIMEFRAMES:
        reasons.append("unsupported_buy_the_dip_timeframe")
    if float(config.take_profit_pct) <= float(config.round_trip_estimated_cost_pct):
        reasons.append("take_profit_not_above_round_trip_cost")
    if float(config.take_profit_pct) < float(config.promotion_required_return_pct):
        reasons.append("take_profit_below_promotion_required_return")
    if float(config.stop_loss_pct) <= 0:
        reasons.append("stop_loss_not_positive")
    if int(config.max_hold_bars) <= 0:
        reasons.append("max_hold_bars_not_positive")
    if int(config.number_of_research_trades) < int(config.minimum_research_trades):
        reasons.append("number_of_trades_below_minimum")
    if not bool(config.economically_viable_config):
        reasons.append("no_economically_viable_research_config")
    if bool(config.synthetic_data_used):
        reasons.append("synthetic_data_not_allowed")
    if config.source_used != "collected_market_data":
        reasons.append("data_source_not_collected_market_data")
    if config.data_age_minutes is not None and float(config.data_age_minutes) > float(config.max_data_age_minutes):
        reasons.append("stale_data")
    return reasons


def generate_buy_the_dip_labels(
    bars: pd.DataFrame,
    config: BuyTheDipLabelConfig,
    *,
    allow_unvalidated: bool = False,
) -> pd.DataFrame:
    rejection_reasons = validate_buy_the_dip_label_config(config)
    if rejection_reasons and not allow_unvalidated:
        raise ValueError(";".join(rejection_reasons))
    if bars.empty:
        return _empty_label_frame(bars)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"missing_required_columns:{','.join(sorted(missing))}")

    out = bars.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    _validate_ohlcv(out)

    entry_labels: list[int | float] = []
    exit_labels: list[int | float] = []
    exit_returns: list[float] = []
    exit_reasons: list[str | None] = []
    hold_bars: list[int | float] = []

    horizon = int(config.max_hold_bars)
    take_profit_pct = float(config.take_profit_pct)
    stop_loss_pct = abs(float(config.stop_loss_pct))
    for index in range(len(out)):
        if index + horizon >= len(out):
            entry_labels.append(np.nan)
            exit_labels.append(np.nan)
            exit_returns.append(np.nan)
            exit_reasons.append(None)
            hold_bars.append(np.nan)
            continue
        result = _future_exit_result(out, index, take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct, horizon=horizon)
        entry_labels.append(1 if result["return_pct"] > 0 else 0)
        exit_labels.append(0 if result["return_pct"] > 0 else 1)
        exit_returns.append(float(result["return_pct"]))
        exit_reasons.append(str(result["reason"]))
        hold_bars.append(int(result["hold_bars"]))

    out[BUY_THE_DIP_ENTRY_LABEL] = entry_labels
    out[BUY_THE_DIP_EXIT_LABEL] = exit_labels
    out[BUY_THE_DIP_EXIT_RETURN_PCT] = exit_returns
    out[BUY_THE_DIP_EXIT_REASON] = exit_reasons
    out[BUY_THE_DIP_HOLD_BARS] = hold_bars
    out[BUY_THE_DIP_TIMEFRAME] = config.timeframe
    out[BUY_THE_DIP_CHRONOLOGICAL] = True
    return out.dropna(subset=[BUY_THE_DIP_ENTRY_LABEL, BUY_THE_DIP_EXIT_LABEL]).reset_index(drop=True)


def _future_exit_result(
    out: pd.DataFrame,
    index: int,
    *,
    take_profit_pct: float,
    stop_loss_pct: float,
    horizon: int,
) -> dict[str, Any]:
    entry = float(out.loc[index, "close"])
    take_profit = entry * (1 + take_profit_pct)
    stop_loss = entry * (1 - stop_loss_pct)
    max_index = min(len(out) - 1, index + horizon)
    for offset, row_index in enumerate(range(index + 1, max_index + 1), start=1):
        row = out.loc[row_index]
        high = float(row["high"])
        low = float(row["low"])
        hit_take_profit = high >= take_profit
        hit_stop_loss = low <= stop_loss
        if hit_take_profit and hit_stop_loss:
            return {"return_pct": -stop_loss_pct, "reason": "ambiguous_stop_first", "hold_bars": offset}
        if hit_stop_loss:
            return {"return_pct": -stop_loss_pct, "reason": "buy_the_dip_stop_loss", "hold_bars": offset}
        if hit_take_profit:
            return {"return_pct": take_profit_pct, "reason": "buy_the_dip_take_profit", "hold_bars": offset}
    exit_close = float(out.loc[max_index, "close"])
    return {
        "return_pct": (exit_close / entry) - 1,
        "reason": "buy_the_dip_max_hold",
        "hold_bars": max(1, max_index - index),
    }


def _validate_ohlcv(out: pd.DataFrame) -> None:
    for column in ("open", "high", "low", "close", "volume"):
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"invalid_{column}")
        out[column] = values
    if ((out[["open", "high", "low", "close"]] <= 0).any(axis=1)).any():
        raise ValueError("ohlc_must_be_positive")
    if (out["volume"] < 0).any():
        raise ValueError("volume_must_be_non_negative")
    if (out["high"] < out[["open", "close"]].max(axis=1)).any():
        raise ValueError("high_below_open_or_close")
    if (out["low"] > out[["open", "close"]].min(axis=1)).any():
        raise ValueError("low_above_open_or_close")


def _empty_label_frame(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    for column in (
        BUY_THE_DIP_ENTRY_LABEL,
        BUY_THE_DIP_EXIT_LABEL,
        BUY_THE_DIP_EXIT_RETURN_PCT,
        BUY_THE_DIP_EXIT_REASON,
        BUY_THE_DIP_HOLD_BARS,
        BUY_THE_DIP_TIMEFRAME,
        BUY_THE_DIP_CHRONOLOGICAL,
    ):
        out[column] = []
    return out
