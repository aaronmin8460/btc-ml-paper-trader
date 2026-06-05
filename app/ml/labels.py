import numpy as np
import pandas as pd


ENTRY_QUALITY_LABEL = "entry_quality_label"
BUY_QUALITY_LABEL = "buy_quality_label"
EXIT_QUALITY_LABEL = "exit_quality_label"
SELL_QUALITY_LABEL = "sell_quality_label"


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon_bars: int = 12,
    take_profit_pct: float = 0.03,
    stop_loss_pct: float = 0.015,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    entry_labels: list[int | float] = []
    exit_labels: list[int | float] = []
    for i in range(len(out)):
        if i + horizon_bars >= len(out):
            entry_labels.append(np.nan)
            exit_labels.append(np.nan)
            continue
        entry = out.loc[i, "close"]
        tp = entry * (1 + take_profit_pct)
        sl = entry * (1 - stop_loss_pct)
        entry_label = 0
        exit_label = 0
        future = out.iloc[i + 1 : i + horizon_bars + 1]
        for _, row in future.iterrows():
            hit_tp = row["high"] >= tp
            hit_sl = row["low"] <= sl
            if hit_tp and hit_sl:
                entry_label = 0
                exit_label = 1
                break
            if hit_tp:
                entry_label = 1
                exit_label = 0
                break
            if hit_sl:
                entry_label = 0
                exit_label = 1
                break
        entry_labels.append(entry_label)
        exit_labels.append(exit_label)
    _assign_entry_exit_label_aliases(out, entry_labels=entry_labels, exit_labels=exit_labels)
    return out.dropna(subset=[BUY_QUALITY_LABEL, EXIT_QUALITY_LABEL]).reset_index(drop=True)


def net_profit_scalping_labels(
    df: pd.DataFrame,
    *,
    horizon_bars: int = 3,
    take_profit_pct: float = 0.0012,
    stop_loss_pct: float = 0.0008,
    trailing_stop_pct: float = 0.0008,
    trailing_stop_arm_profit_pct: float = 0.002,
    fee_bps_per_side: float = 0.0,
    slippage_bps_per_side: float = 0.0,
    spread_cost_pct: float = 0.0,
    min_net_exit_profit_pct: float = 0.0,
    exit_profit_buffer_bps: float = 0.0,
) -> pd.DataFrame:
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")

    out = df.copy().reset_index(drop=True)
    entry_labels: list[int | float] = []
    exit_labels: list[int | float] = []
    exit_returns: list[float] = []
    exit_reasons: list[str | None] = []
    hold_bars: list[int | float] = []

    for i in range(len(out)):
        if i + horizon_bars >= len(out):
            entry_labels.append(np.nan)
            exit_labels.append(np.nan)
            exit_returns.append(np.nan)
            exit_reasons.append(None)
            hold_bars.append(np.nan)
            continue

        entry = float(out.loc[i, "close"])
        if not np.isfinite(entry) or entry <= 0:
            entry_labels.append(0)
            exit_labels.append(0)
            exit_returns.append(-abs(stop_loss_pct))
            exit_reasons.append("invalid_entry")
            hold_bars.append(0)
            continue

        required_exit_return = _required_net_scalping_exit_return(
            out.iloc[i],
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
            spread_cost_pct=spread_cost_pct,
            min_net_exit_profit_pct=min_net_exit_profit_pct,
            exit_profit_buffer_bps=exit_profit_buffer_bps,
        )
        take_profit_return = max(float(take_profit_pct), required_exit_return)
        stop_loss_price = entry * (1 - abs(float(stop_loss_pct)))
        take_profit_price = entry * (1 + take_profit_return)
        highest = entry
        entry_label = 0
        exit_label = 0
        exit_return = -abs(float(stop_loss_pct))
        exit_reason = "no_profitable_exit"
        bars_to_exit = horizon_bars

        future = out.iloc[i + 1 : i + horizon_bars + 1]
        for offset, (_, row) in enumerate(future.iterrows(), start=1):
            high = float(row["high"])
            low = float(row["low"])
            if not np.isfinite(high) or not np.isfinite(low):
                continue

            hit_stop_loss = low <= stop_loss_price
            hit_take_profit = high >= take_profit_price
            highest = max(highest, high)
            trailing_exit_return = _trailing_exit_return(
                entry=entry,
                highest=highest,
                trailing_stop_pct=trailing_stop_pct,
                trailing_stop_arm_profit_pct=trailing_stop_arm_profit_pct,
            )
            hit_trailing_exit = trailing_exit_return is not None and (
                trailing_exit_return >= required_exit_return
                and low <= entry * (1 + trailing_exit_return)
            )

            if hit_stop_loss and (hit_take_profit or hit_trailing_exit):
                exit_label = 1
                exit_reason = "ambiguous_stop_first"
                bars_to_exit = offset
                break
            if hit_take_profit:
                entry_label = 1
                exit_return = take_profit_return
                exit_reason = "scalping_take_profit"
                bars_to_exit = offset
                break
            if hit_trailing_exit and trailing_exit_return is not None:
                entry_label = 1
                exit_return = trailing_exit_return
                exit_reason = "scalping_trailing_stop"
                bars_to_exit = offset
                break
            if hit_stop_loss:
                exit_label = 1
                exit_reason = "scalping_stop_loss"
                bars_to_exit = offset
                break

        entry_labels.append(entry_label)
        exit_labels.append(exit_label)
        exit_returns.append(exit_return)
        exit_reasons.append(exit_reason)
        hold_bars.append(bars_to_exit)

    _assign_entry_exit_label_aliases(out, entry_labels=entry_labels, exit_labels=exit_labels)
    out["buy_exit_return_pct"] = exit_returns
    out["buy_exit_reason"] = exit_reasons
    out["buy_hold_bars"] = hold_bars
    return out.dropna(subset=[BUY_QUALITY_LABEL, EXIT_QUALITY_LABEL]).reset_index(drop=True)


def _assign_entry_exit_label_aliases(
    out: pd.DataFrame,
    *,
    entry_labels: list[int | float],
    exit_labels: list[int | float],
) -> None:
    out[ENTRY_QUALITY_LABEL] = entry_labels
    out[BUY_QUALITY_LABEL] = entry_labels
    out[EXIT_QUALITY_LABEL] = exit_labels
    # Compatibility alias: in this long-only bot, "sell" means exit an existing long.
    out[SELL_QUALITY_LABEL] = exit_labels


def _required_net_scalping_exit_return(
    row: pd.Series,
    *,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
    spread_cost_pct: float,
    min_net_exit_profit_pct: float,
    exit_profit_buffer_bps: float,
) -> float:
    row_spread = _positive_float(row.get("scalping_spread_pct"))
    if row_spread is None:
        row_spread = _positive_float(row.get("orderbook_spread"))
    spread = row_spread if row_spread is not None else max(0.0, float(spread_cost_pct))
    return (
        2 * max(0.0, float(fee_bps_per_side)) / 10_000
        + 2 * max(0.0, float(slippage_bps_per_side)) / 10_000
        + max(0.0, spread)
        + max(0.0, float(min_net_exit_profit_pct))
        + max(0.0, float(exit_profit_buffer_bps)) / 10_000
    )


def _trailing_exit_return(
    *,
    entry: float,
    highest: float,
    trailing_stop_pct: float,
    trailing_stop_arm_profit_pct: float,
) -> float | None:
    if trailing_stop_pct <= 0 or entry <= 0:
        return None
    if highest < entry * (1 + max(0.0, float(trailing_stop_arm_profit_pct))):
        return None
    trailing_exit_price = highest * (1 - max(0.0, float(trailing_stop_pct)))
    return (trailing_exit_price - entry) / entry


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) and parsed > 0 else None
