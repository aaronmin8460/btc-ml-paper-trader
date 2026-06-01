from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.broker.paper_execution import PaperFillResult, simulate_limit_ioc_order, simulate_market_order
from app.config import Settings
from app.data.feature_engineering import BAR_FEATURE_COLUMNS
from app.ml.model import MLSignalModel


ZERO_BACKTEST_METRICS = {
    "number_of_trades": 0,
    "number_of_canceled_orders": 0,
    "partial_fill_count": 0,
    "evaluated_signal_count": 0,
    "ambiguous_candle_count": 0,
    "ambiguous_candle_ratio": 0.0,
    "win_rate": 0.0,
    "win_rate_net": 0.0,
    "gross_return": 0.0,
    "net_return": 0.0,
    "gross_return_pct": 0.0,
    "net_return_pct": 0.0,
    "average_gross_win": 0.0,
    "average_gross_loss": 0.0,
    "average_net_win": 0.0,
    "average_net_loss": 0.0,
    "average_trade_pnl": 0.0,
    "best_trade_pnl": 0.0,
    "worst_trade_pnl": 0.0,
    "profit_factor_gross": 0.0,
    "profit_factor_net": 0.0,
    "max_drawdown": 0.0,
    "max_drawdown_pct": 0.0,
    "fees_paid_estimate": 0.0,
    "slippage_paid_estimate": 0.0,
    "spread_paid_estimate": 0.0,
    "total_fees": 0.0,
    "total_slippage": 0.0,
    "total_spread_cost": 0.0,
    "average_hold_bars": 0.0,
    "starting_equity": None,
    "ending_equity": None,
}


def backtest_assumptions(settings: Settings, *, spread_available: bool = False) -> dict[str, Any]:
    fee_bps = settings.taker_fee_bps if settings.backtest_use_taker_fees else settings.maker_fee_bps
    spread_source = "unavailable"
    if spread_available:
        spread_source = "scalping_spread_pct" if settings.scalping_mode_enabled else "orderbook_spread"
    if not spread_available and settings.scalping_mode_enabled:
        spread_source = "configured_max_spread_bps"
    return {
        "symbol": settings.symbol,
        "paper_trading_only": settings.paper_trading_only,
        "long_only": True,
        "fee_model": "taker" if settings.backtest_use_taker_fees else "maker",
        "fee_bps_per_side": fee_bps,
        "taker_fee_bps": settings.taker_fee_bps,
        "maker_fee_bps": settings.maker_fee_bps,
        "slippage_bps_per_side": settings.slippage_bps,
        "fee_and_slippage_applied_on": "entry_and_exit",
        "spread_source": spread_source,
        "spread_cost_model": "simulated_touch_price_half_spread_paid_per_fill",
        "execution_model": "shared_local_paper_execution_simulator",
        "limit_order_behavior": "ioc_cancel_or_partial_fill",
        "partial_fill_scoring": "matched_quantity_return_with_all_simulated_fill_costs_charged",
        "ambiguous_candle_behavior": "stop_loss_first",
        "hold_duration_model": "row_hold_bars_when_available_else_label_horizon",
        "order_notional_usd": settings.order_notional_usd,
        "return_metrics_unit": "fraction_of_traded_notional",
        "paid_estimates_unit": "usd_from_simulated_fills",
        "gross_return_model": (
            "net_profit_scalping_labels" if settings.scalping_mode_enabled else "triple_barrier_take_profit_or_stop_loss"
        ),
    }


def calculate_fee_aware_metrics(trades: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    if trades.empty:
        return {"valid": False, "reason": "no_trades", **ZERO_BACKTEST_METRICS}
    if "buy_quality_label" not in trades.columns:
        return {"valid": False, "reason": "missing_buy_quality_label", **ZERO_BACKTEST_METRICS}

    take_profit_pct = _take_profit_pct(settings)
    stop_loss_pct = _stop_loss_pct(settings)
    fee_bps = settings.taker_fee_bps if settings.backtest_use_taker_fees else settings.maker_fee_bps
    notional = float(settings.order_notional_usd)
    gross_returns: list[float] = []
    net_trade_returns: list[float] = []
    equity_returns: list[float] = []
    hold_bars: list[float] = []
    total_fees = 0.0
    total_slippage = 0.0
    total_spread_cost = 0.0
    canceled_orders = 0
    partial_fills = 0
    ambiguous_candles = 0

    for _, row in trades.iterrows():
        entry_reference = _entry_reference_price(row)
        gross_exit_return, ambiguous = _resolved_gross_return(
            row,
            entry_reference=entry_reference,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
        )
        ambiguous_candles += int(ambiguous)
        entry_fill = _simulate_backtest_order(
            row,
            settings,
            side="buy",
            reference_price=entry_reference,
            fee_bps=fee_bps,
            notional=notional,
        )
        total_fees += entry_fill.fee_amount
        total_slippage += entry_fill.slippage_amount
        total_spread_cost += entry_fill.spread_cost_amount
        canceled_orders += int(entry_fill.status == "canceled")
        partial_fills += int(entry_fill.status == "partially_filled")
        if entry_fill.filled_qty <= 0:
            continue

        exit_reference = entry_reference * (1 + gross_exit_return)
        exit_fill = _simulate_backtest_order(
            row,
            settings,
            side="sell",
            reference_price=exit_reference,
            fee_bps=fee_bps,
            qty=entry_fill.filled_qty,
        )
        total_fees += exit_fill.fee_amount
        total_slippage += exit_fill.slippage_amount
        total_spread_cost += exit_fill.spread_cost_amount
        canceled_orders += int(exit_fill.status == "canceled")
        partial_fills += int(exit_fill.status == "partially_filled")

        entry_cost = entry_fill.fee_amount + entry_fill.slippage_amount + entry_fill.spread_cost_amount
        if exit_fill.filled_qty <= 0:
            if entry_cost > 0:
                equity_returns.append(-entry_cost / notional)
            continue

        matched_qty = min(entry_fill.filled_qty, exit_fill.filled_qty)
        filled_fraction = matched_qty / entry_fill.requested_qty
        trade_gross_return = gross_exit_return * filled_fraction
        trade_cost = entry_cost + exit_fill.fee_amount + exit_fill.slippage_amount + exit_fill.spread_cost_amount
        trade_net_return = trade_gross_return - trade_cost / notional
        gross_returns.append(trade_gross_return)
        net_trade_returns.append(trade_net_return)
        equity_returns.append(trade_net_return)
        hold_bars.append(_hold_bars(row, settings))

    gross_values = np.asarray(gross_returns, dtype=float)
    net_trade_values = np.asarray(net_trade_returns, dtype=float)
    equity_values = np.asarray(equity_returns, dtype=float)
    trade_pnls = net_trade_values * notional
    gross_return = float(gross_values.sum())
    net_return = gross_return - (total_fees + total_slippage + total_spread_cost) / notional
    max_drawdown = _max_drawdown(equity_values)
    number_of_trades = len(gross_values)
    metrics = {
        "valid": number_of_trades > 0,
        "reason": None if number_of_trades > 0 else "no_filled_trades",
        "number_of_trades": int(number_of_trades),
        "number_of_canceled_orders": canceled_orders,
        "partial_fill_count": partial_fills,
        "evaluated_signal_count": int(len(trades)),
        "ambiguous_candle_count": ambiguous_candles,
        "ambiguous_candle_ratio": ambiguous_candles / len(trades),
        "win_rate": float((gross_values > 0).mean()) if number_of_trades else 0.0,
        "win_rate_net": float((net_trade_values > 0).mean()) if number_of_trades else 0.0,
        "gross_return": gross_return,
        "net_return": net_return,
        "gross_return_pct": gross_return,
        "net_return_pct": net_return,
        "average_gross_win": _mean_or_zero(gross_values[gross_values > 0]),
        "average_gross_loss": _mean_or_zero(gross_values[gross_values < 0]),
        "average_net_win": _mean_or_zero(net_trade_values[net_trade_values > 0]),
        "average_net_loss": _mean_or_zero(net_trade_values[net_trade_values < 0]),
        "average_trade_pnl": _mean_or_zero(trade_pnls),
        "best_trade_pnl": float(trade_pnls.max()) if len(trade_pnls) else 0.0,
        "worst_trade_pnl": float(trade_pnls.min()) if len(trade_pnls) else 0.0,
        "profit_factor_gross": _profit_factor(gross_values),
        "profit_factor_net": _profit_factor(equity_values),
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown,
        "fees_paid_estimate": total_fees,
        "slippage_paid_estimate": total_slippage,
        "spread_paid_estimate": total_spread_cost,
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "total_spread_cost": total_spread_cost,
        "average_hold_bars": _mean_or_zero(np.asarray(hold_bars, dtype=float)),
        "starting_equity": notional,
        "ending_equity": notional * (1 + net_return),
    }

    return metrics


def walk_forward_fee_aware_backtest(
    df: pd.DataFrame,
    settings: Settings,
    *,
    min_train_rows: int,
    threshold: float,
    folds: int = 3,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    if len(df) < min_train_rows:
        return {
            "valid": False,
            "reason": "not_enough_rows",
            "rows": int(len(df)),
            "min_train_rows": int(min_train_rows),
            **ZERO_BACKTEST_METRICS,
        }

    prediction_frame = _walk_forward_prediction_frame(
        df,
        min_train_rows=min_train_rows,
        folds=folds,
        feature_columns=feature_columns,
    )
    if prediction_frame.empty:
        return {
            "valid": False,
            "reason": "no_validation_folds",
            "rows": int(len(df)),
            "min_train_rows": int(min_train_rows),
            **ZERO_BACKTEST_METRICS,
        }

    trades = prediction_frame.loc[prediction_frame["_probability"] >= threshold].copy()
    metrics = calculate_fee_aware_metrics(trades, settings)
    metrics["rows"] = int(len(df))
    metrics["validation_rows"] = int(len(prediction_frame))
    metrics["threshold"] = float(threshold)
    if not metrics["valid"] and metrics.get("reason") == "no_trades":
        metrics["reason"] = "no_trades_above_threshold"
    return metrics


def _walk_forward_prediction_frame(
    df: pd.DataFrame,
    *,
    min_train_rows: int,
    folds: int,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or BAR_FEATURE_COLUMNS
    fold_size = max(50, (len(df) - min_train_rows) // max(1, folds))
    validation_frames: list[pd.DataFrame] = []
    for fold in range(folds):
        train_end = min_train_rows + fold * fold_size
        valid_end = min(len(df), train_end + fold_size)
        if valid_end <= train_end + 10:
            continue
        train = df.iloc[:train_end]
        valid = df.iloc[train_end:valid_end]
        if "buy_quality_label" in train.columns and train["buy_quality_label"].astype(int).nunique() < 2:
            continue
        model = MLSignalModel(feature_columns=feature_columns).train(train)
        frame = valid.copy()
        frame["_probability"] = model.predict_proba(valid)
        validation_frames.append(frame)
    if not validation_frames:
        return pd.DataFrame()
    return pd.concat(validation_frames, ignore_index=True)


def _simulate_backtest_order(
    row: pd.Series,
    settings: Settings,
    *,
    side: str,
    reference_price: float,
    fee_bps: float,
    notional: float | None = None,
    qty: float | None = None,
) -> PaperFillResult:
    quote = _execution_quote(row, settings, side=side, reference_price=reference_price)
    kwargs = {
        "symbol": settings.symbol,
        "side": side,
        "bid_price": quote["bid_price"],
        "ask_price": quote["ask_price"],
        "bid_size": quote["bid_size"],
        "ask_size": quote["ask_size"],
        "latest_price": reference_price,
        "notional": notional,
        "qty": qty,
        "fee_bps": fee_bps,
        "slippage_bps": settings.slippage_bps,
    }
    if settings.order_type == "limit":
        touch_price = quote["ask_price"] if side == "buy" else quote["bid_price"]
        kwargs["limit_price"] = _backtest_limit_price(row, settings, side=side, touch_price=touch_price)
        return simulate_limit_ioc_order(**kwargs)
    return simulate_market_order(**kwargs)


def _execution_quote(
    row: pd.Series,
    settings: Settings,
    *,
    side: str,
    reference_price: float,
) -> dict[str, float | None]:
    phase = "entry" if side == "buy" else "exit"
    bid_price = _row_positive_float(row, f"{phase}_bid_price")
    ask_price = _row_positive_float(row, f"{phase}_ask_price")
    if phase == "entry":
        bid_price = bid_price or _row_positive_float(row, "bid_price")
        ask_price = ask_price or _row_positive_float(row, "ask_price")

    spread_pct = _row_non_negative_float(row, f"{phase}_spread_pct", "scalping_spread_pct", "orderbook_spread")
    if spread_pct == 0 and settings.scalping_mode_enabled:
        spread_pct = None
    observed_spread_pct = _observed_quote_spread_pct(row)
    spread_pct = observed_spread_pct if observed_spread_pct is not None else spread_pct
    if bid_price is not None and ask_price is not None and ask_price >= bid_price:
        spread_pct = (ask_price - bid_price) / ((ask_price + bid_price) / 2)
    else:
        spread_pct = spread_pct if spread_pct is not None else _fallback_spread_pct(settings)
        bid_price = reference_price * (1 - spread_pct / 2)
        ask_price = reference_price * (1 + spread_pct / 2)

    bid_size = _row_non_negative_float(row, f"{phase}_bid_size")
    ask_size = _row_non_negative_float(row, f"{phase}_ask_size")
    bid_size = bid_size if bid_size is not None else _row_non_negative_float(row, "bid_size")
    ask_size = ask_size if ask_size is not None else _row_non_negative_float(row, "ask_size")
    return {
        "bid_price": bid_price,
        "ask_price": ask_price,
        "bid_size": bid_size,
        "ask_size": ask_size,
    }


def _backtest_limit_price(row: pd.Series, settings: Settings, *, side: str, touch_price: float) -> float:
    phase = "entry" if side == "buy" else "exit"
    explicit = _row_positive_float(row, f"{phase}_limit_price")
    if explicit is None and side == "buy":
        explicit = _row_positive_float(row, "limit_price")
    if explicit is not None:
        return explicit
    offset = max(0.0, float(settings.limit_price_offset_bps)) / 10_000
    return touch_price * (1 + offset) if side == "buy" else touch_price * (1 - offset)


def _resolved_gross_return(
    row: pd.Series,
    *,
    entry_reference: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> tuple[float, bool]:
    reason = str(row.get("buy_exit_reason") or "")
    ambiguous = reason == "ambiguous_stop_first" or _explicit_exit_bar_is_ambiguous(
        row,
        entry_reference=entry_reference,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )
    if ambiguous:
        return -abs(stop_loss_pct), True

    label = int(row.get("buy_quality_label", 0))
    fallback = take_profit_pct if label == 1 else -abs(stop_loss_pct)
    return _row_float(row, "buy_exit_return_pct") or fallback, False


def _explicit_exit_bar_is_ambiguous(
    row: pd.Series,
    *,
    entry_reference: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> bool:
    high = _row_positive_float(row, "backtest_exit_high")
    low = _row_positive_float(row, "backtest_exit_low")
    if high is None or low is None:
        return False
    return high >= entry_reference * (1 + take_profit_pct) and low <= entry_reference * (1 - abs(stop_loss_pct))


def _entry_reference_price(row: pd.Series) -> float:
    return _row_positive_float(row, "close", "latest_price") or 100.0


def _fallback_spread_pct(settings: Settings) -> float:
    return max(0.0, float(settings.max_spread_bps)) / 10_000 if settings.scalping_mode_enabled else 0.0


def _observed_quote_spread_pct(row: pd.Series) -> float | None:
    bid_price = _row_positive_float(row, "bid_price")
    ask_price = _row_positive_float(row, "ask_price")
    if bid_price is None or ask_price is None or ask_price < bid_price:
        return None
    return (ask_price - bid_price) / ((ask_price + bid_price) / 2)


def _hold_bars(row: pd.Series, settings: Settings) -> float:
    value = _row_positive_float(row, "buy_hold_bars", "hold_bars")
    if value is not None:
        return value
    return float(settings.scalping_label_horizon_bars if settings.scalping_mode_enabled else 12)


def _row_positive_float(row: pd.Series, *names: str) -> float | None:
    for name in names:
        value = _row_float(row, name)
        if value is not None and value > 0:
            return value
    return None


def _row_non_negative_float(row: pd.Series, *names: str) -> float | None:
    for name in names:
        value = _row_float(row, name)
        if value is not None and value >= 0:
            return value
    return None


def _row_float(row: pd.Series, name: str) -> float | None:
    try:
        value = float(row.get(name))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _mean_or_zero(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _profit_factor(returns: np.ndarray) -> float | None:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return float(gross_win / gross_loss)


def _max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.concatenate(([0.0], np.cumsum(returns)))
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    value = abs(float(drawdown.min()))
    return value if math.isfinite(value) else 0.0


def _take_profit_pct(settings: Settings) -> float:
    return settings.scalping_label_take_profit_pct if settings.scalping_mode_enabled else settings.take_profit_pct


def _stop_loss_pct(settings: Settings) -> float:
    return settings.scalping_label_stop_loss_pct if settings.scalping_mode_enabled else settings.stop_loss_pct
