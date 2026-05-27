from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from app.data.feature_engineering import FEATURE_COLUMNS
from app.ml.model import MLSignalModel


ZERO_BACKTEST_METRICS = {
    "number_of_trades": 0,
    "win_rate": 0.0,
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
    "starting_equity": None,
    "ending_equity": None,
}


def backtest_assumptions(settings: Settings, *, spread_available: bool = False) -> dict[str, Any]:
    fee_bps = settings.taker_fee_bps if settings.backtest_use_taker_fees else settings.maker_fee_bps
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
        "spread_source": "orderbook_spread" if spread_available else "unavailable",
        "spread_cost_model": "full_round_trip_spread_subtracted_when_available",
        "order_notional_usd": settings.order_notional_usd,
        "return_metrics_unit": "fraction_of_traded_notional",
        "paid_estimates_unit": "usd_using_order_notional_per_trade",
        "gross_return_model": "triple_barrier_take_profit_or_stop_loss",
    }


def calculate_fee_aware_metrics(trades: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    if trades.empty:
        return {"valid": False, "reason": "no_trades", **ZERO_BACKTEST_METRICS}
    if "buy_quality_label" not in trades.columns:
        return {"valid": False, "reason": "missing_buy_quality_label", **ZERO_BACKTEST_METRICS}

    labels = trades["buy_quality_label"].astype(int).to_numpy()
    take_profit_pct = _take_profit_pct(settings)
    stop_loss_pct = _stop_loss_pct(settings)
    gross_returns = np.where(labels == 1, take_profit_pct, -stop_loss_pct).astype(float)

    fee_bps = settings.taker_fee_bps if settings.backtest_use_taker_fees else settings.maker_fee_bps
    fee_costs = np.full(len(trades), 2 * (fee_bps / 10_000), dtype=float)
    slippage_costs = np.full(len(trades), 2 * (settings.slippage_bps / 10_000), dtype=float)
    spread_costs = _spread_costs(trades)

    net_returns = gross_returns - fee_costs - slippage_costs - spread_costs
    notional = float(settings.order_notional_usd)
    trade_pnls = net_returns * notional
    gross_return = float(gross_returns.sum())
    net_return = float(net_returns.sum())
    max_drawdown = _max_drawdown(net_returns)

    return {
        "valid": True,
        "number_of_trades": int(len(trades)),
        "win_rate": float((gross_returns > 0).mean()),
        "gross_return": gross_return,
        "net_return": net_return,
        "gross_return_pct": gross_return,
        "net_return_pct": net_return,
        "average_gross_win": _mean_or_zero(gross_returns[gross_returns > 0]),
        "average_gross_loss": _mean_or_zero(gross_returns[gross_returns < 0]),
        "average_net_win": _mean_or_zero(net_returns[net_returns > 0]),
        "average_net_loss": _mean_or_zero(net_returns[net_returns < 0]),
        "average_trade_pnl": _mean_or_zero(trade_pnls),
        "best_trade_pnl": float(trade_pnls.max()) if len(trade_pnls) else 0.0,
        "worst_trade_pnl": float(trade_pnls.min()) if len(trade_pnls) else 0.0,
        "profit_factor_gross": _profit_factor(gross_returns),
        "profit_factor_net": _profit_factor(net_returns),
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown,
        "fees_paid_estimate": float((fee_costs * notional).sum()),
        "slippage_paid_estimate": float((slippage_costs * notional).sum()),
        "spread_paid_estimate": float((spread_costs * notional).sum()),
        "starting_equity": notional,
        "ending_equity": notional * (1 + net_return),
    }


def walk_forward_fee_aware_backtest(
    df: pd.DataFrame,
    settings: Settings,
    *,
    min_train_rows: int,
    threshold: float,
    folds: int = 3,
) -> dict[str, Any]:
    if len(df) < min_train_rows:
        return {
            "valid": False,
            "reason": "not_enough_rows",
            "rows": int(len(df)),
            "min_train_rows": int(min_train_rows),
            **ZERO_BACKTEST_METRICS,
        }

    prediction_frame = _walk_forward_prediction_frame(df, min_train_rows=min_train_rows, folds=folds)
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


def _walk_forward_prediction_frame(df: pd.DataFrame, *, min_train_rows: int, folds: int) -> pd.DataFrame:
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
        model = MLSignalModel(feature_columns=FEATURE_COLUMNS).train(train)
        frame = valid.copy()
        frame["_probability"] = model.predict_proba(valid)
        validation_frames.append(frame)
    if not validation_frames:
        return pd.DataFrame()
    return pd.concat(validation_frames, ignore_index=True)


def _spread_costs(trades: pd.DataFrame) -> np.ndarray:
    if "orderbook_spread" not in trades.columns:
        return np.zeros(len(trades), dtype=float)
    spread = pd.to_numeric(trades["orderbook_spread"], errors="coerce").fillna(0).clip(lower=0)
    return spread.to_numpy(dtype=float)


def _mean_or_zero(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _profit_factor(returns: np.ndarray) -> float | None:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    if gross_loss == 0:
        return None if gross_win > 0 else 0.0
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
    return settings.scalping_take_profit_pct if settings.scalping_mode_enabled else settings.take_profit_pct


def _stop_loss_pct(settings: Settings) -> float:
    return settings.scalping_stop_loss_pct if settings.scalping_mode_enabled else settings.stop_loss_pct
